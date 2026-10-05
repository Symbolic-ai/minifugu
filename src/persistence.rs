//! Durable local storage: a JSON snapshot plus an append-only log of row changes.
//!
//! An acknowledged write appends one line to `namespaces.log` and syncs it. The line holds
//! the namespace metadata and only the rows that the write changed, so the cost of a write
//! follows the size of the change, not the size of the store. Startup replays the log over
//! `namespaces.json` and compacts both into a new snapshot. The log is also compacted while
//! the server runs, once it is larger than the snapshot.

use crate::store::{same_row, Namespace, Row, Rows};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const SNAPSHOT: &str = "namespaces.json";
const LOG: &str = "namespaces.log";
/// A smaller log is never compacted, so a small store does not rewrite its snapshot
/// every few writes.
const MIN_COMPACTION_BYTES: u64 = 16 * 1024 * 1024;

/// One log line as written. It borrows the rows so that a write does not copy them.
#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum RecordOut<'a> {
    Put {
        name: &'a str,
        meta: &'a Namespace,
        replace: bool,
        upsert: BTreeMap<&'a str, &'a Arc<Row>>,
        delete: Vec<&'a str>,
    },
    Drop {
        name: &'a str,
    },
}

/// One log line as read back.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum RecordIn {
    Put {
        name: String,
        meta: Box<Namespace>,
        #[serde(default)]
        replace: bool,
        #[serde(default)]
        upsert: BTreeMap<String, Arc<Row>>,
        #[serde(default)]
        delete: Vec<String>,
    },
    Drop {
        name: String,
    },
}

pub(crate) struct Store {
    directory: PathBuf,
    log: File,
    log_bytes: u64,
    snapshot_bytes: u64,
    min_compaction_bytes: u64,
}

/// Loads the snapshot, replays the log, and compacts both into a new snapshot.
///
/// A corrupt snapshot or a corrupt complete log line stops startup instead of dropping
/// data. A final log line without its newline was never acknowledged, so it is discarded.
pub(crate) fn open(directory: &Path) -> io::Result<(Store, HashMap<String, Namespace>)> {
    fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    }
    let snapshot_path = directory.join(SNAPSHOT);
    let (mut namespaces, mut snapshot_bytes): (HashMap<String, Namespace>, u64) =
        match fs::read(&snapshot_path) {
            Ok(bytes) => (
                serde_json::from_slice(&bytes).map_err(io::Error::other)?,
                bytes.len() as u64,
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => (HashMap::new(), 0),
            Err(error) => return Err(error),
        };
    let replayed = replay(&directory.join(LOG), &mut namespaces)?;
    // Older snapshots lack the cached byte estimate; compute it once on startup.
    for namespace in namespaces.values_mut() {
        if namespace.approx_logical_bytes.is_none() {
            namespace.approx_logical_bytes = Some(namespace.logical_bytes());
        }
    }
    if replayed {
        snapshot_bytes = write_snapshot(directory, &namespaces)?;
    }
    // The snapshot now holds every logged change, so the log starts empty.
    let log = open_log(directory)?;
    Ok((
        Store {
            directory: directory.to_path_buf(),
            log,
            log_bytes: 0,
            snapshot_bytes,
            min_compaction_bytes: MIN_COMPACTION_BYTES,
        },
        namespaces,
    ))
}

impl Store {
    /// Durably records that `name` changed from `old` to `new`.
    pub(crate) fn put(
        &mut self,
        name: &str,
        old: Option<&Namespace>,
        new: &Namespace,
    ) -> io::Result<()> {
        let (replace, upsert, delete) = match old {
            None => (
                true,
                new.rows
                    .iter()
                    .map(|(id, row)| (id.as_str(), row))
                    .collect(),
                Vec::new(),
            ),
            Some(old) => {
                let (upsert, delete) = row_changes(&old.rows, &new.rows);
                (false, upsert, delete)
            }
        };
        let meta = metadata(new);
        self.append(&RecordOut::Put {
            name,
            meta: &meta,
            replace,
            upsert,
            delete,
        })
    }

    /// Durably records that `name` was deleted.
    pub(crate) fn drop_namespace(&mut self, name: &str) -> io::Result<()> {
        self.append(&RecordOut::Drop { name })
    }

    /// Rewrites the snapshot and empties the log once the log outgrows the snapshot.
    ///
    /// The last write is already durable in the log, so a failure here loses nothing.
    pub(crate) fn maybe_compact(
        &mut self,
        namespaces: &HashMap<String, Namespace>,
    ) -> io::Result<()> {
        if self.log_bytes < self.min_compaction_bytes.max(self.snapshot_bytes) {
            return Ok(());
        }
        self.snapshot_bytes = write_snapshot(&self.directory, namespaces)?;
        // A crash before this truncation replays the log over a snapshot that already
        // contains it. Each record assigns absolute values in order, so the result is
        // the same state.
        self.log.set_len(0)?;
        self.log.sync_all()?;
        self.log_bytes = 0;
        Ok(())
    }

    fn append(&mut self, record: &RecordOut<'_>) -> io::Result<()> {
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        // The log is written at a tracked offset rather than in append mode: Windows
        // refuses to truncate a file opened for appending.
        let result = self
            .log
            .seek(SeekFrom::Start(self.log_bytes))
            .and_then(|_| self.log.write_all(&line))
            .and_then(|()| self.log.sync_data());
        match result {
            Ok(()) => {
                self.log_bytes += line.len() as u64;
                Ok(())
            }
            Err(error) => {
                // Remove a partial line, or the next acknowledged line would follow it
                // and the log could not be replayed.
                let _ = self.log.set_len(self.log_bytes);
                Err(error)
            }
        }
    }
}

fn row_changes<'a>(
    old: &'a Rows,
    new: &'a Rows,
) -> (BTreeMap<&'a str, &'a Arc<Row>>, Vec<&'a str>) {
    let upsert = new
        .iter()
        .filter(|(id, row)| !same_row(old.get(*id), row))
        .map(|(id, row)| (id.as_str(), row))
        .collect();
    let delete = old
        .keys()
        .filter(|id| !new.contains_key(*id))
        .map(String::as_str)
        .collect();
    (upsert, delete)
}

/// The namespace without its rows: what a log record needs besides the row changes.
fn metadata(namespace: &Namespace) -> Namespace {
    Namespace {
        schema: namespace.schema.clone(),
        rows: BTreeMap::new(),
        distance_metric: namespace.distance_metric.clone(),
        created_at: namespace.created_at,
        updated_at: namespace.updated_at,
        last_write_at: namespace.last_write_at,
        approx_logical_bytes: namespace.approx_logical_bytes,
        read_only: namespace.read_only,
        cmek_key_name: namespace.cmek_key_name.clone(),
    }
}

/// Applies the log to `namespaces` and returns whether it held any record.
fn replay(path: &Path, namespaces: &mut HashMap<String, Namespace>) -> io::Result<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut replayed = false;
    let mut lines = bytes.split(|byte| *byte == b'\n').peekable();
    while let Some(line) = lines.next() {
        // The segment after the last newline is empty, or a line that was cut off
        // before its write was acknowledged.
        if lines.peek().is_none() {
            break;
        }
        let record: RecordIn = serde_json::from_slice(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("corrupt {LOG}: {error}"),
            )
        })?;
        apply(namespaces, record);
        replayed = true;
    }
    Ok(replayed)
}

fn apply(namespaces: &mut HashMap<String, Namespace>, record: RecordIn) {
    match record {
        RecordIn::Put {
            name,
            meta,
            replace,
            upsert,
            delete,
        } => {
            let mut meta = *meta;
            let entry = namespaces.entry(name).or_default();
            meta.rows = if replace {
                BTreeMap::new()
            } else {
                std::mem::take(&mut entry.rows)
            };
            for id in &delete {
                meta.rows.remove(id);
            }
            meta.rows.extend(upsert);
            *entry = meta;
        }
        RecordIn::Drop { name } => {
            namespaces.remove(&name);
        }
    }
}

fn write_snapshot(directory: &Path, namespaces: &HashMap<String, Namespace>) -> io::Result<u64> {
    let bytes = serde_json::to_vec(namespaces).map_err(io::Error::other)?;
    let path = directory.join(SNAPSHOT);
    let temporary = path.with_extension("json.tmp");
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    sync_directory(directory)?;
    Ok(bytes.len() as u64)
}

/// Opens the log empty.
fn open_log(directory: &Path) -> io::Result<File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(directory.join(LOG))?;
    file.sync_all()?;
    sync_directory(directory)?;
    Ok(file)
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn namespace(ids: &[&str]) -> Namespace {
        Namespace {
            rows: ids
                .iter()
                .map(|id| {
                    (
                        id.to_string(),
                        Arc::new(json!({"id": id}).as_object().unwrap().clone()),
                    )
                })
                .collect(),
            ..Namespace::default()
        }
    }

    #[test]
    fn row_changes_lists_only_changed_and_deleted_rows() {
        let old = namespace(&["1", "2", "3", "5"]).rows;
        // A clone shares every row; the write then copies, adds and removes some.
        let mut new = old.clone();
        Arc::make_mut(new.get_mut("2").unwrap()).insert("title".into(), json!("changed"));
        // An equal row in a new allocation is still unchanged.
        new.insert("5".into(), Arc::new(old["5"].as_ref().clone()));
        new.remove("3");
        new.extend(namespace(&["4"]).rows);
        let (upsert, delete) = row_changes(&old, &new);
        assert_eq!(upsert.keys().copied().collect::<Vec<_>>(), ["2", "4"]);
        assert_eq!(delete, ["3"]);
    }

    #[test]
    fn a_log_larger_than_the_snapshot_is_compacted() {
        let directory = tempfile::tempdir().unwrap();
        let (mut store, mut namespaces) = open(directory.path()).unwrap();
        store.min_compaction_bytes = 0;
        let first = namespace(&["1"]);
        store.put("ns", None, &first).unwrap();
        namespaces.insert("ns".into(), first);
        store.maybe_compact(&namespaces).unwrap();
        assert_eq!(fs::metadata(directory.path().join(LOG)).unwrap().len(), 0);

        // The next append lands at the start of the emptied log, so it replays cleanly.
        let second = namespace(&["1", "2"]);
        store.put("ns", namespaces.get("ns"), &second).unwrap();
        let log = fs::read(directory.path().join(LOG)).unwrap();
        assert_eq!(log.first(), Some(&b'{'));
        drop(store);
        let (_store, reopened) = open(directory.path()).unwrap();
        assert_eq!(reopened["ns"].rows.keys().collect::<Vec<_>>(), ["1", "2"]);
    }

    #[test]
    fn a_small_log_is_not_compacted() {
        let directory = tempfile::tempdir().unwrap();
        let (mut store, mut namespaces) = open(directory.path()).unwrap();
        let first = namespace(&["1"]);
        store.put("ns", None, &first).unwrap();
        namespaces.insert("ns".into(), first);
        store.maybe_compact(&namespaces).unwrap();
        assert!(fs::metadata(directory.path().join(LOG)).unwrap().len() > 0);
        assert!(!directory.path().join(SNAPSHOT).exists());
    }
}
