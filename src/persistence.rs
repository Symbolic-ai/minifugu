use crate::store::Namespace;
use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

pub(crate) fn open(directory: &Path) -> io::Result<(PathBuf, HashMap<String, Namespace>)> {
    fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    }
    let path = directory.join("namespaces.json");
    let namespaces = match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(io::Error::other)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => HashMap::new(),
        Err(error) => return Err(error),
    };
    Ok((path, namespaces))
}

pub(crate) fn save(path: &Path, namespaces: &HashMap<String, Namespace>) -> io::Result<()> {
    let bytes = serde_json::to_vec(namespaces).map_err(io::Error::other)?;
    let temporary = path.with_extension("json.tmp");
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary)?;
    use std::io::Write;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    #[cfg(unix)]
    fs::File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}
