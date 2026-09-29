//! Full-text analysis. `word_v3` and `word_v4` use alyze, the analyzer the live service runs
//! for `word_v4`. The older `word_v0` to `word_v2` tokenizers segment text by character
//! class, then pass each segment through the same filters.

use alyze::analyze::{
    AnalysisOptions, Analyzer, LanguageWithStopwords, ReusableBuffer, StemmingLanguage,
    StopwordRemoval, TokenizerOptions,
};
use serde_json::Value;
use std::{cell::RefCell, collections::HashMap, ops::Range, rc::Rc};

/// A normalized token and where it came from in the attribute value.
#[derive(Clone, Debug)]
pub(crate) struct Token {
    pub text: String,
    /// Every word-like segment consumes a position, including the ones a filter drops, so
    /// phrase distances survive stopword removal.
    pub position: usize,
    pub byte_range: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tokenizer {
    Word(u8),
    PreTokenized,
}

const LANGUAGES: [&str; 18] = [
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hungarian",
    "italian",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
];
const DEFAULT_MAX_TOKEN_LENGTH: usize = 39;
/// Distinct tokenizations kept per thread before the cache starts over.
const TOKEN_CACHE_LIMIT: usize = 100_000;

/// Tokenizations keyed by analysis settings and the joined input text.
type TokenCache = HashMap<(Rc<str>, String), Rc<Vec<Token>>>;

/// The analysis settings of one `full_text_search` attribute.
#[derive(Clone, Debug)]
pub(crate) struct TextAnalysis {
    tokenizer: Tokenizer,
    options: AnalysisOptions,
    key: Rc<str>,
}

thread_local! {
    /// One buffer per stemming language: alyze caches stems in the buffer without the
    /// language, so a shared buffer would reuse English stems for French text.
    static BUFFERS: RefCell<HashMap<String, ReusableBuffer>> = RefCell::new(HashMap::new());
    static TOKENS: RefCell<TokenCache> = RefCell::new(HashMap::new());
}

/// Drops cached tokenizations. Call at the start of every request that reads rows, so a
/// cache never outlives the data it describes.
pub(crate) fn reset_cache() {
    TOKENS.with(|cache| cache.borrow_mut().clear());
}

impl TextAnalysis {
    /// The analysis for a schema definition, or `None` when full-text search is off.
    pub fn for_field(definition: &Value) -> Option<Self> {
        match definition.get("full_text_search") {
            Some(Value::Bool(true)) => Some(Self::from_config(&Value::Null)),
            Some(config @ Value::Object(_)) => Some(Self::from_config(config)),
            _ => None,
        }
    }

    /// Builds the analysis from a validated configuration object. `Null` means defaults.
    fn from_config(config: &Value) -> Self {
        let tokenizer = match config.get("tokenizer").and_then(Value::as_str) {
            Some("pre_tokenized_array") => Tokenizer::PreTokenized,
            Some(name) => Tokenizer::Word(
                name.strip_prefix("word_v")
                    .and_then(|version| version.parse().ok())
                    .unwrap_or(4),
            ),
            None => Tokenizer::Word(4),
        };
        let flag = |key| config.get(key).and_then(Value::as_bool).unwrap_or(false);
        let case_sensitive = tokenizer == Tokenizer::PreTokenized
            || config
                .get("case_sensitive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let language = config
            .get("language")
            .and_then(Value::as_str)
            .unwrap_or("english");
        let mut options = AnalysisOptions {
            tokenizer: TokenizerOptions::UAX29Word(Default::default()),
            maximum_token_length: Some(
                config
                    .get("max_token_length")
                    .and_then(Value::as_u64)
                    .map_or(DEFAULT_MAX_TOKEN_LENGTH, |length| length as usize),
            ),
            case_sensitive,
            stopword_removal: flag("remove_stopwords")
                .then(|| stopword_language(language))
                .flatten()
                .map(StopwordRemoval::ForLanguage),
            stemming: flag("stemming")
                .then(|| stemming_language(language))
                .flatten(),
            ascii_folding: flag("ascii_folding"),
        };
        if !options.valid() {
            // Validation rejects these combinations at write time; never panic on them.
            options.stemming = None;
            options.stopword_removal = None;
        }
        let key = format!("{tokenizer:?}{options:?}").into();
        Self {
            tokenizer,
            options,
            key,
        }
    }

    pub fn is_pre_tokenized(&self) -> bool {
        self.tokenizer == Tokenizer::PreTokenized
    }

    /// Tokens of a `string` or `[]string` value, with positions threaded across elements.
    pub fn analyze(&self, value: &Value) -> Rc<Vec<Token>> {
        let inputs: Vec<&str> = match value {
            Value::String(text) => vec![text.as_str()],
            Value::Array(values) => values.iter().filter_map(Value::as_str).collect(),
            _ => return Rc::new(Vec::new()),
        };
        let cache_key = (self.key.clone(), cache_text(&inputs));
        if let Some(tokens) = TOKENS.with(|cache| cache.borrow().get(&cache_key).cloned()) {
            return tokens;
        }
        let tokens = Rc::new(self.analyze_inputs(&inputs));
        TOKENS.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= TOKEN_CACHE_LIMIT {
                cache.clear();
            }
            cache.insert(cache_key, tokens.clone());
        });
        tokens
    }

    /// Token texts of a query operand: a string, or a `[]string` of tokens for
    /// `pre_tokenized_array`.
    pub fn query_tokens(&self, query: &Value) -> Vec<String> {
        self.analyze(query)
            .iter()
            .map(|token| token.text.clone())
            .collect()
    }

    fn with_buffer<R>(&self, work: impl FnOnce(&mut ReusableBuffer) -> R) -> R {
        BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            let buffer = buffers
                .entry(format!("{:?}", self.options.stemming))
                .or_insert_with(ReusableBuffer::new);
            work(buffer)
        })
    }

    fn analyze_inputs(&self, inputs: &[&str]) -> Vec<Token> {
        match self.tokenizer {
            Tokenizer::PreTokenized => inputs
                .iter()
                .enumerate()
                .map(|(index, text)| Token {
                    text: (*text).to_owned(),
                    position: index,
                    byte_range: 0..text.len(),
                })
                .collect(),
            Tokenizer::Word(3 | 4) => {
                let analyzer = Analyzer::new(self.options);
                let mut tokens = Vec::new();
                self.with_buffer(|buffer| {
                    analyzer.analyze_inputs(inputs.iter().copied(), buffer, |token| {
                        tokens.push(Token {
                            text: token.text.to_owned(),
                            position: token.position,
                            byte_range: token.byte_range,
                        });
                        true
                    });
                });
                tokens
            }
            Tokenizer::Word(version) => self.analyze_legacy(inputs, version),
        }
    }

    /// `word_v0` to `word_v2`: segments are runs of alphanumeric codepoints, single
    /// ideographs (v2) and emoji glyphs (v1, v2). Each segment is one token after the
    /// length limit, lowercasing, stopword, stemming and folding filters.
    fn analyze_legacy(&self, inputs: &[&str], version: u8) -> Vec<Token> {
        let mut filters = self.options;
        filters.maximum_token_length = None;
        let analyzer = Analyzer::new(filters);
        let mut tokens = Vec::new();
        let mut position = 0;
        self.with_buffer(|buffer| {
            for input in inputs {
                for range in legacy_segments(input, version) {
                    let segment = &input[range.clone()];
                    let this_position = position;
                    position += 1;
                    if self
                        .options
                        .maximum_token_length
                        .is_some_and(|limit| segment.chars().count() > limit)
                    {
                        continue;
                    }
                    // A segment can hold several UAX #29 words (for example an ideograph run
                    // in v0 and v1); the filters are per character or per word, so joining the
                    // filtered words rebuilds the filtered segment.
                    let mut text = String::new();
                    analyzer.analyze(segment, buffer, |token| {
                        text.push_str(token.text);
                        true
                    });
                    if !text.is_empty() {
                        tokens.push(Token {
                            text,
                            position: this_position,
                            byte_range: range,
                        });
                    }
                }
            }
        });
        tokens
    }
}

fn cache_text(inputs: &[&str]) -> String {
    // A separator that cannot occur in JSON text keeps ["ab"] and ["a", "b"] apart.
    inputs.join("\u{0}")
}

fn legacy_segments(text: &str, version: u8) -> Vec<Range<usize>> {
    let mut segments = Vec::new();
    let mut characters = text.char_indices().peekable();
    while let Some((start, character)) = characters.next() {
        let mut end = start + character.len_utf8();
        if version >= 1 && is_emoji(character) {
            while let Some(&(index, next)) = characters.peek() {
                if !is_emoji_continuation(next) {
                    break;
                }
                end = index + next.len_utf8();
                characters.next();
            }
            segments.push(start..end);
        } else if version >= 2 && is_ideograph(character) {
            segments.push(start..end);
        } else if character.is_alphanumeric() {
            while let Some(&(index, next)) = characters.peek() {
                if !next.is_alphanumeric() || (version >= 2 && is_ideograph(next)) {
                    break;
                }
                end = index + next.len_utf8();
                characters.next();
            }
            segments.push(start..end);
        }
    }
    segments
}

fn is_ideograph(character: char) -> bool {
    matches!(character as u32,
        0x3006 | 0x3007 | 0x3021..=0x3029 | 0x3038..=0x303A | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F | 0x30000..=0x323AF)
}

fn is_emoji(character: char) -> bool {
    matches!(character as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2300..=0x23FF | 0x2B00..=0x2BFF)
}

fn is_emoji_continuation(character: char) -> bool {
    is_emoji(character) || matches!(character as u32, 0x200D | 0xFE0F | 0xE0020..=0xE007F)
}

fn stopword_language(language: &str) -> Option<LanguageWithStopwords> {
    Some(match language {
        "danish" => LanguageWithStopwords::Danish,
        "dutch" => LanguageWithStopwords::Dutch,
        "english" => LanguageWithStopwords::English,
        "finnish" => LanguageWithStopwords::Finnish,
        "french" => LanguageWithStopwords::French,
        "german" => LanguageWithStopwords::German,
        "hungarian" => LanguageWithStopwords::Hungarian,
        "italian" => LanguageWithStopwords::Italian,
        "norwegian" => LanguageWithStopwords::Norwegian,
        "portuguese" => LanguageWithStopwords::Portuguese,
        "russian" => LanguageWithStopwords::Russian,
        "spanish" => LanguageWithStopwords::Spanish,
        "swedish" => LanguageWithStopwords::Swedish,
        _ => return None,
    })
}

fn stemming_language(language: &str) -> Option<StemmingLanguage> {
    Some(match language {
        "arabic" => StemmingLanguage::Arabic,
        "danish" => StemmingLanguage::Danish,
        "dutch" => StemmingLanguage::Dutch,
        "english" => StemmingLanguage::English,
        "finnish" => StemmingLanguage::Finnish,
        "french" => StemmingLanguage::French,
        "german" => StemmingLanguage::German,
        "greek" => StemmingLanguage::Greek,
        "hungarian" => StemmingLanguage::Hungarian,
        "italian" => StemmingLanguage::Italian,
        "norwegian" => StemmingLanguage::Norwegian,
        "portuguese" => StemmingLanguage::Portuguese,
        "romanian" => StemmingLanguage::Romanian,
        "russian" => StemmingLanguage::Russian,
        "spanish" => StemmingLanguage::Spanish,
        "swedish" => StemmingLanguage::Swedish,
        "tamil" => StemmingLanguage::Tamil,
        "turkish" => StemmingLanguage::Turkish,
        _ => return None,
    })
}

/// A schema error, and whether the live service reports it as a JSON shape error (HTTP 422)
/// rather than a semantic one (HTTP 400).
pub(crate) enum ConfigError {
    Shape(String),
    Invalid(String),
}

/// Validates an object-form `full_text_search` configuration with the live service's rules.
/// Unknown keys are accepted and ignored, as they are live.
pub(crate) fn validate_config(field_type: &str, config: &Value) -> Result<(), ConfigError> {
    let Some(config) = config.as_object() else {
        return Ok(());
    };
    let shape = |message: &str| ConfigError::Shape(message.to_owned());
    for (key, value) in config {
        match key.as_str() {
            "tokenizer"
                if !value.as_str().is_some_and(|name| {
                    matches!(
                        name,
                        "word_v0"
                            | "word_v1"
                            | "word_v2"
                            | "word_v3"
                            | "word_v4"
                            | "pre_tokenized_array"
                    )
                }) =>
            {
                return Err(shape("unknown full_text_search tokenizer"));
            }
            "language" if !value.as_str().is_some_and(|name| LANGUAGES.contains(&name)) => {
                return Err(shape("unknown full_text_search language"));
            }
            "case_sensitive" | "stemming" | "remove_stopwords" | "ascii_folding"
                if !value.is_boolean() =>
            {
                return Err(shape("full_text_search flags must be booleans"));
            }
            "max_token_length" if !value.is_u64() => {
                return Err(shape("max_token_length must be an integer"));
            }
            "k1" | "k3" if value.as_f64().is_none_or(|n| !n.is_finite() || n <= 0.0) => {
                return Err(ConfigError::Invalid(format!(
                    "{key} must be greater than zero"
                )));
            }
            "b" if value
                .as_f64()
                .is_none_or(|n| !n.is_finite() || !(0.0..=1.0).contains(&n)) =>
            {
                return Err(ConfigError::Invalid("b must be between 0 and 1".into()));
            }
            _ => {}
        }
    }
    let flag = |key: &str| config.get(key).and_then(Value::as_bool);
    let invalid = |message: String| Err(ConfigError::Invalid(message));
    if config
        .get("max_token_length")
        .and_then(Value::as_u64)
        .is_some_and(|length| !(1..=255).contains(&length))
    {
        return invalid("max_token_length must be between 1 and 255".into());
    }
    if config.get("tokenizer").and_then(Value::as_str) == Some("pre_tokenized_array") {
        if field_type != "[]string" {
            return invalid(format!(
                "full text search with tokenizer `pre_tokenized_array` does not support attribute type {field_type} (valid attribute types: []string)"
            ));
        }
        if flag("stemming") == Some(true) {
            return invalid("cannot specify stemming with `pre_tokenized_array` tokenizer".into());
        }
        if flag("remove_stopwords") == Some(true) {
            return invalid(
                "cannot specify remove_stopwords with `pre_tokenized_array` tokenizer".into(),
            );
        }
        if flag("case_sensitive") == Some(false) {
            return invalid(
                "cannot specify case_sensitive=false with `pre_tokenized_array` tokenizer".into(),
            );
        }
        if config.contains_key("language") {
            return invalid("cannot specify language with `pre_tokenized_array` tokenizer".into());
        }
        return Ok(());
    }
    if flag("case_sensitive") == Some(true) {
        if flag("stemming") == Some(true) {
            return invalid("stemming is not supported when case sensitivity is enabled".into());
        }
        if flag("remove_stopwords") == Some(true) {
            return invalid(
                "stopword removal is not supported when case sensitivity is enabled".into(),
            );
        }
    }
    let language = config
        .get("language")
        .and_then(Value::as_str)
        .unwrap_or("english");
    if flag("remove_stopwords") == Some(true) && stopword_language(language).is_none() {
        return invalid(format!(
            "stopword removal is not supported for language: {language}"
        ));
    }
    Ok(())
}

/// The live service's normalized `full_text_search` object for `GET .../schema`.
pub(crate) fn normalized_config(config: &Value) -> Value {
    let analysis = TextAnalysis::from_config(if config.is_object() {
        config
    } else {
        &Value::Null
    });
    let number =
        |key: &str, default: f64| config.get(key).and_then(Value::as_f64).unwrap_or(default);
    let pre_tokenized = analysis.is_pre_tokenized();
    let tokenizer = config
        .get("tokenizer")
        .and_then(Value::as_str)
        .unwrap_or("word_v4");
    let flag = |key: &str| config.get(key).and_then(Value::as_bool).unwrap_or(false);
    serde_json::json!({
        "k1": number("k1", 1.2),
        "b": number("b", 0.75),
        "k3": number("k3", 8.0),
        "ascii_folding": flag("ascii_folding"),
        "case_sensitive": analysis.options.case_sensitive,
        "language": if pre_tokenized {
            Value::Null
        } else {
            config.get("language").cloned().unwrap_or_else(|| "english".into())
        },
        "stemming": flag("stemming"),
        "remove_stopwords": flag("remove_stopwords"),
        "tokenizer": tokenizer,
        "max_token_length": if pre_tokenized {
            Value::Null
        } else {
            analysis.options.maximum_token_length.into()
        },
    })
}

/// Byte ranges of UAX #29 sentences in `text`, trimmed of surrounding whitespace.
pub(crate) fn sentences(text: &str) -> Vec<Range<usize>> {
    let mut breaks = vec![0];
    alyze::uax29::sentence::tokenize(text, Default::default(), |position| {
        breaks.push(position);
        true
    });
    breaks.push(text.len());
    breaks.dedup();
    breaks
        .windows(2)
        .filter_map(|window| trim(text, window[0]..window[1]))
        .collect()
}

/// Byte ranges of newline-separated paragraphs in `text`, trimmed of surrounding whitespace.
pub(crate) fn paragraphs(text: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (index, _) in text.match_indices('\n') {
        ranges.extend(trim(text, start..index));
        start = index + 1;
    }
    ranges.extend(trim(text, start..text.len()));
    ranges
}

fn trim(text: &str, range: Range<usize>) -> Option<Range<usize>> {
    let slice = &text[range.clone()];
    let leading = slice.len() - slice.trim_start().len();
    let trimmed = slice.trim();
    (!trimmed.is_empty()).then(|| range.start + leading..range.start + leading + trimmed.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn texts(config: Value, text: &str) -> Vec<String> {
        TextAnalysis::from_config(&config)
            .analyze(&json!(text))
            .iter()
            .map(|token| token.text.clone())
            .collect()
    }

    #[test]
    fn legacy_tokenizers_split_on_character_classes() {
        let text = "e.g. don't 東京タワー 🐡 sea.com";
        assert_eq!(
            texts(json!({"tokenizer":"word_v0"}), text),
            ["e", "g", "don", "t", "東京タワー", "sea", "com"]
        );
        assert_eq!(
            texts(json!({"tokenizer":"word_v1"}), text),
            ["e", "g", "don", "t", "東京タワー", "🐡", "sea", "com"]
        );
        assert_eq!(
            texts(json!({"tokenizer":"word_v2"}), text),
            [
                "e",
                "g",
                "don",
                "t",
                "東",
                "京",
                "タワー",
                "🐡",
                "sea",
                "com"
            ]
        );
        assert_eq!(
            texts(json!({}), text),
            ["e.g", "don't", "東", "京", "タワー", "🐡", "sea.com"]
        );
    }

    #[test]
    fn filters_follow_the_live_order() {
        // Stemming runs before folding, so "straße" folds to "strasse" but "strasse" stems
        // to "strass" first.
        let config = json!({"stemming":true,"ascii_folding":true});
        assert_eq!(texts(config.clone(), "Straße"), ["strasse"]);
        assert_eq!(texts(config, "strasse"), ["strass"]);
        assert_eq!(texts(json!({"max_token_length":4}), "café clams"), ["café"]);
    }

    #[test]
    fn stopwords_leave_position_gaps() {
        let tokens = TextAnalysis::from_config(&json!({"remove_stopwords":true}))
            .analyze(&json!("visited a café"));
        let positions: Vec<_> = tokens.iter().map(|token| token.position).collect();
        assert_eq!(positions, [0, 2]);
    }

    #[test]
    fn fragments_split_sentences_and_paragraphs() {
        let text = "One two. Three four.\n\nFive six.\nSeven.";
        let slices = |ranges: Vec<Range<usize>>| {
            ranges
                .into_iter()
                .map(|range| &text[range])
                .collect::<Vec<_>>()
        };
        assert_eq!(
            slices(sentences(text)),
            ["One two.", "Three four.", "Five six.", "Seven."]
        );
        assert_eq!(
            slices(paragraphs(text)),
            ["One two. Three four.", "Five six.", "Seven."]
        );
    }
}
