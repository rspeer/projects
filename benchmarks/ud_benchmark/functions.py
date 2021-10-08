import datasets
import itertools
import langcodes

from spacy import util, Language
from spacy.tokens import Doc
from spacy.training.example import Example
from typing import Optional, Iterable, Iterator, Callable

from spacious_corpus.corpus import corpus_reader


# I copied this dict from
# https://github.com/huggingface/datasets/blob/master/datasets/oscar/oscar.py.
# I'd just import it, but it's private, so that would be bad form.
OSCAR_LANGUAGES = {
    "af": "Afrikaans",
    "als": "Tosk Albanian",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "arz": "Egyptian Arabic",
    "ast": "Asturian",
    "as": "Assamese",
    "av": "Avaric",
    "azb": "South Azerbaijani",
    "az": "Azerbaijani",
    "bar": "Bavarian",
    "ba": "Bashkir",
    "bcl": "Central Bikol",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bh": "Bihari",
    "bn": "Bengali",
    "bo": "Tibetan",
    "bpy": "Bishnupriya",
    "br": "Breton",
    "bs": "Bosnian",
    "bxr": "Russia Buriat",
    "ca": "Catalan",
    "cbk": "Chavacano",
    "ceb": "Cebuano",
    "ce": "Chechen",
    "ckb": "Central Kurdish",
    "cs": "Czech",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "diq": "Dimli",
    "dsb": "Lower Sorbian",
    "dv": "Dhivehi",
    "el": "Modern Greek",
    "eml": "Emilian-Romagnol",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "frr": "Northern Frisian",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gom": "Goan Konkani",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hsb": "Upper Sorbian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ilo": "Iloko",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jbo": "Lojban",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "krc": "Karachay-Balkar",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kirghiz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lez": "Lezghian",
    "li": "Limburgan",
    "lmo": "Lombard",
    "lo": "Lao",
    "lrc": "Northern Luri",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mai": "Maithili",
    "mg": "Malagasy",
    "mhr": "Eastern Mari",
    "min": "Minangkabau",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mrj": "Western Mari",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "mwl": "Mirandese",
    "my": "Burmese",
    "myv": "Erzya",
    "mzn": "Mazanderani",
    "nah": "Nahuatl languages",
    "nap": "Neapolitan",
    "nds": "Low German",
    "ne": "Nepali",
    "new": "Newari",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "os": "Ossetian",
    "pam": "Pampanga",
    "pa": "Panjabi",
    "pl": "Polish",
    "pms": "Piemontese",
    "pnb": "Western Panjabi",
    "ps": "Pushto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "ro": "Romanian",
    "ru": "Russian",
    "sah": "Yakut",
    "sa": "Sanskrit",
    "scn": "Sicilian",
    "sd": "Sindhi",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "tyv": "Tuvinian",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vec": "Venetian",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "war": "Waray",
    "wa": "Walloon",
    "wuu": "Wu Chinese",
    "xal": "Kalmyk",
    "xmf": "Mingrelian",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Yue Chinese",
    "zh": "Chinese",
}


def stream_oscar_2018(
    lang: str,
    limit: Optional[int] = None,
    cache_path: Optional[str] = None
):
    """
    Stream the OSCAR 2018 corpus in a particular language from HuggingFace
    Datasets.
    """
    dataset = datasets.load_dataset(
        "oscar",
        f"unshuffled_deduplicated_{lang}",
        split="train",
        streaming=True,
        cache_dir=cache_path,
    )
    if limit is not None:
        dataset = itertools.islice(dataset, limit)
    return dataset


@util.registry.readers("elake.OscarCorpus.v1")
def create_oscar_reader(
    oscar_version: str,
    limit: Optional[int],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Callable[[Language], Iterable[Doc]]:
    """
    Create an object that streams an OSCAR corpus from HuggingFace Datasets.
    """
    # future proofing, because OSCAR 21.09 is out but it isn't in Datasets yet
    if oscar_version != "2018":
        raise ValueError("OSCAR 2018 is the only supported version so far.")

    def iterate_oscar_language(nlp: Language) -> Iterator[Example]:
        # Get OSCAR in a language that matches the nlp object.
        # We need to align the language code with OSCAR's list, because for
        # example spaCy's "nb" is OSCAR's "no".
        lang = langcodes.closest_supported_match(
            nlp.lang,
            list(OSCAR_LANGUAGES)
        )
        if lang is None:
            raise ValueError(f"Language '{lang}' is not supported by OSCAR")

        for (i, record) in enumerate(stream_oscar_2018(lang, limit, cache_path)):
            doc = nlp.make_doc(record["text"])

            # debugging for now
            if i % 10000 == 0:
                print(i, len(doc), record)

            if min_length is not None and len(doc) < min_length:
                continue
            elif max_length is not None and len(doc) >= max_length:
                # the >= surprises me but it matches JsonlCorpus. I can see
                # how these would be interpreted like a Python range.
                continue
            elif len(record["text"]) >= nlp.max_length:
                print("Warning - text exceeds NLP max length:")
                print(record["text"])
                continue

            words = [w.text for w in doc]
            spaces = [bool(w.whitespace_) for w in doc]
            yield Doc(nlp.vocab, words=words, spaces=spaces)
    
    return iterate_oscar_language
