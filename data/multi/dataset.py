 # Multilingual dataset (combining all 5 - German(de), French(fr), Spanish(es), Italian(it), Swedish(sv))
from datasets import concatenate_datasets

from data.de.dataset import load_german_dataset
from data.fr.dataset import load_french_dataset
from data.es.dataset import load_spanish_dataset
from data.it.dataset import load_italian_dataset
from data.sv.dataset import load_swedish_dataset


def load_multilingual_dataset():
    de = load_german_dataset()
    fr = load_french_dataset()
    es = load_spanish_dataset()
    it = load_italian_dataset()
    sv = load_swedish_dataset()

    combined = concatenate_datasets([de, fr, es, it, sv])

    return combined