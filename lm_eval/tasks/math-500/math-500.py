"""TODO(Math-500): Add a description here."""


import json
import os
import datasets


# BibTeX citation
_CITATION = """\
@inproceedings{lightman2023let,
  title={Let's verify step by step},
  author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yuri and Edwards, Harrison and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
"""

_DESCRIPTION = ""
'''
_DESCRIPTION = """
HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI. A paper was published at ACL2019.
"""
_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/"
_URLS = {
    "train": _URL + "hellaswag_train.jsonl",
    "test": _URL + "hellaswag_test.jsonl",
    "dev": _URL + "hellaswag_val.jsonl",
}
'''


class Math500(datasets.GeneratorBasedBuilder):
    """TODO(hellaswag): Short description of my dataset."""

    # TODO(hellaswag): Set up version.
    # VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(hellaswag): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "problem": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/openai/prm800k/tree/main",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(hellaswag): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        # The manual_dir is expected to contain the local dataset file.
        manual_dir = dl_manager.manual_dir
        if manual_dir is None:
            raise ValueError(
                "This is a manual dataset. Please specify the directory where the dataset file is located."
            )
        filepath = os.path.join(manual_dir, "test.jsonl")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepath},
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "problem": data["problem"],
                    "answer": data["answer"]
                }