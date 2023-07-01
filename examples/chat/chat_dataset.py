#!/usr/bin/env python
#  coding=utf-8
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

"""Corpus for chat dataset"""

import datasets
import jsonlines


_DESCRIPTION = """\
Chat dataset to fine tune an llm to chat capability
"""

_CITATION = """\
hand crafted and databricks dolly dataset
"""

_WEBPAGE = ""


class Chat(datasets.GeneratorBasedBuilder):
    """Chat dataset loader"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string"),
                    "category": datasets.Value("string")
                }
            ),
            homepage=_WEBPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        train_path = '/home/arif/Documents/Data/Share_GPT/chat_trainformat_train.jsonl'
        validation_path = '/home/arif/Documents/Data/Share_GPT/chat_trainformat_val.jsonl'
        test_path = '/home/arif/Documents/Data/Share_GPT/chat_trainformat_test.jsonl'
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": validation_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        print(f"filepath in generate examples {filepath}")
        key = 0
        with open(filepath, "r", encoding="utf-8") as reader:

            for item in jsonlines.Reader(reader):
                yield key, item
                key += 1
