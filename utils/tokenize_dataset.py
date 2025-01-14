# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import argparse

import datasets
from transformers import AutoTokenizer


logger = logging.getLogger(__file__)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Use streaming dataset instead to speed up loading
    input_texts = datasets.load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        #num_proc=args.num_proc,
        trust_remote_code=True,
        streaming=True
    )

    def tokenize(example):
        tokenized = tokenizer(
            example[args.feature],
            add_special_tokens=False,
            padding=True,
            truncation=False,
            max_length=sys.maxsize,
            return_attention_mask=True,
        )
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]
        example["tokenized_len"] = len(tokenized["input_ids"])
        return example

    processed_samples = []
    for example in input_texts:
        tokenized_example = tokenize(example)
        processed_samples.append(tokenized_example)

    dataset = datasets.Dataset.from_list(processed_samples)

    dataset.save_to_disk(args.save_tokenized)
    logger.info(f"Saved tokenized dataset to {args.save_tokenized}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--feature", type=str)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--num-proc", type=int, default=8)

    main(parser.parse_args())
