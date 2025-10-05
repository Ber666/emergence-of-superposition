# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """
        # print(features)
        ## print(features[0]["input_ids"])
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k not in ["position_ids", "candidate_nodes"]
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        batch["candidate_nodes"] = [feature["candidate_nodes"] for feature in features]
        return batch


def expand_data(data, k, max_steps, neg_sampling=False, shuffle_nodes=False):
    
    assert k <= max_steps + 1
    # k = 1, 2, 3, 4, 5
    
    # Shuffle node indices if requested
    if shuffle_nodes:
        # Extract all unique node indices from the data
        node_indices = set()
        
        # Add indices from edges
        for edge in data['edges']:
            node_indices.add(edge[0])
            node_indices.add(edge[1])
        
        # Convert to sorted list for consistent ordering
        node_list = sorted(list(node_indices))
        
        # Create a random shuffle of the node indices
        shuffled_indices = node_list.copy()
        random.shuffle(shuffled_indices)
        
        # Create mapping from old index to new index
        index_mapping = {old_idx: new_idx for old_idx, new_idx in zip(node_list, shuffled_indices)}
        
        # Apply shuffle to the data (create a copy to avoid modifying original)
        data = data.copy()
        data['edges'] = [[index_mapping[edge[0]], index_mapping[edge[1]]] for edge in data['edges']]
        data['root'] = index_mapping[data['root']]
        data['target'] = index_mapping[data['target']]
        data['neg_target'] = index_mapping[data['neg_target']]
        
        # Apply shuffle to neighbor_k
        shuffled_neighbor_k = {}
        for k_val, neighbors in data['neighbor_k'].items():
            shuffled_neighbor_k[k_val] = [index_mapping[neighbor] for neighbor in neighbors]
        data['neighbor_k'] = shuffled_neighbor_k
        
        # Create new idx_to_symbol array based on the shuffle mapping
        reverse_mapping = {new_idx: old_idx for old_idx, new_idx in index_mapping.items()}
        data['idx_to_symbol'] = [data['idx_to_symbol'][reverse_mapping[i]] for i in range(len(data['idx_to_symbol']))]
    
    symbol_to_idx = {}
    for i, s in enumerate(data['idx_to_symbol']):
        symbol_to_idx[s] = i
    
    def get_prefix(data):
        
        random.shuffle(data['edges'])

        question = "<eos> " + "|".join([f" {e[0]} {e[1]} " for e in data['edges']]).strip() + \
            " [Q] "

        if random.random() < 0.5:
            question += str(data['target']) + " " + str(data['neg_target'])
        else:
            question += str(data['neg_target']) + " " + str(data['target'])

        question += " [R] " + str(data['root'])
        
        return question

    candidate_nodes = []

    # return_data = None
    if k <= max_steps:
        # for n in data["neighbor_k"][str(k)]:
        if neg_sampling:
            if random.random() < 0.2:
                question = get_prefix(data) + " <|latent|>" * (k-1) + " [A] "
                continuation = "<|no-answer|>"
                return_data = (question, continuation, candidate_nodes)
        
            else:
                n = random.choice(data["neighbor_k"][str(k)])
                question = get_prefix(data) + " <|latent|>" * (k-1) + " "
                continuation = str(n)
                candidate_nodes = data["neighbor_k"][str(k)]
                return_data = (question, continuation, candidate_nodes)
        else:
            n = random.choice(data["neighbor_k"][str(k)])
            question = get_prefix(data) + " <|latent|>" * (k-1) + " "
            continuation = str(n)
            candidate_nodes = data["neighbor_k"][str(k)]
            return_data = (question, continuation, candidate_nodes)


    elif k == max_steps + 1:
        if neg_sampling:
            if random.random() < 0.2:
                question = get_prefix(data) + " <|latent|>" * random.randint(0, max_steps - 1) + " [A] "
                continuation = "<|no-answer|>"
                return_data = (question, continuation, candidate_nodes)
            else:
                question = get_prefix(data) + " <|latent|>" * max_steps + " [A] "
                continuation = str(data["target"])
                return_data = (question, continuation, candidate_nodes)
        else:
            question = get_prefix(data) + " <|latent|>" * max_steps + " [A] "
            continuation = str(data["target"])
            return_data = (question, continuation, candidate_nodes)
    
    else:
        raise ValueError(f"k is {k}, max_steps is {max_steps}")
    
    return return_data



def get_graph_latent_cot_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
    shuffle_nodes
):
    base_dataset = json.load(open(dataset_path))
    
    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.randint(0, min(scheduled_stage, len(sample["steps"])))
        else:
            scheduled_stage_to_train = min(scheduled_stage, len(sample["steps"]))
            # 0, 1, 2, 3, 4

        # this range is [0, ..., len(sample["steps"])]
        # including both ends

        expanded_data = expand_data(sample, scheduled_stage_to_train + 1, len(sample["steps"]), shuffle_nodes=shuffle_nodes)
        
        # Process each question-continuation pair
        processed_samples = []
        for question, continuation, candidate_nodes in [expanded_data]:
            question_tokenized = tokenizer.encode(question, add_special_tokens=False)
            continuation_tokenized = tokenizer.encode(continuation, add_special_tokens=False)
            
            tokens = question_tokenized + continuation_tokenized
            
            processed_sample = {
                "input_ids": tokens,
                "labels": [-100] * len(question_tokenized) + continuation_tokenized,
                "attention_mask": [1] * len(tokens),
                "position_ids": list(range(len(tokens))),
                "candidate_nodes": candidate_nodes,
            }
            processed_samples.append(processed_sample)
            
        return processed_samples

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for sample in base_dataset:
                processed_samples = process_dataset(sample)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Process each sample and collect all results
        all_processed_samples = []
        for sample in base_dataset:
            processed_samples = process_dataset(sample)
            all_processed_samples.extend(processed_samples)
        
        processed_dataset = all_processed_samples
        random.shuffle(processed_dataset)
        dataset = processed_dataset
    return dataset

def get_graph_latent_question_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
):
    base_dataset = json.load(open(dataset_path))
    if configs.debug:
        base_dataset = base_dataset[:10000]
    # similar to get_graph_latent_dataset, but we only keep the question
    # without the continuation
    
    def process_dataset(sample, idx):
        expanded_data = expand_data(sample, len(sample["steps"]) + 1, len(sample["steps"]), shuffle_nodes=False)
        processed_samples = []
        for question, continuation, candidate_nodes in [expanded_data]:
            question_tokenized = tokenizer.encode(question, add_special_tokens=False)
            processed_samples.append({
                "input_ids": question_tokenized,
                "attention_mask": [1] * len(question_tokenized),
                "position_ids": list(range(len(question_tokenized))),
                "idx": idx,
                "candidate_nodes": candidate_nodes,
            })
        return processed_samples
    
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for idx, sample in enumerate(base_dataset):
                processed_samples = process_dataset(sample, idx)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    return dataset

