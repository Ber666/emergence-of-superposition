# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoConfig

from stokenizer import STokenizer
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut import Coconut
from dataset import (
    MyCollator,
    get_graph_latent_question_dataset,
    get_graph_latent_cot_dataset,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed

def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()
    
    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    
    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(configs.model_id)
    )
    
    print(model)

    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    shuffle_nodes = configs.shuffle_nodes if hasattr(configs, "shuffle_nodes") else False
    first_stage_epochs = configs.first_stage_epochs if hasattr(configs, "first_stage_epochs") else 0

    # Wrap model with Coconut
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    answers_val = [
        d["target"] for d in json.load(open(configs.val_path))
    ]

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None


    optimizer = optim.AdamW(
        parallel_model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
    )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        if epoch < first_stage_epochs:
            scheduled_stage = 0
        else:
            scheduled_stage = (epoch - first_stage_epochs) // configs.epochs_per_stage
        print("scheduled_stage", scheduled_stage)

        dataset_gen_val = get_graph_latent_question_dataset(
            configs.val_path,
            scheduled_stage,
            configs,
            tokenizer
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            dataset_train = get_graph_latent_cot_dataset(
                configs.train_path,
                scheduled_stage,
                configs,
                tokenizer,
                shuffle_nodes=shuffle_nodes,
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).
            dataset_loss_val = get_graph_latent_cot_dataset(
                configs.val_path,
                scheduled_stage,
                configs,
                tokenizer,
                shuffle_nodes=shuffle_nodes,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            last_scheduled_stage = (epoch - 1) // configs.epochs_per_stage

            reset_optimizer_tag = False
            
            if configs.reset_optimizer and \
                scheduled_stage <= configs.max_latent_stage and \
                    scheduled_stage > last_scheduled_stage:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

                reset_optimizer_tag = True

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0 and (epoch == 0 or reset_optimizer_tag):
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["attention_mask"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})
                    # this will produce larger and larger tables as the training progresses
                    # so we don't log it to wandb when the epoch number is set large
                
                total_train_steps += 1
                candidate_nodes = batch["candidate_nodes"]  # a list of lists
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key not in ["idx", "candidate_nodes"]
                }
                batch["candidate_nodes"] = candidate_nodes
                # pass loss config flag
                if hasattr(configs, "bfs_paper_loss") and configs.bfs_paper_loss:
                    batch["bfs_paper_loss"] = True
                elif hasattr(configs, "bfs-paper-loss") and getattr(configs, "bfs-paper-loss"):
                    batch["bfs-paper-loss"] = True
                outputs = parallel_model(**batch)


                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()
                
                # ------------
                # calculate the prediction for the latest embedding
                logits = outputs.logits  # (bs, seq_len, d_model)
                # find the position of the first token with label != 100
                
                labels = batch["labels"] # (bs, seq_len)
                shift_labels = labels[..., 1:].contiguous()
                first_non_pad_idx = [
                    torch.nonzero(shift_labels[i] != -100)[0][0].item() for i in range(len(shift_labels))
                ]
                count_correct = 0
                count_correct_candidate = 0
                
                if step == 0 and rank == 0:
                    print("="*100)
                    print("train...")
                for i in range(len(candidate_nodes)):
                    predicted_node = logits[i][first_non_pad_idx[i]].argmax(dim=-1).item()
                    label = shift_labels[i][first_non_pad_idx[i]].item()
                    if step == 0 and i == 0 and rank == 0:
                        print("predicted_node, label, candidate_nodes[i]", predicted_node, label, candidate_nodes[i])
                    if predicted_node == label:
                        count_correct += 1
                    if predicted_node in candidate_nodes[i]:
                        count_correct_candidate += 1
                
                if step == 0 and rank == 0:
                    print("="*100)
                    print("summary:",count_correct, count_correct_candidate, len(candidate_nodes))
                # extract the latest 
                # calculate the prediction for the latest embedding
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                        "train/prediction_acc": count_correct / len(candidate_nodes),
                        "train/neighbor_candidate_acc": count_correct_candidate / len(candidate_nodes),
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                total_count_correct = 0
                total_count_correct_candidate = 0
                total_samples = 0
                for step, batch in enumerate(valid_loss_dataloader):

                    candidate_nodes = batch["candidate_nodes"]
                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key not in ["idx", "candidate_nodes"]
                    }
                    batch["candidate_nodes"] = candidate_nodes
                    # pass loss config flag
                    if hasattr(configs, "bfs_paper_loss") and configs.bfs_paper_loss:
                        batch["bfs_paper_loss"] = True
                    elif hasattr(configs, "bfs-paper-loss") and getattr(configs, "bfs-paper-loss"):
                        batch["bfs-paper-loss"] = True
                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                    # calculate the prediction for the latest embedding
                    logits = outputs.logits  # (bs, seq_len, d_model)
                    # find the position of the first token with label != 100
                    
                    labels = batch["labels"] # (bs, seq_len)
                    shift_labels = labels[..., 1:].contiguous()
                    first_non_pad_idx = [
                        torch.nonzero(shift_labels[i] != -100)[0][0].item() for i in range(len(shift_labels))
                    ]
                    count_correct = torch.tensor(0, device=rank, dtype=torch.long)
                    count_correct_candidate = torch.tensor(0, device=rank, dtype=torch.long)
                    n_samples = torch.tensor(len(candidate_nodes), device=rank, dtype=torch.long)
                    
                    if step == 0 and rank == 0:
                        print("="*100)
                        print("eval...")
                    for i in range(len(candidate_nodes)):
                        predicted_node = logits[i][first_non_pad_idx[i]].argmax(dim=-1).item()
                        label = shift_labels[i][first_non_pad_idx[i]].item()
                        count_correct += int(predicted_node == label)
                        count_correct_candidate += int(predicted_node in candidate_nodes[i])
                        if step == 0 and i == 0 and rank == 0:
                            print("predicted_node, label, candidate_nodes[i]", predicted_node, label, candidate_nodes[i])
                    
                    if step == 0 and rank == 0:
                        print("="*100)
                        print("summary:",count_correct, count_correct_candidate, len(candidate_nodes))
                    
                    dist.all_reduce(count_correct, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_correct_candidate, op=dist.ReduceOp.SUM)
                    dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
                
                    total_samples += n_samples.item()
                    total_count_correct += count_correct.item()
                    total_count_correct_candidate += count_correct_candidate.item()

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                        "eval/prediction_acc": total_count_correct / total_samples,
                        "eval/neighbor_candidate_acc": total_count_correct_candidate / total_samples,
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # if scheduled_stage >= configs.max_latent_stage:
        if True:
            # val generation accuracy
            total_length = len(valid_gen_dataloader)

            pbar = tqdm(
                colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
            )
            cor, cor_cot, total = (
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
            )

            with torch.no_grad():
                parallel_model.module.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(rank)
                        for k, v in batch.items()
                        if v != None and k not in ["idx", "position_ids", "candidate_nodes"]
                    }
                    # https://github.com/huggingface/transformers/issues/32492

                    assert len(batch["input_ids"]) == 1
                    answer = str(answers_val[test_idx.cpu().item()])
                    # answer_cot = cot_val[test_idx.cpu().item()]
                    # question = question_val[test_idx.cpu().item()]

                    total += 1

                    # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                    outputs = parallel_model.module.generate(
                        **batch,
                        max_new_tokens=1,
                        synced_gpus=not configs.only_eval,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("<eos>", "").strip()
                    answer_output = text_output.split("[A]")[-1].replace(",", "").strip()
                    cot_output = (
                        ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                    )

                    if idx < 5 and rank == 0:
                        # print some examples
                        print(
                            f"Question {test_idx}: Answer = '{answer}'"
                        )
                        print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                        print(f"Extracted Output: '{answer_output}'")

                    cor += answer_output == answer
                    # cor_cot += cot_output == answer_cot

                    pbar.update(1)
                    pbar.set_description(
                        f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                    )

                pbar.close()
                print(f"Device {rank}: Cor={cor}, Total={total}")

            dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            # cor_cot = cor_cot.item()
            cor = cor.item()
            total = total.item()
            if rank == 0:
                print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
                # print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
            sys.stdout.flush()

            if wandb_run:
                wandb_run.log({"eval/answer_acc": cor / total})

            if configs.only_eval:
                break

            dist.barrier()
            if (
                cor / total > best_acc
                and configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()

                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    print("saving model.")

                best_acc = cor / total

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# torchrun --nnodes 1 --nproc_per_node 4 run.py args/sprosqa_coconut_graph_2l_8h_768d_fixed_bug_mix_no_answer_coconut_0.0_50ep.yaml