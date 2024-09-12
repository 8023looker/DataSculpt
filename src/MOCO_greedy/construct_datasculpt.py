# Phase 2. Greedy Policy
# construct DatasSulpt
# depend on CPU ray cluster

import numpy as np
from tqdm import tqdm
import random
import ujson
import ast
import subprocess
import time
import datetime
import errno
import gc
import ray
import os
import sys

import logging
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y%m%d %H:%M:%S")
def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def write_with_retry(fout, context_window_line_dict):
    max_retries = 100000
    retry_delay = 1
    
    for _ in range(max_retries):  
        try:
            fout.write(ujson.dumps(context_window_line_dict, ensure_ascii = False) + "\n")  
            break
        except OSError as e:  
            print(f"An error occurred: {e}")  
            print("Retrying...")  
            time.sleep(retry_delay)
    else:  
        print("Failed to write after multiple attempts.")


@ray.remote(num_cpus=2)
def handle_token_num(args): # [input_file_path, output_file_path]
    start_time = time.time()
    status_dict = {"args": args}
    
    input_original_file, output_file_path, context_window_size = args
    
    token_num_list, token_num_dict = [], {}
    total_token_num, context_window_num = 0, 0
    with open(input_original_file, "r", encoding="utf-8", errors="ignore") as fin1:
        for idx, line in enumerate(fin1):
            line_dict = {}
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                line_token_num = int(line_dict["token_len"])
                total_token_num += line_token_num
                token_num_list.append([str(idx), line_dict, 0]) # (doc_idx: str, token_num: int, bag_idx: default_0)
            except ValueError as e:
                print(f"JSON parser error: {e}")
        context_window_num = total_token_num // context_window_size

    if context_window_num >= 1:
        token_num_list = sorted(token_num_list, key=lambda x: int(x[1]["token_len"]), reverse=True) # decending order
        
        # sample documents
        sample_token_num = context_window_num * context_window_size
        truncated_token_lst, truncated_idx = [], 0
        sum_token_num = 0
        for idx, token in enumerate(token_num_list):
            sum_token_num += int(token[1]["token_len"])
            if sum_token_num > sample_token_num:
                truncated_idx = idx + context_window_num
                break
        truncated_token_lst = token_num_list[:truncated_idx]
        
        # find context window (largest-fit)
        context_window_bags = [{"capacity": context_window_size,
                                "center_vector": [0 for _ in range(1024)], # embedding is a 1024-dimension vector
                                "doc_num": 0,
                                # "docs": [], # no need
                                } for _ in range(context_window_num)]

        for idx, doc in enumerate(truncated_token_lst):
            if not all_bag_capacities_zero(context_window_bags):
                # original version: find the largest space
                # bag_idx, _ = max(enumerate(context_window_bags), key=lambda x: x[1]["capacity"])
                # plus version: largest length + low cosine similarity
                bag_idx, _ = max(enumerate(context_window_bags), key=lambda x: compute_combined_score(doc[1], x[1]))
                # update
                # context_window_bags[bag_idx]["docs"].append(doc)
                truncated_token_lst[idx][2] = bag_idx # belong to bag_idx
                context_window_bags[bag_idx]["capacity"] -= int(doc[1]["token_len"])
                context_window_bags[bag_idx]["doc_num"] += 1
                context_window_bags[bag_idx]["center_vector"] = [(x * (context_window_bags[bag_idx]["doc_num"] - 1) + y) / context_window_bags[bag_idx]["doc_num"] for x, y in zip(context_window_bags[bag_idx]["center_vector"], ast.literal_eval(doc[1]["vector_encoded"]))]
            else:
                break
            
        token_num_dict = {x[0]: { # key: doc_idx(str)
            "token_num": int(x[1]["token_len"]), # int
            "bag_idx": x[2], # int
            } for x in truncated_token_lst} # truncate
        # release memory
        del token_num_list, truncated_token_lst
        gc.collect()
        
        context_window_dict = {f'{i}': {
            "total_token_num": 0,
            "docs": []
        } for i in range(context_window_num)} 
        with open(input_original_file, "r", encoding="utf-8", errors="ignore") as fin3:
            for idx, line in enumerate(fin3):
                if str(idx) in token_num_dict:
                    line_dict = {}
                    try:
                        line_dict = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                        belonging_bag_idx = str(token_num_dict[str(idx)]["bag_idx"])
                        context_window_dict[belonging_bag_idx]["docs"].append(line_dict)
                        context_window_dict[belonging_bag_idx]["total_token_num"] += token_num_dict[str(idx)]["token_num"]
                    except ValueError as e:
                        print(f"JSON parser error: {e}")
                        
                        
        # write to output file
        with open(output_file_path, "a", encoding="utf-8", errors="ignore") as fout:
            for key in context_window_dict:
                write_with_retry(fout, context_window_dict[key])
                # fout.write(ujson.dumps(context_window_dict[key], ensure_ascii = False) + "\n")
                
    else:
        print(f"token number is too small: {total_token_num}")
        
    end_time = time.time()
    status_dict["time"] = end_time - start_time
    return status_dict

           
def all_bag_capacities_zero(bags):
    for bag in bags:
        if bag["capacity"] > 0:
            return False
    return True


# compute cosine similarity between two vectors
def cosine_similarity(v1, v2): # input: np.array([])
    # v1: center_vector, v2: doc_vector
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if not (all(element == 0 for element in v1) or all(element == 0 for element in v2)) else 1 # the bigger, the better


# cpmpute the combined score
def compute_combined_score(doc, bag):
    # print("doc", doc)
    semantic_score = cosine_similarity(bag["center_vector"], ast.literal_eval(doc["vector_encoded"])) # average cosine similarity
    packing_score = bag["capacity"] # largest-fit packing
    return semantic_score * packing_score


# store status dictionary
def process_incremental(status_all, status_dict):
    status_all.append(status_dict)
    return status_all


if __name__ == "__main__":
    context_window_len = sys.argv[1:] # 16k, 32k, 64k
    ray_log = get_logger("ray_faiss_index", "construct_datasculpt.log")
    ray_log.info("--- start new ray task ---")
    # ray.init(address="auto")
    
    raw_data_folder = "/path/to/your/cluster_result/" # "/xxx/DataSculpt/data_sample/cluster_rs/"
    
    # output_root_folder = "/train_dataset/dlcresult/panda/data/data_icp/keerlu/semantic_cluster/data_preparation/16k/icp_plus/" # 等待创建完成
    output_root_folder = "/path/to/your/output/folder/" # "/xxx/DataSculpt/data_sample/output/data_sculpt/"
    os.makedirs(output_root_folder, exist_ok=True)
    
    args_list = []
    for dirpath, dirnames, filenames in os.walk(raw_data_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            args_list.append((file_path, output_root_folder + filename, context_window_len)) # [input_file_path, faiss_output_folder, output_file_path, singular_file_path]
                
    ray_log.info(f"total num of files: {len(args_list)}")
    
    status_all = []
    # build tasks
    rs_ids = [handle_token_num.remote(args) for args in args_list] # .options(memory=2.5 * 1024 * 1024 * 1024)
    while len(rs_ids):
        done_ids, rs_ids = ray.wait(rs_ids)
        status_all = process_incremental(status_all, ray.get(done_ids[0]))
        ray_log.info(status_all[-1])
        ray_log.info(
            f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
        )
    
    # debug
    # handle_token_num(("/data_train/data/keerlu/text_cluster/k_means/data_preparation/12.jsonl", 
    #               "/data_train/data/keerlu/text_cluster/k_means/data_preparation/12_rs.jsonl", 
    #               16000))