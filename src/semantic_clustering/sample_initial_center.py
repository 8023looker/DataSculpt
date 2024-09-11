# Phase 1. Semantic Clustering
# 1.2. initial cluster center

import numpy as np
from tqdm import tqdm
import random
import ujson
import os
import ast
import subprocess
import time
import datetime
import ray

import logging
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y%m%d %H:%M:%S")
def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

@ray.remote
def sample_node(args):
    # get total line number 
    line_num = subprocess.run(['wc', '-l', args[0]], stdout=subprocess.PIPE)  
    line_count = int(line_num.stdout.split()[0])
    print(f"total line number: {line_count}")
        
    chosen_prob = 100 / line_count # initial sample rate
    with open("semantic_density.txt", "r") as f:
        chosen_prob = float(f.read())

    with open(args[0], "r", encoding="utf-8", errors="ignore") as fin, open(
              args[1], "a", encoding="utf-8", errors="ignore") as fout:
        for idx, line in enumerate(fin):
            if random.random() < chosen_prob:
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    fout.write(ujson.dumps(line_dict, ensure_ascii = False) + "\n") # 写入 sampled node 文件
                except ValueError as e:
                    print(f"JSON 解析错误: {e}")

# store status dictionary
def process_incremental(status_all, status_dict):
    status_all.append(status_dict)
    return status_all

def merge_sample_nodes(sample_nodes_folder, output_merged_folder):
    # merge all sample nodes
    with open(output_merged_folder + "sample_nodes_0.jsonl", "a", encoding="utf-8", errors="ignore") as fout:
        for dirpath, dirnames, filenames in os.walk(sample_nodes_folder):
            for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
                    for idx, line in enumerate(fin):
                        try:
                            line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                            fout.write(ujson.dumps(line_dict, ensure_ascii = False) + "\n")
                        except ValueError as e:
                            print(f"JSON parser error: {e}")

if __name__ == "__main__":
    ray_log = get_logger("ray_initial_center", "ray_initial_center.log")
    ray_log.info("--- start new ray task ---")
    ray.init(address="auto")
    
    raw_data_folder = "/path/to/embedding_folder/" # path to the embedding folder (/xxx/DataSculpt/data_sample/embedding_rs/)
    output_root_folder, output_merged_folder = "/path/to/sample_nodes/", "/path/to/merged_nodes/" # /xxx/DataSculpt/data_sample/faiss/center_nodes/, /xxx/DataSculpt/data_sample/intermediate_cluster/
    os.makedirs(output_root_folder, exist_ok=True)

    args_list = []
    for dirpath, dirnames, filenames in os.walk(raw_data_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            args_list.append((file_path, output_root_folder + filename)) # [input_file_path, output_file_path]
                
    ray_log.info(f"total num of files: {len(args_list)}")
    
    status_all = []
    # build tasks
    rs_ids = [sample_node.remote(args) for args in args_list] # .options(memory=2.5 * 1024 * 1024 * 1024)
    while len(rs_ids):
        done_ids, rs_ids = ray.wait(rs_ids)
        status_all = process_incremental(status_all, ray.get(done_ids[0]))
        ray_log.info(status_all[-1])
        ray_log.info(
            f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
        )

    # merge all sample nodes
    merge_sample_nodes(output_root_folder, output_merged_folder)