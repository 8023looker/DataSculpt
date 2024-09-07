# Phase 1. Semantic Clustering
# 1.1. decide initial cluster number
# depend on CPU ray cluster
import faiss
import numpy as np
from tqdm import tqdm
import ujson
import ast
import ray
import os

# FAISS
dimension = 1024 # embedding dimension using bge-m3 is 1024

# density list
density_list = []

# biuld FAISS index for each file
@ray.remote
def build_sample_index(args): # need args[0], args[1]
    # Create an index  
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT) # 64

    input_file_path, faiss_output_path = args[0], args[1]
    try:
        embedding_list = [] # 2-dimension (for adding FAISS index)
        with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    embedding_list.append(ast.literal_eval(line_dict["vector_encoded"]))
                except ValueError as e:
                    print(f"JSON parser error: {e}")
        if len(embedding_list) > 0: 
            vectors = np.array(embedding_list).astype('float32')
            index.add(vectors)
    
    except OSError as e:  
        print(f"An error occurred: {e}")
    
    # Save the index to disk
    faiss.write_index(index, faiss_output_path)
    
@ray.remote
def compute_density(args): # need args[0], args[1]
    index = faiss.read_index(args[1]) # read FAISS index
    
    part_density = 0
    with open(args[0], "r", encoding="utf-8", errors="ignore") as fin:
        for idx, line in enumerate(fin):
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                search_vector = np.array([ast.literal_eval(line_dict["vector_encoded"])])
                D, I = index.search(search_vector, 2) # nearest 2
                line_dict["cluster_distance"] = float(D[0][0])
                part_density = (line_dict["cluster_distance"] + part_density * idx) / (idx + 1)
            
            except ValueError as e:
                print(f"JSON 解析错误: {e}")
                
    density_list.append(part_density)

# store status dictionary
def process_incremental(status_all, status_dict):
    status_all.append(status_dict)
    return status_all
    
if __name__ == "__main__":
    ray_log = get_logger("ray_faiss_index", "node_num_decision.log")
    ray_log.info("--- start new ray task ---")
    ray.init(address="auto")
    
    raw_data_folder = "/path/to/embedding_folder/" # path to the embedding folder (/xxx/DataSculpt/data_sample/embedding_rs/)
    faiss_output_folder = "/path/to/part_faiss/" # path to the part_faiss folder (/xxx/DataSculpt/data_sample/faiss/part_faiss/)

    args_list = []
    for dirpath, dirnames, filenames in os.walk(raw_data_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            args_list.append((file_path, faiss_output_folder + filename)) # [input_file_path, faiss_index_output_path]
    
    ray_log.info(f"total num of files: {len(args_list)}")
    
    status_all = []
    # step1: build faiss index
    rs_ids = [build_sample_index.remote(args) for args in args_list] # .options(memory=2.5 * 1024 * 1024 * 1024)
    while len(rs_ids):
        done_ids, rs_ids = ray.wait(rs_ids)
        status_all = process_incremental(status_all, ray.get(done_ids[0]))
        ray_log.info(status_all[-1])
        ray_log.info(
            f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
        )
        
    # step2: compute density
    rs_ids = [compute_density.remote(args) for args in args_list] # .options(memory=2.5 * 1024 * 1024 * 1024)
    while len(rs_ids):
        done_ids, rs_ids = ray.wait(rs_ids)
        status_all = process_incremental(status_all, ray.get(done_ids[0]))
        ray_log.info(status_all[-1])
        ray_log.info(
            f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
        )

    cluster_density = density_list.mean() # mean of density list
    print(f"Semantic Density: {cluster_density}") # This result is all you need!
    with open("semantic_density.txt", "w") as f:
        f.write(str(cluster_density))