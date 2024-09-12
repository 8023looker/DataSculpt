# Phase 1. Semantic Clustering
# 1.3. Varient of ISODATA
# depend on CPU ray cluster

import faiss
import numpy as np
from tqdm import tqdm
import random
import ujson
import ast
import subprocess
import time
import datetime
import errno
import ray
import glob
import os
import sys
import shutil

import logging
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y%m%d %H:%M:%S")
def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# FAISS
dimension = 1024 # embedding dimension using bge-m3 is 1024
faiss_object_id = ""


def write_with_retry(file, data, retries=10, delay=1):
    for _ in range(retries):
        try:
            file.write(data)  
            return
        except OSError as e:
            if e.errno == errno.EIO:
                print("I/O error, retrying...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Failed to write data after multiple attempts")


class ISODATAVarient:
    def __init__(self, args): # delta: distance threshold, epsilon: alteration threshold, T: maximum iterations
        self.faiss_output_folder, self.intermediate_cluster_output_folder, self.cluster_output_folder, self.delta, self.epsilon, self.t = args
        if self.t == 0:
            self.build_faiss_index(False) # build initially
            # self.faiss_index = ray.get(faiss_object_id) # faiss index (from ray object)
        self.faiss_index = faiss.read_index(self.faiss_output_folder + "faiss_index")   
        self.alteration_dst = 0 # distance threshold
    
    
    @ray.remote(num_cpus=2)
    def write_isodata_flag(self, input_file_path=None):
        """ flag the documents via ISODATA Varient Algorithm """
        node_info_path = self.get_node_info_path()
        
        filename = input_file_path.split("/")[-1]
        output_file_path = self.intermediate_cluster_output_folder + filename
        
        with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin, open(
                  output_file_path, "a", encoding="utf-8", errors="ignore") as fout1, open(
                  node_info_path, "a", encoding="utf-8", errors="ignore") as fout2: # update center node info file (add)
            for idx, line in enumerate(fin):
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    search_vector = np.array([ast.literal_eval(line_dict["vector_encoded"])])
                    
                    D, I = index.search(search_vector, 2) # nearest
                    line_dict["cluster_distance"] = float(D[0][0])
                    if line_dict["cluster_distance"] < self.delta: # threshold
                        try:
                            write_with_retry(fout2, ujson.dumps(line_dict, ensure_ascii = False) + "\n")  
                        except Exception as e:
                            print(f"An error occurred: {e}")

                    line_dict["cluster_id"] = str(I[0][0])  
                    fout1.write(ujson.dumps(line_dict, ensure_ascii = False) + "\n")
            
                except ValueError as e:
                    print(f"JSON parser error: {e}")
                    
                    
    @ray.remote(num_cpus=2)
    def write_isodata_result(self, input_file_path=None):
        """ write ISODATA results to clusters """
        node_info_path = self.get_node_info_path()
        with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    with open(self.cluster_output_folder + line_dict["cluster_id"] + ".jsonl", "a", encoding="utf-8", errors="ignore") as fout:
                        fout.write(ujson.dumps(line_dict, ensure_ascii = False) + "\n")
            
                except ValueError as e:
                    print(f"JSON parser error: {e}")
    
    
    def build_faiss_index(self, update=False):
        """ build / update FAISS index """
        # Create an index
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT) # 64

        node_info_path = self.get_node_info_path()
        iter_idx = int(node_info_path.split(".jsonl")[0].split("_")[-1])
        node_info_path_update = nodes_info_folder + f"sample_nodes_{iter_idx + 1}.jsonl"
        
        embedding_list = [] # 2-dimension (for adding FAISS index)
        with open(node_info_path, "r", encoding="utf-8", errors="ignore") as fin: # original node info
            for idx, line in enumerate(fin):
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    embedding_list.append(ast.literal_eval(line_dict["vector_encoded"]))
                    if update: # update node info file
                        with open(node_info_path_update, "a", encoding="utf-8", errors="ignore") as fout:
                            fout.write(ujson.dumps(line_dict, ensure_ascii = False) + "\n")
                except ValueError as e:
                    print(f"JSON parser error: {e}")
        if len(embedding_list) > 0:
            vectors = np.array(embedding_list).astype('float32')
            index.add(vectors)
        
        # Save (update) the index to disk
        faiss.write_index(index, self.faiss_output_folder + "faiss_index")
        
    
    def recalculate_centroid(self):
        """ recalculate centroid """
        node_info_path = self.get_node_info_path()
        
        embedding_list = [] # 2-dimension (for adding FAISS index)
        for dirpath, dirnames, filenames in os.walk(self.cluster_output_folder):
            for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                file_path = os.path.join(dirpath, filename)
                cluster_embedding_list = []
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
                    for idx, line in enumerate(fin):
                        try:
                            line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                            cluster_embedding_list.append(ast.literal_eval(line_dict["vector_encoded"]))
                        except ValueError as e:
                            print(f"JSON parser error: {e}")
                if len(cluster_embedding_list) > 0: # 2-dimensional
                    vector_mean_update = np.mean(np.array(cluster_embedding_list), axis=0) # 1-dimensional
                    
                    line_idx = filename.split(".jsonl")[0]
                    result = subprocess.run(['sed', '-n', f'{line_idx}p', node_info_path], capture_output=True, text=True)
                    vector_mean_origin = ast.literal_eval(ujson.loads(result.stdout)["vector_encoded"])
                    
                    self.alteration_dst += (1 - cosine_similarity(np.array(vector_mean_update), np.array(vector_mean_origin)))
                    # self.alteration_dst += np.linalg.norm(np.abs(np.array(vector_mean_update) - np.array(vector_mean_origin)))
                    
                    embedding_list.append(vector_mean_update)
        
            self.alteration_dst /= len(filenames)
        
        if len(embedding_list) > 0: 
            vectors = np.array(embedding_list).astype('float32')
            index.add(vectors)
            
        # Save (update) the index to disk
        faiss.write_index(index, self.faiss_output_folder + "faiss_index")
    
    
    def cluster_merge(self):
        """ merge clusters """
        node_info_path = self.get_node_info_path()
        iter_idx = int(node_info_path.split(".jsonl")[0].split("_")[-1])
        node_info_path_update = nodes_info_folder + f"sample_nodes_{iter_idx + 1}.jsonl"

        with open(node_info_path, "r", encoding="utf-8", errors="ignore") as fin, open( # original node info
                  node_info_path_update, "a", encoding="utf-8", errors="ignore") as fout: # updated node info
            for idx, line in enumerate(fin):
                try:
                    line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                    search_vector = np.array([ast.literal_eval(line_dict["vector_encoded"])])
                    
                    D, I = index.search(search_vector, 2) # nearest
                    line_dict["cluster_distance"] = float(D[0][0])
                    if line_dict["cluster_distance"] < self.delta or idx <= int(I[0][0]): # threshold
                        try:
                            write_with_retry(fout, ujson.dumps(line_dict, ensure_ascii = False) + "\n")  
                        except Exception as e:
                            print(f"An error occurred: {e}")
                    else:
                        print(f"Merge cluster! Cluster distance: {line_dict['cluster_distance']}")
                
                except ValueError as e:
                    print(f"JSON parser error: {e}")
    
               
    def check_dst(self):
        """ check alteration distance """
        if self.alteration_dst < self.epsilon:
            return True
        else:
            shutil.rmtree(self.intermediate_cluster_output_folder)
            os.mkdir(self.intermediate_cluster_output_folder, exist_ok=True)
            return False
        
    
    def get_node_info_path(self):
        """ obtain the latest node info path """
        nodes_info_folder = self.faiss_output_folder + "iterations/"
        if self.t == 0:
            node_info_path = nodes_info_folder + "sample_nodes_0.jsonl"
        else:
            iter_idx = 0
            for dirpath, dirnames, filenames in os.walk(nodes_info_folder):
                for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                    iter_idx = max(iter_idx, int(filename.split(".jsonl")[0].split("_")[-1]))
            node_info_path = nodes_info_folder + f"sample_nodes_{iter_idx}.jsonl"
        return node_info_path # the latest node info path


# compute cosine similarity between two vectors
def cosine_similarity(v1, v2): # input: np.array([])
    # v1: center_vector, v2: doc_vector
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if not (all(element == 0 for element in v1) or all(element == 0 for element in v2)) else 1 # the bigger, the better


# store status dictionary
def process_incremental(status_all, status_dict):
    status_all.append(status_dict)
    return status_all

if __name__ == "__main__":
    delta, epsilon, iter_T = sys.argv[1:] # delta: distance threshold, epsilon: alteration threshold, iter_T: maximum iterations
    # delta, epsilon, iter_T = 0.1, 0.1, 10
    ray_log = get_logger("ray_faiss_index", "ray_isodata_varient.log")
    ray_log.info("--- start new ray task ---")
    # ray.init(address="auto")
    
    output_root_folder = "/data_cfs_new/keerlu/src/DataSculpt/data_sample/intermediate_cluster/" # your output data folder path ("xxx/DataSculpt/data_sample/intermediate_cluster/")
    os.makedirs(output_root_folder, exist_ok=True)
    
    for t in range(int(iter_T)): # iter_T > 1
        """
        faiss_output_folder: "/xxx/DataSculpt/data_sample/faiss/",
        intermediate_cluster_output_folder: "/xxx/DataSculpt/data_sample/intermediate_cluster/",
        cluster_output_folder: "/xxx/DataSculpt/data_sample/cluster_rs/",
        raw_data_folder: "/xxx/DataSculpt/data_sample/embedding_rs/" if t == 0 else "/xxx/DataSculpt/data_sample/intermediate_cluster/"
        """
        faiss_output_folder, intermediate_cluster_output_folder, cluster_output_folder = "/data_cfs_new/keerlu/src/DataSculpt/data_sample/faiss/", "/data_cfs_new/keerlu/src/DataSculpt/data_sample/intermediate_cluster/", "/data_cfs_new/keerlu/src/DataSculpt/data_sample/cluster_rs/"
        os.makedirs(faiss_output_folder, exist_ok=True)
        os.makedirs(intermediate_cluster_output_folder, exist_ok=True)
        os.makedirs(cluster_output_folder, exist_ok=True)

        raw_data_folder = "/data_cfs_new/keerlu/src/DataSculpt/data_sample/embedding_rs/" # your input data folder path ("xxx/DataSculpt/data_sample/embedding_rs/")
        
        ISODATAVarient_obj = ISODATAVarient((
                                faiss_output_folder,
                                intermediate_cluster_output_folder,
                                cluster_output_folder,
                                delta, # distance threshold
                                epsilon, # alteration threshold
                                t
                            ))
        
        """ build write_isodata_flag tasks """
        rs_ids = []
        for dirpath, dirnames, filenames in os.walk(raw_data_folder):
            for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                file_path = os.path.join(dirpath, filename)
                rs_ids.append(ISODATAVarient_obj.write_isodata_flag.remote(file_path))
                    
        ray_log.info(f"total num of tasks: {len(rs_ids)}")
        
        # read faiss index and put it onto ray object
        # index = faiss.read_index(faiss_output_folder + "faiss_index")
        # ray_log.info(f"Building FAISS Index done.")
        # faiss_object_id = ray.put(index)
        
        status_all = []
        while len(rs_ids):
            done_ids, rs_ids = ray.wait(rs_ids)
            status_all = process_incremental(status_all, ray.get(done_ids[0]))
            ray_log.info(status_all[-1])
            ray_log.info(
                f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
            )

        """ recalculate_centroid """
        ISODATAVarient_obj.recalculate_centroid()
        
        """ cluster_merge """
        ISODATAVarient_obj.cluster_merge()
        
        """ check_dst """
        if ISODATAVarient_obj.check_dst():
            """ build write_isodata_result tasks """
            rs_ids = []
            for dirpath, dirnames, filenames in os.walk(intermediate_cluster_output_folder):
                for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                    file_path = os.path.join(dirpath, filename)
                    rs_ids.append(ISODATAVarient_obj.write_isodata_result.remote(file_path))
            ray_log.info(f"total num of tasks: {len(rs_ids)}")     
            status_all = []
            while len(rs_ids):
                done_ids, rs_ids = ray.wait(rs_ids)
                status_all = process_incremental(status_all, ray.get(done_ids[0]))
                ray_log.info(status_all[-1])
                ray_log.info(
                    f"total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done."
                )    
            break
        else:
            ISODATAVarient_obj.build_faiss_index(True) # update FAISS Index && node info file