# Phase 0: data preprocessing
# doctruncate && embedding
import os
import json
import ujson
from pathlib import Path
import zipfile
from FlagEmbedding import BGEM3FlagModel
import copy
import ray
import sys
import numpy as np
import logging
import time

formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y%m%d %H:%M:%S")
def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


@ray.remote(num_cpus=2, num_gpus=1)
class ModelBGEM3:
    def __init__(self, model_path='BAAI/bge-m3'):
        self.model = BGEM3FlagModel(model_path,  
                                    use_fp16=True)
    def get_paragraphs(self, file_content): # content → a single file
        # context length
        context_len = 16384 # enter your context length here, default to 16K
        # store the result_dict, text
        text_list, raw_text_list = [], []
        for line in file_content: # 1st round
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))
                
                text = line_dict["text"] if "text" in line_dict else (line_dict["content"] if "content" in line_dict else "")
                while text:
                    raw_text_list.append(text[:context_len])
                    text = text[context_len:]
                
            except ValueError as e: 
                print(f"JSON parser error: {e}")
                
        # use bge_m3 model to embed
        embedding_list = self.model.encode(raw_text_list, 
                                           batch_size=3, 
                                           max_length=8192,
                                        )['dense_vecs'] # 2nd round
        
        # store the results
        emb_index = 0
        for line in file_content: # 3rd round         
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace('\\/', '/'))              
                # general
                line_dict["source_id"] = line_dict.pop("docid") if "docid" in line_dict else ""
                
                # for Chinese
                if "text" in line_dict:
                    # modify the key in line_dict      
                    line_dict["chunk"] = line_dict.pop("text") if "text" in line_dict else ""
                    line_dict.pop("content") if "content" in line_dict else None # del "content" for Chinese       
                # for English
                elif "content" in line_dict:
                    # modify the key in line_dict            
                    line_dict["chunk"] = line_dict.pop("content") if "content" in line_dict else ""
                # for "" (no content), abandon
                
                text = copy.deepcopy(line_dict["chunk"])
                # truncate the chunk
                while text:
                    # embedding vector
                    line_dict["vector_encoded"] = str(embedding_list[emb_index].tolist())       
                    emb_index += 1
                    # chunk
                    line_dict["chunk"][:context_len] = text[:context_len]
                    text_list.append(line_dict)
                    text = text[context_len:]
                                
            except ValueError as e: 
                print(f"JSON parser error: {e}")
        
        return text_list
                
    def handle_file(self, args):
        i_file, o_file, log = args
        status_dict = {'i_file': i_file, 'flag': True}
        print(i_file)

        # write in result .json data
        with open(Path(i_file), 'r', encoding='utf-8', errors='ignore') as fin, open(Path(o_file), 'w', encoding = 'utf-8') as fout:
            doc = fin.readlines()
            rs_list = self.get_paragraphs(doc) # api file_content
            json.dump(rs_list, fout, ensure_ascii=False)
        
        return status_dict 


def main():
    ray_log = get_logger('ray_emb', 'ray.log.emb')
    ray_log.info('--- start new ray task ---')
    ray.init(address='auto')
    
    input_folder = "/your/input/data_folder_path/" # your input data folder path ("xxx/DataSculpt/data_sample/input/")
    output_root_folder = "/your/output/data_folder_path/" # your output data folder path ("xxx/DataSculpt/data_sample/embedding_rs/")
    
    args_list = []
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            args_list.append((file_path, # input_file_path
                              output_root_folder + filename, # output_file_path
                              None # logs
                            ))
    
    ray_log.info(f'total num of files: {len(args_list)}')

    n_model = 104
    models = [ ModelBGEM3.remote() for _ in range(n_model) ]
    rs_ids = []

    status_all = []
    def process_incremental(status_all, status_dict): 
        status_all.append(status_dict)
        return status_all

    for i in range(0, len(args_list), n_model):
        rs_ids.extend([ m.handle_file.remote(args) for m, args in zip (models, args_list[i : i + n_model]) ])
        
    while len(rs_ids):
        done_ids, rs_ids = ray.wait(rs_ids)
        status_all = process_incremental(status_all, ray.get(done_ids[0]))
        ray_log.info(status_all[-1])
        ray_log.info(f'total {len(args_list)} tasks, {len(rs_ids)} tasks waiting, {len(status_all)} tasks done.')

    
if __name__ == "__main__":
    main()
