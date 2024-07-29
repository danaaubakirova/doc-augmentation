import argparse
import os
import json
import tarfile
import tempfile
from PIL import Image
import concurrent.futures
from typing import List, Tuple
from webdataset import TarWriter
import shutil
from functools import partial
from render_text_on_image import mask_and_replace_text
from text_aug import TextAugmenter
import logging
import time
from datetime import datetime
from threading import Lock

# Configure logging

logging.basicConfig(level=logging.INFO, filename=datetime.now().strftime('/fsx/dana_aubakirova/data-logs/70_data_aug_%H_%M_%d_%m_%Y.log'), filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ThreadSafeSet:
    def __init__(self):
        self.set = set()
        self.lock = Lock()
    
    def add(self, item):
        with self.lock:
            if item in self.set:
                return False
            self.set.add(item)
            return True
    
    def __contains__(self, item):
        with self.lock:
            return item in self.set
write_lock = Lock()
written_files = ThreadSafeSet()
Image.MAX_IMAGE_PIXELS = None
_logger = logging.getLogger('endless_attempts')
def process_pair(tiff_path: str, json_path: str, writers: List[TarWriter], pair_base_name: str, font_dir: str, rand_aug: TextAugmenter):
    _logger.info(f"Processing pair: {tiff_path} and {json_path}")
    try:
        image = Image.open(tiff_path)
        with open(json_path, 'r') as json_file:
            metadata = json.load(json_file)
        
        images, jsons = mask_and_replace_text(image, metadata, font_dir, rand_aug)

        for version, (img, metadata) in enumerate(zip(images, jsons)):
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            key_name = f"{pair_base_name}_{version}"
            if not written_files.add(key_name):
                _logger.info(f"Skipping writing {key_name} as it's already processed.")
                continue
            sample_to_write = {"__key__": key_name, "tif": img.getvalue(), "json": metadata_bytes}
            with write_lock:
                writers[version].write(sample_to_write)
                _logger.info(f"Wrote {key_name} to tar.")
            #sample_to_write = {"__key__": key_name, "tif": img.getvalue(), "json": metadata_bytes}
            #with write_lock:
                #writers[version].write(sample_to_write)  
                #_logger.info(f"Wrote {key_name} to tar.")
    except Exception as e:
        _logger.error(f"Error processing image {tiff_path}: {e}", exc_info=True)

def process_tar_file(tar_path: str, temp_dir: str, final_dir: str, n_parallel_files_per_shard: int, font_dir: str, rand_aug: TextAugmenter):

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=temp_dir)
    files = os.listdir(temp_dir)
    # check if logic here is sound - maybe an assert to check if pairs are indeed matched pairwise
    tiffs = sorted(f for f in files if f.endswith('.tif'))
    jsons = sorted(f for f in files if f.endswith('.json'))
    total_length = len(jsons)

    writers = [TarWriter(f"{final_dir}/{os.path.splitext(os.path.basename(tar_path))[0]}_{i}.tar") for i in range(3)]
    pair_paths = [(os.path.join(temp_dir, tiffs[i]), os.path.join(temp_dir, jsons[i])) for i in range(len(tiffs))]
    # pre-pass writers and temp dir to a partial func in order to map the pairs we want to it
    # each writer will be copied n_parallel_shards times, but that should be an ok tradeoff
    process_function = partial(process_pair_wrapper, writers=writers, font_dir=font_dir, rand_aug=rand_aug)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel_files_per_shard) as executor:
            executor.map(process_function, pair_paths)
    except Exception as e:
        _logger.error(f"Failed to process tar file {tar_path}: {e}", exc_info=True)    
    finally:
        for writer in writers:  
            writer.close()
        shutil.rmtree(temp_dir)
        _logger.info("Cleaning up resources.")

def process_pair_wrapper(pair_paths, writers, font_dir, rand_aug):

    tiff_path, json_path = pair_paths
    pair_base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    process_pair(tiff_path, json_path, writers, pair_base_name, font_dir, rand_aug)

def process_directory(current_shard: str, final_dir: str, n_parallel_shards: int, n_parallel_files_per_shard: int, font_dir: str, rand_aug: TextAugmenter):
    _logger.info("Starting to process directory.")
    start_time = time.time()
    #tar_files = sorted(f for f in os.listdir(directory) if f.endswith('.tar'))
   # all_tar_files = [f for f in os.listdir(directory) if f.endswith('.tar')]
    #print(all_tar_files)
    #tar_files = sorted(f for f in all_tar_files if f.endswith('.tar') and '00010' <= f[10:15] <= '00029')
    #print(tar_files)
    #tar_files = sorted(directory)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel_shards) as executor:
        futures = []
        #for tar in tar_files:
        temp_dir = tempfile.mkdtemp()
        futures.append(executor.submit(process_tar_file, current_shard, temp_dir, final_dir, n_parallel_files_per_shard, font_dir, rand_aug))
        concurrent.futures.wait(futures)

    end_time = time.time()
    _logger.info(f"Finished processing directory in {end_time - start_time:.2f} seconds.")
'''

def process_single_tar(current_shard: str, final_dir: str,  n_parallel_files_per_shard: int, font_dir: str, rand_aug: TextAugmenter):
    _logger.info("Starting to process tar file.")
    start_time = time.time()
    # Extract the base name of the current shard
    shard_base_name = os.path.basename(current_shard).replace('.tar', '')
    
    # Create a unique temporary directory for each job
    temp_dir = tempfile.mkdtemp(prefix=f"{shard_base_name}_")

   # temp_dir = tempfile.mkdtemp()
    process_tar_file(current_shard, temp_dir, final_dir, n_parallel_files_per_shard, font_dir=font_dir, rand_aug=rand_aug)

    end_time = time.time()
    _logger.info(f"Finished processing tar file in {end_time - start_time:.2f} seconds.")
'''
if __name__ == "__main__":
    """
    Untested script to extract, multiply and rewrite tar shards. Should be used in an sbatch script for good measure
    """
    parser = argparse.ArgumentParser(description="Process webdataset shards in parallel.")
    parser.add_argument("--current_shard", type=str, help="The shard to process when using job arrays.")
    parser.add_argument("--final_dir", type=str, help="Directory to write the processed shards to.")
    parser.add_argument("--n_parallel_shards", type=int, default=3, help="Number of processes to assign shards to.")
    parser.add_argument("--n_parallel_files_per_shard", type=int, default=12, help="Number of threads to process files within a shard.")
    #parser.add_argument("--font_dir", type=str, default="/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", help="The font directory that you want to use.")
    parser.add_argument("--font_dir", type=str, default="/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", help="The font directory that you want to use.")
    parser.add_argument("--stopwords_path", type=str, default="/fsx/dana_aubakirova/stopwords.txt", help="Path to the stopwords file for text processing.")
    parser.add_argument("--pos_model_path", type=str, default="/fsx/dana_aubakirova/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger", help="Path to the Stanford POS tagger model file.")
    parser.add_argument("--pos_jar_path", type=str, default="/fsx/dana_aubakirova/stanford-postagger-2018-10-16/stanford-postagger.jar", help="Path to the Stanford POS tagger jar file.")

    args = parser.parse_args()
    _logger.info('Data augmentation starts')
    rand_aug = TextAugmenter(args.stopwords_path, args.pos_model_path, args.pos_jar_path)
    process_directory(args.current_shard, args.final_dir, args.n_parallel_shards, args.n_parallel_files_per_shard, args.font_dir, rand_aug)
    _logger.info('Data augmentation stops')
