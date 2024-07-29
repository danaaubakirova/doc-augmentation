import random
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import multiprocessing
from functools import partial
import io
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Any
import time 
import logging
from datetime import datetime
#logging.basicConfig(filename=datetime.now().strftime('/fsx/dana_aubakirova/data-logs/page_level_data_aug_%H_%M_%d_%m_%Y.log'), level=logging.INFO,
#                    format='%(asctime)s:%(levelname)s:%(message)s')

def create_in_memory_tiff(images):# -> Optional[Image.Image]:
    if not images:
        return None  # Handle empty image list
    buffer = io.BytesIO()
    if len(images) == 1:
        images[0].save(buffer, format='TIFF', compression='tiff_deflate')
    else:
        images[0].save(buffer, format='TIFF', save_all=True, append_images=images[1:], compression='tiff_deflate')
    return buffer

def modify_text_section(img, page: Dict[str, Any], selected_indexes: List[int], font_path: str, aug_func):
    draw = ImageDraw.Draw(img)
    for selected_index in selected_indexes:
        selected_bbox = page['bbox'][selected_index]
        left, top, width_norm, height_norm = selected_bbox
        width, height = img.size
        x0, y0 = left * width, top * height
        box_width, box_height = width_norm * width, height_norm * height
        font = ImageFont.truetype(font_path, int(0.90 * box_height))
        old_text = page['text'][selected_index]
        choices = ["swap", "deletion", "insertion", "kreplacement"]
        choice = random.choice(choices)
        if choice == "kreplacement" and len(old_text)<=50:
            choice = random.choice(choices[:-1])
        new_text = aug_func.random_aug(old_text, 0.10, choice)  # Pass choice along with parameters
        page['text'][selected_index] = new_text
        draw.rectangle([x0, y0, x0 + box_width, y0 + box_height], fill="white")
        draw.text((x0, y0), new_text, fill="black", font=font)
    return img, page
def process_page(image: Image.Image, page, font_path: str,  aug_func):
    
    if len(page['text']) < 20:
        return zip(*[(image.copy(), page) for _ in range(3)])  # Skip pages with too little text
    lines = len(page['text'])
    selected_lines = int(max(1, 0.4 * lines))
    splits = [random.sample(range(lines), min(selected_lines, lines)) for i in range(3)]
    # Update to pass aug_func and stopwords to modify_text_section
    partial_modify_text_section = partial(modify_text_section, font_path=font_path, aug_func=aug_func)
    image_copies = [image, image.copy(), image.copy()] #TODO more flexible
    page_copies = [page, deepcopy(page), deepcopy(page)]
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(partial_modify_text_section, img, page, split) for img, page, split in zip(image_copies, page_copies, splits)]

    results = [future.result() for future in futures]
    if results:
        images, pages = zip(*results)
        return images, pages
    else:
        return None
'''
def mask_and_replace_text(sample, metadata, font_path, aug_func):
    versioned_images = [[] for _ in range(3)]  # List of lists to hold three versions separately
    versioned_anns = [[] for _ in range(3)]
    
    annotation = metadata['pages']
    #logging.info(f'Processing document with {len(annotation)} pages') 
    for i in range(len(annotation)):
        sample.seek(i)
        image_array = sample.copy()
        #breakpoint()
        modified_imgs, pages = process_page(image_array, annotation[i], font_path, aug_func)
        for idx, (img, pg) in enumerate(zip(modified_imgs, pages)):
            versioned_images[idx].append(img)  # Append each version to its corresponding list
            versioned_anns[idx].append(pg)
           
    multi_page_tiffs = [create_in_memory_tiff(img_list) for img_list in versioned_images]
    annotation = [{'pages': ann} for ann in versioned_anns]
    return multi_page_tiffs, annotation
'''
def process_page_wrapper(args):
    image_array, page_annotation, font_path, aug_func = args
    return process_page(image_array, page_annotation, font_path, aug_func)

def mask_and_replace_text(sample, metadata, font_path, aug_func):
    annotation = metadata['pages']
    num_threads = min(3, max(len(annotation), 1))  # Define number of threads
    
    images = []
    # this avoids passing the whole sample to each thread
    # but it creates a BIG memory overhead potentially, let's see

    for i in range(len(annotation)):
        sample.seek(i)
        # this should change the .seek()... to a list of images containing a copy of current sample
        images.append(sample.copy())

    # then we can build args to pass in parallel
    args = [(images[i], annotation[i], font_path, aug_func) for i in range(len(annotation))]
    # the executor.map will do the rest and return the results in order

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_page_wrapper, args))
    
    versioned_images = [[], [], []]
    versioned_anns = [[], [], []]
    for modified_imgs, pages in results:
        for idx, (img, pg) in enumerate(zip(modified_imgs, pages)):
            versioned_images[idx].append(img)
            versioned_anns[idx].append(pg)

    multi_page_tiffs = [create_in_memory_tiff(img_list) for img_list in versioned_images]
    annotations = [{'pages': ann} for ann in versioned_anns]
    
    return multi_page_tiffs, annotations
