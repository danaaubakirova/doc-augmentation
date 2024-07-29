# Data Augmentation Pipeline 🤗 

## Overview

This repository contains a data augmentation pipeline designed for augmenting text within document images. The pipeline processes TIFF images and associated JSON files that describe text lines and their bounding boxes within the document images.
The code is the part of the [HuggingFace 🤗  internal repo](https://github.com/huggingface/pixparse-data)
## Code Structure

- **text_aug.py**: Contains all the text augmentation functions.
- **render_text_on_image.py**: Manages multiprocessing for modifying text and rendering it back to the document.
- **augment_idl_shards_utils.py**: Handles multiprocessing at the tar file level.
- **Examples**: Provides some original and modified samples.

## Pipeline Description

### Input

The input consists of TIFF images (multiple pages per document) along with a JSON file. The JSON file contains text lines and their corresponding bounding boxes (i.e., text locations within the document image).

### Identifying Text Proportion to Mask

We determine the proportion of the text to be masked. For example, with a hyperparameter of 0.4, we compute the number of lines to be modified:

```python
selected_lines = int(max(1, 0.4 * len(lines)))
```

The indices of the selected lines of text are passed to the `modify_text_section` function where the core modifications occur.

## Text Augmentation

We randomly select one of the four text augmentation functions: "swap", "deletion", "insertion", "kreplacement".

- **Keyword replacement**: Requires more text to identify important keywords and replace them with synonyms. Therefore, we skip keyword replacement for text with fewer than 50 characters.

The chosen augmentation function is then applied to the selected lines. In `aug_func`, the argument `0.1` indicates the proportion of words to be modified within a selected line (e.g., one word out of ten in a line will be altered randomly). All text-related functions are located in the `text_aug.py` file.

```python
choices = ["swap", "deletion", "insertion", "kreplacement"]
choice = random.choice(choices)
if choice == "kreplacement" and len(old_text) <= 50:
    choice = random.choice(choices[:-1])
new_text = aug_func.random_aug(old_text, 0.10, choice)
```

## Rendering Modified Text

1. Obtain the modified text.
2. Replace the old text in the JSON file with the new one.
3. Mask the old text with a white patch using the bounding box coordinates.
4. Render the new text onto the document image on top of the white patches.

The font used is `/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf`, which is available on Linux. The font size is determined by 0.95 times the height of the bounding box of the line.

## Output

The output consists of rendered images of the document and the modified JSON.

## Additional Notes

- The data format used is TIFF, which contains all the pages of a single document. Each page is processed individually and then assembled back to form a single document.
- The code is parallelized, creating three versions of each document concurrently. Top-level multiprocessing is also implemented to process tar files in parallel.
- Pages with insufficient text to apply augmentation are skipped:

```python
if len(page['text']) < 20:
    return zip(*[(image.copy(), page) for _ in range(3)])
```

## Future Considerations

We are considering adding a small-size LLM to replace the current NLTK-based modifications. This could enhance the accuracy of text modifications by applying random swap, deletion, insertion, paraphrase, and synonym replacement.

## References

For text augmentation code, you can take a look at the original repository: [GenAug](https://github.com/styfeng/GenAug).

