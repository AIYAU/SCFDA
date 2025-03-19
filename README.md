# SCFDA
This is a code demo for the paper "Spectral Contextual Learning and Frequency Domain Alignment for Cross-Domain Few-Shot Hyperspectral Image Classification".

## Requirements

- CUDA = 12.2

- python = 3.9.18 

- torch = 1.11.0+cu113 

- transformers = 4.30.2

- sklearn = 0.0.post9

- numpy = 1.26.0

## Datasets

- source domain dataset
  - Chikusei

- target domain datasets
  - Indian Pines
  - Houston
  - Salinas

An example datasets folder has the following structure:

```
datasets
├── Chikusei_imdb_128_7_7.pickle
├── Chikusei_raw_mat
│   ├── HyperspecVNIR_Chikusei_20140729.mat
│   └── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
├── IP
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── Houston
│   ├── data.mat
│   ├── mask_train.mat
│   └── mask_test.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── WHU-Hi-LongKou
    ├── WHU_Hi_LongKou.mat
    └── WHU_Hi_LongKou_gt.mat
```

## Pretrain model

You can download the pre-trained model of Base Bert, bert-base-uncased, at huggingface, and move to folder `pretrain-model`.

An example pretrain-model folder has the following structure:

```
pretrain-model
└── bert-base-uncased
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

## Usage

1. Download the required source and target datasets and move to folder `datasets`.

2. Download the required Base Bert pre-trained model and move to folder `pretrain-model`.
3. Run `train.py`. 
