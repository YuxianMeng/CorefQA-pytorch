# CorefQA: Coreference Resolution as Query-based Span Prediction

The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**CorefQA: Coreference Resolution as Query-based Span Prediction** <br>
Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1911.01746)<br>
If you find this repo helpful, please cite the following:
```latex
@article{wu2019coreference,
  title={Coreference Resolution as Query-based Span Prediction},
  author={Wu, Wei and Wang, Fei and Yuan, Arianna and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.01746},
  year={2019}
}
```
For any question, please feel free to post Github issues.

## Contents 
- [Overview](#overview)
- [Experimental Results](#experimental-results)
- [Data Preprocess](#data-preprocess)
- [Replicate Experimental Results](#replicate-experimental-results)
    - [Install Package Dependencies](#install-package-dependencies)
    - [Train CorefQA Model](#train-corefqa-model)
    - [Prediction](#prediction)
- [Evaluating the Trained Model](#evaluating-the-trained-model)
- [Descriptions of Directories](#descriptions-of-directories)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## Overview 


## Experimental Results 

| Model          | F1 (%) |
| -------------- |:------:|
| CorefQA + SpanBERT-base  | 79.9  |
| CorefQA + SpanBERT-large | 83.1   |



## Data Preprocess 
1. Download the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) dataset.
2. Split train/dev/test datasets and preprocess the official release `ontonotes-release-5.0` for coreference resolution annotations. <br>
Run `./scripts/preprocess_conll_data.sh  <PATH-TO-ontonotes-release-5.0> <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <PATH-TO-CorefQA-REPOSITORY>`. <br>
E.g.: `./scripts/preprocess_conll_data.sh /home/shannon/ontonotes-release-5.0 /home/shannnon/conll12_coreference_data /home/shannon/CorefQA`
3. Generate and save training datasets to TFRecord files. <br>
Run `./scripts/generate_train_data.sh <PATH-TO-SAVE-CoNLL-FORMAT-DATASETS> <LANGUAGE> <NUMBER-of-SLIDING-WINDOW-SIZE>`<br>
E.g.: `./scripts/generate_train_data.sh /home/shannon/conll12_coreference_data english 384`


## Replicate Experimental Results 

### Install Package Dependencies 

* Install packages dependencies via : `pip install -r requirements.txt`
* GPU or TPU <br> 
  - V100 (16RAM Memory) with Tensorflow 1.15 Python 3.6 CUDA 10.0 (NVIDIA Docker Image 19.2). 
  - Cloud TPU v2-8 device with Tensorflow 1.15 Python 3.5. 

### Train CorefQA Model

1. Download Data Augmentation Models <br>
Run `./scripts/download_qauad2_finetune_model.sh <model-scale> <path-to-save-model>` to download finetuned SpanBERT on SQuAD2.0. <br>
The `<model-scale>` should take the value of `[base, large]`. <br>
The `<path-to-save-model>` is the path to save finetuned spanbert on SQuAD2.0 datasets. <br>
2. Train CoNLL-12 Coreference Resolution Model. <br> 
If using TPU, please run `./scripts/tpu/train_tpu.sh`<br>
If using GPU, please run `./scripts/gpu/train_spanbert.sh`. 

### Prediction

* Save the text for prediction in a txt file. If the text contains speaker name information, wrap the speaker with `<speaker></speaker>` and put it in front of its utterence. For example:
```text
<speaker> Host </speaker> A traveling reporter now on leave and joins us to tell her story. Thank you for coming in to share this with us.
```
* run `python3 ./run/evaluate.py <experiment> <input_file> <output_file>` will save the prediction results in `<output_file>`, The prediction for each instance is a list of clusters，each cluster is a list of mentions. Each mention is (text, (span_start, span_end)). For example:
```python
[[('A traveling reporter', (26, 46)), ('her', (81, 84)), ('you', (98, 101))]]
```


## Descriptions of Directories

Name | Descriptions 
----------- | ------------- 
log | A collection of training logs in experments.   
script |  Shell files help to reproduce our results.  
data_preprocess | Files to generate train/dev/test datasets. 
metric | Evaluation metrics for CorefQA. 
model | An implementation of CorefQA based on Pytorch.
module | Components for building CorefQA model.  
run | Train / Evaluate MRC-NER models.
config | Config files for BERT models. 



## Acknowledgement
Thanks to the previous work `https://github.com/mandarjoshi90/coref`.

## Contact

Feel free to discuss papers/code with us through issues/emails!

