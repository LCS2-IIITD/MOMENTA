# MOMENTA

This is the repo for "MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets" accepted (conditionally) at Findings of EMNLP '21.

setting up dependencies
```
if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"    
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"    
elif CUDA_version == "10.2":
    torch_version_suffix = ""    
else:
    torch_version_suffix = "+cu110"
```
For installing CLIP
```
! pip3 install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html ftfy regex --user
! wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O bpe_simple_vocab_16e6.txt.gz
```
For sentence transformer: Follow steps from https://github.com/UKPLab/sentence-transformers


## Instructions
The .py contains the exhaustive set of steps required to be run in sequence.
For data-set loading:<br>
_Pl Note: For demo purposes, only 50 total data samples are linked as part of this repo._
1. It contains code for loading pre-saved ROI and entity features, which can be loaded if available.
2. Otherwise the code for extracting features on-demand is also included.

Dataset and Features (links):<br>

Datasets (Sample) <br>
Harm-P: https://drive.google.com/file/d/10Otu_cAZSX1tXMh2puqUl5dqLRAxr3ui/view?usp=sharing <br>
Harm-C: https://drive.google.com/file/d/1X_Ty1DsuV2hD3naiKeih5bMvRIyCuNxy/view?usp=sharing

<!-- Complete dataset links to be released upon approval -->

<!-- Datasets (Complete) <br>
Harm-P: https://drive.google.com/file/d/1fw850yxKNqzpRpQKH88D13yfrwX1MLde/view?usp=sharing <br>
Harm-C: https://drive.google.com/file/d/1dxMrnyXcED-85HCcQiA_d5rr8acwl6lp/view?usp=sharing -->

Entity features: https://drive.google.com/file/d/1KBltp_97CJIOcrxln9VbDfoKxbVQKcVN/view?usp=sharing <br>
ROI features: https://drive.google.com/file/d/1KRAJcTme4tmbuNXLQ72NTfnQf3x2YQT_/view?usp=sharing <br>
ROI + Entity features: https://drive.google.com/file/d/1xeviXtHE46md3usybEO2FIAcRkBmXZN7/view?usp=sharing <br>

For initializing dataset and data loader for pytorch

1. Load the data-set for training and testing as per the requirement of the run.

Experimental setting<br>
Configurations for the binary/multi-class setting (training/testing/evaluation) has to be considered as per the requirement, code blocks for which are provided and suitably commented out.
