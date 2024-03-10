# MOMENTA

This is the repo for "MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets" accepted at Findings of EMNLP '21.

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
<br>
<!-- _Pl Note: A sample only 50 total data samples are linked as part of this repo._ -->
<ol>
    <li>It contains code for loading pre-saved ROI and entity features, which can be loaded if available.</li>
    <li>Otherwise the code for extracting features on-demand is also included.</li>
    <li>For initializing dataset and data loader for pytorch: Load the data-set for training and testing as per the requirement of the run.</li>
    <li><i>Experimental settings:</i><br>
Configurations for the binary/multi-class setting (training/testing/evaluation) has to be considered as per the requirement, code blocks for which are provided and suitably commented out.</li>
</ol>

## Dataset, Features and Meta-info:<br>

<!-- Datasets (Sample) <br>
Harm-P: https://drive.google.com/file/d/10Otu_cAZSX1tXMh2puqUl5dqLRAxr3ui/view?usp=sharing <br>
Harm-C: https://drive.google.com/file/d/1X_Ty1DsuV2hD3naiKeih5bMvRIyCuNxy/view?usp=sharing -->

<!-- Complete dataset links to be released upon approval -->

<strong>Please note: TWO versions of Harm-P data for "Harmfulness" are provided as part of HarMeme-V0 and HarMeme-V1, respectively. We recommend using <a href="https://github.com/LCS2-IIITD/MOMENTA/tree/main/HarMeme_V1">HarMeme-V1</a> for updated and correct version for "Harmfulness" data.</strong> 

<ol>
    <li><strong><u>HarMeme-V0:</u></strong> <strong><i>CAUTION! OBSOLETE FOR HARM-P - Contains duplicates in Harm-P.</i></strong> Thanks to <a href="https://github.com/mingshanhee">mingshanhee</a> and <a href="https://github.com/uprihtness">uprihtness</a> for pointing out the discrepancies. See the upgraded version (V1) below for the deduplicated version. HarMeme-V0 (OBSOLETE) content can be accessed via the following links:</li>
<ul>
<li><a href="https://drive.google.com/file/d/1fw850yxKNqzpRpQKH88D13yfrwX1MLde/view?usp=sharing">Harm-P (US Politics)</a>, <a href="https://drive.google.com/file/d/1dxMrnyXcED-85HCcQiA_d5rr8acwl6lp/view?usp=sharing">Harm-C (Covid-19)</a></li>
<li><a href="https://drive.google.com/file/d/1KBltp_97CJIOcrxln9VbDfoKxbVQKcVN/view?usp=sharing">Entity features</a>, <a href="https://drive.google.com/file/d/1KRAJcTme4tmbuNXLQ72NTfnQf3x2YQT_/view?usp=sharing">ROI features</a>, <a href="https://drive.google.com/file/d/1xeviXtHE46md3usybEO2FIAcRkBmXZN7/view?usp=sharing">ROI + Entity features</a></li>
</ul>
    <li><strong><u>HarMeme-V1:</u></strong> Updated + Complete Version. Check the folder named: "HarMeme_V1" in this repo for data files. <strong>Please refer <a href="https://drive.google.com/file/d/1fw850yxKNqzpRpQKH88D13yfrwX1MLde/view?usp=sharing">Harm-P (US Politics)</a>, <a href="https://drive.google.com/file/d/1dxMrnyXcED-85HCcQiA_d5rr8acwl6lp/view?usp=sharing">Harm-C (Covid-19)</a> links for meme images </strong>. For additional details about HarMeme-V1, refer the <a href="https://github.com/LCS2-IIITD/MOMENTA/blob/main/HarMeme_V1/README.md">README</a> in it's repo folder. The repo folder contains the following:</li>
    <ul>
        <li>Annotations (Same format as V0: [id, image, labels, text]), but complete set.</li>
        <li>Meta-info (Collected using <a href="https://cloud.google.com/vision?hl=en">GCV API</a>): Meme id, OCR Text, Web Entities, Best labels, Titles, Objects, ROI Info.</li>
    </ul>
</ol>    
