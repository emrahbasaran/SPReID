#  Human Semantic Parsing for Person Re-identification
Code for our CVPR 2018 [paper](https://arxiv.org/abs/1804.00216) - Human Semantic Parsing for Person Re-identification </br></br>
We have used [Chainer framework](https://chainer.org/) for the implementation. SPReID<sup>w/fg</sup> and SPReID<sup>w/fg-ft</sup> results mentioned in Table 5 (with weight sharing setting) in the paper can be reproduced using this code. To download the semantic parsing model (LIP_iter_30000.chainermodel), please use this [link](https://www.dropbox.com/s/22relvz5o9a3n05/LIP_iter_30000.chainermodel?dl=0). 

## Directories & Files
```shell
/
├── checkpoints/  # checkpoint models are saved into this directory
│
├── data/dump/  # inceptionv3 weights pre-trained on imagenet
│
├── evaluation_features/ # extracted features are saved into this directory
│
├── evaluation_list/ # there are two image lists to extract features for each evaluation datasets, one for gallery and one for query
│   ├── cuhk03_gallery.txt
│   ├── cuhk03_query.txt
│   ├── duke_gallery.txt
│   ├── duke_query.txt
│   ├── market_gallery.txt
│   └── market_query.txt
│
├── train_list/ # image lists to train the models
│   ├── train_10d.txt # training images collected from 10 datasets
│   ├── train_cuhk03.txt # training images from cuhk03
│   ├── train_duke.txt # training images from duke
│   └── train_market.txt # training images from market
│
├── LIP_iter_30000.chainermodel # download this model using this [link](https://www.dropbox.com/s/22relvz5o9a3n05/LIP_iter_30000.chainermodel?dl=0)
├── datachef.py
├── main.py
└── modelx.py
```


## Train
```shell 
cd $SPREID_ROOT
# train SPReID on 10 datasets
python main.py --train_set "train_10d" --label_dim "16803" --scales_reid "512,170" --optimizer "lr:0.01--lr_pretrained:0.01" --dataset_folder "/path/to/the/dataset"
# fine-tune SPReID on evaluation datasets (Market-1501, DukeMTMC-reID, CUHK03) with high-resolution images
python main.py --train_set "train_market" --label_dim_ft "751" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --max_iter "50000" --dataset_folder "/path/to/the/dataset" --model_path_for_ft "/path/to/the/model"
python main.py --train_set "train_duke" --label_dim_ft "702" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --max_iter "50000" --dataset_folder "/path/to/the/dataset" --model_path_for_ft "/path/to/the/model"
python main.py --train_set "train_cuhk03" --label_dim_ft "1367" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --max_iter "50000" --dataset_folder "/path/to/the/dataset" --model_path_for_ft "/path/to/the/model"
```
## Feature Extraction
```shell 
cd $SPREID_ROOT
# Extract features using the model trained on 10 datasets. You should run this command two times for each dataset using --eval_split "DATASET_gallery" and --eval_split "DATASET_query"
python main.py --extract_features 1 --train_set "train_10d" --eval_split "market_gallery" --scales_reid "512,170" --checkpoint 200000 --dataset_folder "/path/to/the/dataset"
# Extract features using the models trained on evaluation datasets.
python main.py --extract_features 1 --train_set "train_market" --eval_split "market_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder "/path/to/the/dataset"
python main.py --extract_features 1 --train_set "train_duke" --eval_split "duke_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder "/path/to/the/dataset"
python main.py --extract_features 1 --train_set "train_cuhk03" --eval_split "cuhk03_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder "/path/to/the/dataset"
```

## Results
<table>
  <tr>
    <th></th>
    <th colspan="2">Market-1501</th>
    <th colspan="2">CUHK03</th>
    <th colspan="2">DukeMTMC-reID</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>mAP(%)</th><th>rank-1</th>
    <th>mAP(%)</th><th>rank-1</th>
    <th>mAP(%)</th><th>rank-1</th>
  </tr>
  <tr>
    <th>SPReID<sup>w/fg</sup></th>
    <th>77.62</th><th>90.88</th>
    <th>-</th><th>87.69</th>
    <th>65.66</th><th>81.73</th>
  </tr>
  <tr>
    <th>SPReID<sup>w/fg-ft</sup></th>
    <th>80.54</th><th>92.34</th>
    <th>-</th><th>89.68</th>
    <th>69.29</th><th>83.80</th>
  </tr>
</table>

## Citation
```
@InProceedings{Kalayeh_2018_CVPR,
author = {Kalayeh, Mahdi M. and Basaran, Emrah and Gökmen, Muhittin and Kamasak, Mustafa E. and Shah, Mubarak},
title = {Human Semantic Parsing for Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
