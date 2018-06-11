#  Human Semantic Parsing for Person Re-identification
Code for our CVPR 2018 paper - Human Semantic Parsing for Person Re-identification 

## Train
### Train Human Semantic Parsing
### Train SPReID
```shell 
cd $SPREID_ROOT
# train SPReID on 10 datasets
python main.py --train_set "train_10d" --label_dim "16803" --scales_reid "512,170" --optimizer "lr:0.01--lr_pretrained:0.01" --dataset_folder = "/path/to/the/dataset"
# fine-tune SPReID on evaluation datasets (Market-1501, DukeMTMC-reID, CUHK03) with high-resolution images
python main.py --train_set "train_market" --label_dim "751" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --dataset_folder = "/path/to/the/dataset"
python main.py --train_set "train_duke" --label_dim "702" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --dataset_folder = "/path/to/the/dataset"
python main.py --train_set "train_cuhk03" --label_dim "1367" --scales_reid "778,255" --optimizer "lr:0.01--lr_pretrained:0.001" --dataset_folder = "/path/to/the/dataset"
```
## Feature Extraction
```shell 
cd $SPREID_ROOT
# Extract features using the model trained on 10 datasets. You should run this command two times for each dataset using --eval_split "DATASET_gallery" and --eval_split "DATASET_query"
python main.py --extract_features 1 --train_set "train_10d" --eval_split "market_gallery" --scales_reid "512,170" --checkpoint 200000 --dataset_folder = "/path/to/the/dataset"
# Extract features using the models trained on evaluation datasets.
python main.py --extract_features 1 --train_set "train_market" --eval_split "market_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder = "/path/to/the/dataset"
python main.py --extract_features 1 --train_set "train_duke" --eval_split "duke_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder = "/path/to/the/dataset"
python main.py --extract_features 1 --train_set "train_cuhk03" --eval_split "cuhk03_gallery" --scales_reid "778,255" --checkpoint 50000 --dataset_folder = "/path/to/the/dataset"
```

## Directories & Files
```shell
/
├── checkpoints/  # checkpoint models are saved into this directory
│
├── data/dump/  # inceptionv3 weights pre-trained on imagenet
│
├── evaluation_features/ # extracted features are saved into this directory
│
├── evaluation_list/ # In this directory, there are two image lists to extract features for each evaluation datasets, one for gallery and one for query
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
├── README.md
├── datachef.py
├── main.py
└── modelx.py
```

## Results
<table style="width:100%">
  <tr>
    <th>Name</th>
    <th colspan="2">Telephone</th>
  </tr>
  <tr>
    <td>Bill Gates</td>
    <td>555 77 854</td>
    <td>555 77 855</td>
  </tr>
</table>

## Citation
```
@article{kalayeh2018human,
  title={Human Semantic Parsing for Person Re-identification},
  author={Kalayeh, Mahdi M and Basaran, Emrah and Gokmen, Muhittin and Kamasak, Mustafa E and Shah, Mubarak},
  journal={arXiv preprint arXiv:1804.00216},
  year={2018}
}
```
