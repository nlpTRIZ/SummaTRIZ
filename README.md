# SummaTRIZ

This code is for ICMLA 2020 paper **SummaTRIZ: Summarization Networks for Mining Patent Contradiction**

## 1) Download Dataset

Google drive: https://drive.google.com/file/d/1FSDhlQHSBzq5WKJp4ZwnNiJCY22IqYre/view?usp=sharing

## 2) Preprocess your data

Follow instructions in section Preprocessing:
https://github.com/nlpguarino/Patent_preprocessing

## 3) Uses

To train model:
```bash
python3 train.py -task ext -mode train -bert_data_path ../data_patents/STATE_OF_THE_ART -model_path ../models -lr 2e-4 -visible_gpus 0 -report_every 100 -save_checkpoint_steps 100 -train_steps 2000  -max_pos 1500 -finetune_bert False
```
To finetune pretrained model (on CNN/DM):
```bash
python3 train.py -task ext -mode train -bert_data_path ../data_patents/STATE_OF_THE_ART -model_path ../models -lr 2e-4 -visible_gpus 0 -report_every 100 -save_checkpoint_steps 100 -train_steps 2000  -max_pos 1500 -finetune_bert False -train_from ../models/model.pt
```
Download pretrained model on CNN/DM:
https://drive.google.com/file/d/13JHBwWB_G4PiLb6-POdpc8hNlO4DWcjB/view?usp=sharing
