# Multi-task-Learning-for-few-shot-Biomedical-Relation-Extraction
Official repository for the [paper](https://link.springer.com/article/10.1007/s10462-023-10484-6) Multi-task learning for few-shot Biomedical Relation Extraction.
# Requirements
You need to run the requirements.txt file in the mt-dnn folder. Then you will need to uninstall and install the latest version of apex.
```python
!pip install -r requirements.txt
!pip3 uninstall apex
!git clone https://www.github.com/nvidia/apex
%cd apex
!python3 setup.py install
```
# Dataset and Preprocessing
The datasets used for experimentation are DDI2013, Chemprot, and I2B2-2010-RE. They belong to the BLUE benchmark. The download and preprocessing of all the tasks in the BLUE benchmark are performed through scripts provided by [ClinicalNLP].(https://github.com/facebookresearch/bio-lm). 
```python
!bash download_all_task_data.sh
!bash preprocess_all_classification_datasets.sh
```
A license is required for the I2B2-2010 dataset. Therefore, the processed datasets have been included in the "Dataset" folder. It should be noted that this folder will also contain the data necessary for few-shot learning. The sizes of the shots used are: 1-10-50-100-1000.
# Model and pre-trained model
The model used is based on [MT-DNN](https://github.com/namisan/mt-dnn).It is used in cases of single-task learning, multi-task learning, adversarial training, knowledge distillation, and few-shot learning.
The pre-trained model is taken from the paper [ClinicalNLP](https://github.com/facebookresearch/bio-lm). You can choose to download it from there, but it will then need to be converted from the tensorflow .bin format to the model.pt format. This has already been done and the base model can be downloaded from my Mega account: https://mega.nz/folder/Bc0CXJaa#qhY1Cp4CGaaBaOGU__bFig.
# Training
The model training in various modes is based on [MT-DNN](https://github.com/namisan/mt-dnn) through train.py. The metrics and losses of the tasks can be modified in the tutorial_task_def.yml file.
```python
chemprot:
  data_format: PremiseOnly
  enable_san: false
  labels:
  - CPR:3
  - CPR:4
  - CPR:5
  - CPR:6
  - CPR:9
  - CPRFalse
  metric_meta:
  - ACC
  - F1_chem
  - Precision_chem
  - Recall_chem
  loss: CeCriterion
  kd_loss: MseCriterion
  adv_loss: SymKlCriterion
  n_class: 6
  task_type: Classification 
```
Tokenization is required before performing the training. In particular, the model to be used must be selected, it can be a local backbone as in our case or a model available on [Huggingface].(https://huggingface.co/models). 
```python
!python prepro_std.py --model mt_dnn_models/robertaFB  --root_dir tutorials/ --task_def tutorials/tutorial_task_def.yml
```
Multi-task training can be performed as follows, by selecting the output directory, using the default mt-dnn parameters, and adding those listed. Single-task training can be performed by removing the two datasets and the explicit parameters that refer to MTL from the parameters.
```python
!python train.py --task_def tutorials/tutorial_task_def.yml --data_dir tutorials/mt_dnn_models/robertaFB   --train_datasets ddi2013,chemprot,i2b2 --test_datasets ddi2013,chemprot,i2b2 --epochs=10 --batch_size=8 --bert_model_type="roberta"  --encoder_type=2  --output_dir="Addestramento" --init_checkpoint="mt_dnn_models/robertaFB" --grad_clipping=1.0 --adam_eps=1e-7  --seed=2010 --mtl_opt=1  #--model_ckpt="SingleI2B2SEED2010/model_2.pt" --resume
```
To use adversarial training, the parameters to add are:
```python
--adv
--adv_opt=1 
```
To perform knowledge distillation, it is necessary to create a tokenized file (in .json format) for the dataset with a soft label column. The soft label column can be obtained by making predictions of the model on the training set. Then, given the new tokenized file as input, the parameter to add for training is:
```python
--mkd_opt=1
```
# Evaluation
The prediction phase is carried out through the predict.py script. When performing the prediction of a single task model, it can be run without any further difficulties. You can choose to perform the prediction on the test also epoch by epoch during training, but this is too computationally expensive, so it is recommended to avoid it. The prediction phase of a multi-task model requires adjustments, in particular, because of the way the model is made, the prediction is always and exclusively carried out on the first output layer (for example, related to the DDI2013 task). To perform the prediction also on the other tasks of a multi-task model, the predict.py file must be modified, exchanging the order of the layers and selecting as the first output layer the one related to the task to be predicted.
```python
def load_model_ckpt(model, checkpoint):
        model_state_dict = torch.load(checkpoint)
        if 'state' in model_state_dict:
            sd=model_state_dict
            #print(sd['state'])
            #sd['state'][:-3]
            #SE uso chemprot devo scambiare il layer 0 con il 1 che è il rispettivo layer in MTDNN
            sd['state']['scoring_list.0.weight']=sd['state']['scoring_list.1.weight']
            sd['state']['scoring_list.0.bias']=sd['state']['scoring_list.1.bias']
             #SE uso I2B2 devo scambiare il layer 0 con il 2 che è il rispettivo layer in MTDNN
            #sd['state']['scoring_list.0.weight']=sd['state']['scoring_list.2.weight']
            #sd['state']['scoring_list.0.bias']=sd['state']['scoring_list.2.bias']
            
            #print(model.fc.weight)
            #self.network.load_state_dict(model_state_dict['state'], strict=False)
            model.load_state_dict(sd['state'], strict=False)
```
Predizione
```python
!python predict.py --task_def tutorials/tutorial_task_def.yml --task chemprot --task_id=0 --prep_input="tutorials/robertaFB/chemprot_train.json" --score="ScoreEsempio.txt"  --model_checkpoint="MTLEsempio/model_9.pt" --checkpoint="ChemprotSingleF2000/model_9.pt"  --with_label
```
# Baseline 
The few shot learning baselines that were compared are:  [PET](https://github.com/timoschick/pet),  [Protonet](https://github.com/jingyuanz/protonet-bert-text-classification) , [Siamese](https://github.com/subhasisj/Few-Shot-Learning)
# Notebook
An example notebook that includes the task similarity phase is available.

# Cite the article
@article{MultitaskFewShot,
  title={Multi-task learning for few-shot biomedical relation extraction},
  author={Moscato V, Napolano G, Postiglione M, Sperlì G},
  journal={Artificial Intelligence Review},
  url={https://doi.org/10.1007/s10462-023-10484-6},
  year={2023}
}
