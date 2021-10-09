# Multi-task-Learning-for-Biomedical-Relation-Extraction
Repository relativa all'implementazione : Multi-task learning for Biomedical Relation Extraction
# Requirements
Si deve eseguire il file requirements.txt presente nella cartella mt-dnn. Successivamente sarà necessario disinstallare ed installare l'ultima versione di apex
```python
!pip install -r requirements.txt
!pip3 uninstall apex
!git clone https://www.github.com/nvidia/apex
%cd apex
!python3 setup.py install
```
# Dataset and Preprocessing
I dataset utilizzati per la sperimentazione sono DDI2013, Chemprot ed I2B2-2010-RE. Essi appartengono al benchmark BLUE. Si effettua il download ed il preprocessing di tutti i task del benchmark BLUE tramite script messi a disposizione da [ClinicalNLP](https://github.com/facebookresearch/bio-lm). 
```python
!bash download_all_task_data.sh
!bash preprocess_all_classification_datasets.sh
```
Per il dataset I2B2-2010 è necessaria la richiesta di licenza. Perciò si è scelto di inserire i dataset già processati nella cartella Dataset. Da notare che in tale cartella ci saranno anche i dati necessari al few shot learning. Le dimensioni degli shot usati sono: 1-10-50-100-1000.
# Model and pre-trained model
Il modello utilizzato è basato su [MT-DNN](https://github.com/namisan/mt-dnn). Esso è utilizzato nei casi di single task learning, multi-task learning, adversarial training, knowledge distillation e few shot learning.
Il modello pre-addestrato è prelevato dal paper [ClinicalNLP](https://github.com/facebookresearch/bio-lm). Si può scegliere di effettuarne il download da lì, però dovrà essere successivamente convertito dal formato tensorflow .bin al formato model.pt . Questo è stato già fatto ed il modello base è scaricabile dal mio account Mega : https://mega.nz/folder/Bc0CXJaa#qhY1Cp4CGaaBaOGU__bFig .
# Training
Il training del modello nelle varie modalità è quello di [MT-DNN](https://github.com/namisan/mt-dnn) tramite train.py. Le metriche e le loss dei task sono modificabili dal file tutorial_task_def.yml 
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
E' necessaria la tokenizzazione prima di eseguire il training. In particolare, dev'essere scelto il modello da utilizzare,può essere una backbone locale come nel nostro caso oppure un modello presente su [Huggingface](https://huggingface.co/models). 
```python
!python prepro_std.py --model mt_dnn_models/robertaFB  --root_dir tutorials/ --task_def tutorials/tutorial_task_def.yml
```
Si può eseguire l'addestramento multi-task come di seguito, scegliendo la directory di output, utilizzando i parametri di default di mt-dnn ed aggiungendo quelli elencati. L'addestramento single task può essere eseguito eliminando dai parametri i due dataset ed i parametri espliciti che si riferiscono al MTL.
```python
!python train.py --task_def tutorials/tutorial_task_def.yml --data_dir tutorials/mt_dnn_models/robertaFB   --train_datasets ddi2013,chemprot,i2b2 --test_datasets ddi2013,chemprot,i2b2 --epochs=10 --batch_size=8 --bert_model_type="roberta"  --encoder_type=2  --output_dir="Addestramento" --init_checkpoint="mt_dnn_models/robertaFB" --grad_clipping=1.0 --adam_eps=1e-7  --seed=2010 --mtl_opt=1  #--model_ckpt="SingleI2B2SEED2040/model_2.pt" --resume
```
Per utilizzare l'adversarial training i parametri da aggiungere sono:
```python
--adv
--adv_opt=1 
```
Per realizzare la knowledge distillation è necessario creare per il dataset un file tokenizzato (formato .json) avente una colonna soft label.
La colonna softlabel è possibile ottenerla effettuando le predizioni del modello sul training set. Successivamente dato in ingresso il nuovo file tokenizzato, il parametro da aggiungere per l'addestramento è:
```python
--mkd_opt=1
```
# Evaluation
La fase di predizione viene effettuata tramite lo script predict.py. Quando dev'essere effettuata la predizione di un modello single task,può essere eseguito senza ulteriori difficoltà. Si può scegliere di effettuare la predizione sul test anche in fase di training epoca per epoca. Questo però risulta troppo oneroso dal punto di vista computazionale, perciò si consiglia di evitarlo. La fase di predizione di un modello multi-task necessità di accorgimenti , in particolare per come è fatto il modello si va ad effettuare la predizione sempre ed unicamente sul primo layer di uscita (ad esempio relativo al task DDI2013). Per effettuare la predizione anche sugli altri task di un modello multi-task dev'essere modificato il file predict.py, scambiando l'ordine dei layer e selezionando come primo layer di uscita quello relativo al task da predire.
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
Le baseline di few shot learning con cui si è effettuato il confronto sono:  [PET](https://github.com/timoschick/pet),  [Protonet](https://github.com/jingyuanz/protonet-bert-text-classification) , [Siamese](https://github.com/subhasisj/Few-Shot-Learning)
# Notebook
E' stato reso disponibile un notebook di esempio che comprende anche la fase di similarità tra i task.
