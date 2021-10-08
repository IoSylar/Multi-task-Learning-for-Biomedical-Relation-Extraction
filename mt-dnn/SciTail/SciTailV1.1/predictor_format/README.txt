AllenNLP predictors work only with JSONL format. This folder contains the SciTail train/dev/test in JSONL format
so that it can be loaded into the predictors. Each line is a JSON object with the following keys:

gold_label : the example label from {entails, neutral}
sentence1: the premise
sentence2: the hypothesis
sentence2_structure: structure from the hypothesis in the same format as described in dgem_format/README.txt

Examples:
{"question":"Which of the following processes is responsible for changing liquid water into water vapor?","gold_label":"entails","sentence1":"Facts: Liquid water droplets can be changed into invisible water vapor through a process called evaporation .","answer":"evaporation.","sentence2":"Evaporation is responsible for changing liquid water into water vapor.","sentence2_structure":"Evaporation<>is<>responsible for changing liquid water into water vapor"}
{"question":"Which characteristic do single-celled organisms and multicellular organisms have in common?","gold_label":"neutral","sentence1":"Multicellular organisms that have a eukaryotic cell type, mitochondria and a complex nervous system.","answer":"Both have a way to get rid of waste materials.","sentence2":"Single-celled organisms and multicellular organisms have this in common: both have a way to get rid of waste materials.","sentence2_structure":"both<>have<>a way to get rid of waste materials$$$Single-celled organisms and multicellular organisms<>have<>this"}

