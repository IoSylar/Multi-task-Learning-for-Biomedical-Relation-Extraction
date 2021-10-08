import argparse

from data_utils import load_score_file
from experiments.exp_def import TaskDefs

parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
parser.add_argument("--task", type=str)
parser.add_argument("--add_soft_label", action="store_true",
                    help="without this option, we replace hard label with soft label")

parser.add_argument("--std_input", type=str)
parser.add_argument("--score", type=str)
parser.add_argument("--std_output", type=str)

args = parser.parse_args()

task_def_path = args.task_def
task = args.task
task_defs = TaskDefs(task_def_path)

n_class = task_defs.get_task_def(task).n_class
print(str(n_class))
#sample_id_2_pred_score_seg_dic = load_score_file(args.score, n_class)
sample_id_2_pred_score_seg_dic = load_score_file(args.score, n_class)
#print(str(sample_id_2_pred_score_seg_dic[0]))
#print(str(sample_id_2_pred_score_seg_dic[0][1][0]))
with open(args.std_output, "w", encoding="utf-8") as out_f:
    for line in open(args.std_input, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        #print(fields)
        #sample_id = int(fields[0])
        sample_id = fields[0]
        #print(str(sample_id))
        #print(str(sample_id_2_pred_score_seg_dic['15652']))
       # target_score_idx = 1  # TODO: here we assume binary classification task
        #target_score_idx = 
        #score=[]
    
        score0 = sample_id_2_pred_score_seg_dic[sample_id][1][0]
        score1 = sample_id_2_pred_score_seg_dic[sample_id][1][1]
        score2 = sample_id_2_pred_score_seg_dic[sample_id][1][2]
        score3 = sample_id_2_pred_score_seg_dic[sample_id][1][3]
        score4 = sample_id_2_pred_score_seg_dic[sample_id][1][4]
        ###Chemprot
        score5 = sample_id_2_pred_score_seg_dic[sample_id][1][5]
        ###
        score6 = sample_id_2_pred_score_seg_dic[sample_id][1][6]
        score7 = sample_id_2_pred_score_seg_dic[sample_id][1][7]
        score8 = sample_id_2_pred_score_seg_dic[sample_id][1][8]
        #print(str(score0[0]))
        #print(str(score))
        if args.add_soft_label:
            #print(str(score))
            #DDI
            #fields = fields[:2] + [str(str('[')+ str(score0) + str(',') + str(score1) + str(',') + str(score2) + str(',') + str(score3)+ str(',') + str(score4)+ str('] '))] +  fields[2:]
            #Chemprot
            #fields = fields[:2] + [str(str('['))+ str(score0) + str(',') + str(score1) + str(',') + str(score2) + str(',') + str(score3)+ str(',') + str(score4) + str(',') + str(score5)+ str(']')] +  fields[2:]
            ##I2B2
            fields = fields[:2] + [str(str('[')+ str(score0) + str(',') + str(score1) + str(',') + str(score2) + str(',') + str(score3)+ str(',') + str(score4)+ str(',') + str(score5)+ str(',')+str(score6)+ str(',')+ str(score7)+ str(',')+ str(score8)+ str('] '))] +  fields[2:]

        #else:
            #fields[1] = str(score)
        out_f.write("\t".join(fields))
        out_f.write("\n")
