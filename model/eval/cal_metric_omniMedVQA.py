import argparse
import json
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from glossary import *
import os
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    # parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-medplib-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--candidate_set', type=str, default=None, help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def get_name_type():
    root_path = '/public/home/s20213081508/huangxiaoshuang/data/OmniMedVQA/QA_information/Open-access'
    name_type_dict = {}
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                
                # Read each JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    if 'modality_type' in item:
                        if item['image_path'] not in name_type_dict:
                            name_type_dict[item['image_path']] = item['modality_type']
    return name_type_dict


def evaluate(test_dict_lst, args, criterion=None):
    print('pred file', args.pred)
    print('candidate_set', args.candidate_set)
    if args.candidate_set is not None:
        with open(args.candidate_set, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        candidate_set = []
        # for item in dataset:
        #     candidate_set.append(item['conversations'][1]['value'])
        
        for item in test_dict_lst:
            candidate_set.append(item['gt'])
        
        print(len(candidate_set))
        candidate_set = set(candidate_set)
        print('candidate_set',len(candidate_set))

    name_type_dict = get_name_type()
    closed_scores_modal_dict = {}


    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)


    eval_open = False
    eval_closed = False
    open_cnt = 0
    closed_cnt = 0
    for item in tqdm(test_dict_lst):
        gt_value = item['gt'].lower()
        pred_value = item['text'].lower()


        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if item['answer_type'].lower() in ['open','other','number']:
            eval_open = True
            # for open-ended question
            # if gt_value in pred_value:
            #     hit = 1.0
            # else:
            #     hit = 0.0
            # open_hit_scores['hit'].append(hit)

            if args.candidate_set is not None:
                open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate_set))
                open_hit_scores['q_id'].append(item['question_id'])

            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(item['question_id'])

            # import pdb; pdb.set_trace()

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(item['question_id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)
            open_cnt += 1

        elif item['answer_type'].lower() in ["yes/no", 'closed']:
            relat_img_path = item['image_path'].split('OmniMedVQA')[-1][1:]
            modality = name_type_dict[relat_img_path]
            if modality not in closed_scores_modal_dict:
                closed_scores_modal_dict[modality] = collections.defaultdict(list)


            eval_closed = True
            # for close-ended question (Yes/No)
            closed_scores_modal_dict[modality]['q_id'].append(item['question_id'])
            # if 'yes' in pred_value or 'no' in pred_value:
            #     if gt_value in pred_value:
            #         closed_scores['hit'].append(1)
            #     else:
            #         closed_scores['hit'].append(0)
            # else:
            #     closed_scores['hit'].append(0)
            if gt_value == pred_value:
                closed_scores_modal_dict[modality]['hit'].append(1)
            else:
                closed_scores_modal_dict[modality]['hit'].append(0)
            closed_cnt += 1
    
    print('open_cnt', open_cnt)
    print('closed_cnt', closed_cnt)
    if eval_open:
        # import pdb; pdb.set_trace()

        if args.candidate_set is not None:
            print(sum(open_hit_scores['hit']),len(open_hit_scores['hit']), len(open_hit_scores['q_id']))
            open_hit_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
        else:
            open_hit_score = 11
        exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
        f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
        precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
        recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

        bleu_score   = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
        bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
        bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
        bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])
    else:
        exact_score = 0.0
        f1_score = 0.0
        precision = 0.0
        recall = 0.0
        open_hit_score = 0.0
        bleu_score   = 0.0
        bleu_score_1 = 0.0
        bleu_score_2 = 0.0
        bleu_score_3 = 0.0

    mean_lst = []
    closed_scores_modal_dict_res = {}
    if eval_closed:
        for k in closed_scores_modal_dict.keys():
            closed_score = sum(closed_scores_modal_dict[k]['hit']) / len(closed_scores_modal_dict[k]['hit']) if len(closed_scores_modal_dict[k]['hit']) != 0 else 0.0
            closed_scores_modal_dict_res[k] = round(closed_score*100, 2)
            mean_lst.append(closed_score)
    else:
        closed_score = 0.0

    print(closed_scores_modal_dict_res)
    print('mean acc', sum(mean_lst)/len(mean_lst))


    num_close, num_open = len(closed_scores['hit']), len(exact_scores['q_id'])
    
    
    # print(f'num_open {num_open} || num_close {num_close}')

    # return tabulate(
    #     [
    #         ['exact match acc score', exact_score*100], 
    #         ['f1 score', f1_score*100], 
    #         ['precision', precision*100], 
    #         ['recall', recall*100], 
    #         ['bleu_score', bleu_score*100], 
    #         ['bleu_score_1', bleu_score_1*100], 
    #         ['bleu_score_2', bleu_score_2*100], 
    #         ['bleu_score_3', bleu_score_3*100], 
    #         ['open accuracy', open_hit_score*100],
    #         ['yes/no accuracy', closed_score*100]
    #     ], 
    #     headers=['Metric', 'Performance']
    # )

if __name__ == '__main__':
    args = parse_option()

    # gt = json.load(open(args.gt, 'r'))
    test_dict_lst = load_jsonl(args.pred)

    # gt_ids = [item['id'] for item in gt]
    # pred_ids = [item['question_id'] for item in pred]
    # # import pdb; pdb.set_trace()
    # assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(test_dict_lst, args)
    print(results)
