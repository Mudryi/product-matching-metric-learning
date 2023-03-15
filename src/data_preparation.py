import zipfile
import os
from tqdm.auto import tqdm
import pandas as pd


def prepare_product_evaluation_data():
    if not os.path.exists("datasets/product_matching_evaluation"):
        with zipfile.ZipFile("datasets/MCS2023_development_test_data.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets")
        os.rename("datasets/development_test_data", "datasets/product_matching_evaluation")
        print('final evaluation extracted')
    else:
        print('final evaluation already exist')


def prepare_test_10k_as_train():
    if not os.path.exists("datasets/all_data"):
        with zipfile.ZipFile("datasets/test.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets")
        os.rename("datasets/test", "datasets/all_data")
        print('Training data extracted')
    else:
        print('Training data already exist')


def prepare_train_eval_split_for_cls(train_labels, eval_labels):
    os.mkdir('dataset/train')
    os.mkdir('dataset/eval')

    for name, class_ in zip(train_labels["name"], train_labels["class"]):
        if not os.path.exists(f'dataset/train/{class_}'):
            os.mkdir(f'dataset/train/{class_}')
        os.rename(f'dataset/test/{name}', f'dataset/train/{class_}/{name}')

    for name, class_ in zip(eval_labels["name"], eval_labels["class"]):
        if not os.path.exists(f'dataset/eval/{class_}'):
            os.mkdir(f'dataset/eval/{class_}')
        os.rename(f'dataset/test/{name}', f'dataset/eval/{class_}/{name}')


def triplet_dataset_preparation(train_labels, eval_labels, rebuild=False):
    if (os.path.exists('datasets/train_triplets.csv') and os.path.exists('datasets/eval_triplets.csv')) and (
    not rebuild):
        return
    train_triplets = {"anchor": [], "pos": [], "neg": []}
    eval_triplets = {"anchor": [], "pos": [], "neg": []}

    for name, class_ in tqdm(zip(train_labels['name'], train_labels['class']), total=len(train_labels)):
        if sum(train_labels['class'] == class_) < 2:
            continue
        train_triplets['anchor'].append('datasets/all_data/' + name)
        train_triplets['pos'].append('datasets/all_data/' + train_labels[(train_labels['class'] == class_) &
                                                                        (train_labels['name'] != name)].sample()[
            'name'].values[0])
        train_triplets['neg'].append(
            'datasets/all_data/' + train_labels[(train_labels['class'] != class_)].sample()['name'].values[0])

    for name, class_ in tqdm(zip(eval_labels['name'], eval_labels['class']), total=len(eval_labels)):
        if sum(eval_labels['class'] == class_) < 2:
            continue
        eval_triplets['anchor'].append('datasets/all_data/' + name)
        eval_triplets['pos'].append('datasets/all_data/' + eval_labels[
            (eval_labels['class'] == class_) & (eval_labels['name'] != name)].sample()['name'].values[0])
        eval_triplets['neg'].append(
            'datasets/all_data/' + eval_labels[(eval_labels['class'] != class_)].sample()['name'].values[0])

    pd.DataFrame(train_triplets).to_csv('datasets/train_triplets.csv', index=False)
    pd.DataFrame(eval_triplets).to_csv('datasets/eval_triplets.csv', index=False)
