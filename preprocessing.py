from src.data_preparation import prepare_product_evaluation_data, prepare_test_10k_as_train, triplet_dataset_preparation
import pandas as pd

prepare_product_evaluation_data()
prepare_test_10k_as_train()

labels = pd.read_csv("datasets/test_kaggletest.csv").sort_values("class")
print('Unique labels = ', labels["class"].nunique())
n_classes = labels["class"].nunique()

train_size = 0.9
labels = labels.sample(frac=1, random_state=703)

train_labels = labels[:int(len(labels)*train_size)]
eval_labels = labels[int(len(labels)*train_size):]
print(f'Train size = {len(train_labels)}\nTest size = {len(eval_labels)}')

# prepare_train_eval_split_for_cls(train_labels, eval_labels)
triplet_dataset_preparation(train_labels, eval_labels, rebuild=False)
