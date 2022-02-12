'''
Demo for CS 324 Winter 2022: Project 1.

This file demonstrates using ANLI data how to:

1. How to construct queries
2. How to specify decoding parameters
3. How to submit queries/decoding parameters to LLM API
4. How to generate a prediction from API response
5. How to evaluate the accuracy of predictions
'''

import getpass
import csv
import pprint
import random

from authentication import Authentication
from request import Request, RequestResult
from accounts import Account
from remote_service import RemoteService
from data import load_datasets

from transformers import DistilBertTokenizer, DistilBertModel
import torch


def get_query_from_example(premise: str, hypothesis: str) -> str:
    # prompt = 'How are the Premise and Hypothesis related?'
    prompt = 'If premise entails hypothesis, ans="+"; If premise contradicts hypothesis, ans="-"; If premise is neutral to hypothesis, ans="="; ans="'
    query = 'Premise: ' + premise + ' &&& ' + 'Hypothesis: ' + hypothesis + ' &&& ' + prompt
    return query


def get_query(premise: str, hypothesis: str) -> str:
    """Construct a query by specifying a prompt to accompany the input x."""
    num_examples = 5
    query_strings = []
    encoding_text = 'premise: ' + premise + '; hypothesis: ' + hypothesis
    encoded_text = tokenizer(encoding_text, return_tensors='pt', padding=True)
    text_embedding = model(**encoded_text).last_hidden_state.mean(dim=1)
    example_indices = torch.norm(text_embedding - training_dataset_embeddings,
                                 dim=-1).topk(num_examples,
                                              largest=False).indices
    # example_indices = random.sample(list(range(len(train_dataset))),
    #                                 k=num_examples)
    for example_index in example_indices:
        example_index = int(example_index)
        example_premise = train_dataset[example_index]['premise']
        example_hypothesis = train_dataset[example_index]['hypothesis']
        example_y = train_dataset[example_index]['label']
        if example_y == 0:
            example_y = '+'
        elif example_y == 1:
            example_y = '='
        elif example_y == 2:
            example_y = '-'
        else:
            raise NotImplementedError
        query_strings.append(
            get_query_from_example(example_premise, example_hypothesis) +
            example_y + '"')

    query_strings.append(get_query_from_example(premise, hypothesis))
    return '; '.join(query_strings)


def get_decoding_params():
    """
    Specify decoding parameters.
    See `request.py` for full list of decoding parameters.
    """
    decoding_params = {'top_k_per_token': 10, 'max_tokens': 1}
    return decoding_params


def get_request(query, decoding_params) -> Request:
    """
    Specify request given query and decoding parameters.
    See `request.py` for how to format request.
    """
    return Request(prompt=query, **decoding_params)


def make_prediction(request_result: RequestResult) -> int:
    """
    Map the API result to a class label (i.e. prediction)
    """
    # TODO: this is a stub, please improve!
    for completion in request_result.completions:
        # completion = request_result.completions[0].text.lower()
        completion = request_result.completions[0].text.lower()
        if '+' in completion:
            return 0
        if '=' in completion:
            return 1
        if '-' in completion:
            return 2

    print('None of the options was predicted')
    return 1
    # if completion == 'entailment':
    #     return 0
    # elif completion == 'neutral':
    #     return 1
    # elif completion == 'contradiction':
    #     return 2
    # else:
    #     return 1


# Writes results to CSV file.
def write_results_csv(predictions, run_name, header):
    file_name = 'results/{}.csv'.format(run_name)
    with open(file_name, 'w', encoding='utf8') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        for entry in predictions:
            row = [entry[column_name] for column_name in header]
            writer.writerow(row)


# An example of how to use the request API.
auth = Authentication(api_key=open('api_key.txt').read().strip())
service = RemoteService()

# Access account and show current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# As an example, we demonstrate how to use the codebase to work with the ANLI dataset
dataset_name = 'anli'
train_dataset, dev_dataset, test_dataset = load_datasets(dataset_name)
predictions = []

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

max_encoding_samples = 100
train_dataset_list = list(train_dataset)[:max_encoding_samples]
train_dataset_text = [
    'premise: ' + sample['premise'] + '; hypothesis: ' + sample['hypothesis'] +
    '; label: ' + str(sample['label']) + '; reason: ' + sample['reason']
    for sample in train_dataset_list
]

training_dataset_embeddings = []
for text in train_dataset_text:
    encoding = tokenizer(text, return_tensors='pt', padding=True)
    embedding = model(**encoding).last_hidden_state.mean(dim=1).view(-1)
    training_dataset_embeddings.append(embedding)

training_dataset_embeddings = torch.stack(training_dataset_embeddings)

max_examples = -1  # TODO: set this to -1 once you're ready to run it on the entire dataset

for i, row in enumerate(test_dataset):
    if i >= max_examples and max_examples != -1:
        break
    premise, hypothesis, y = row['premise'], row['hypothesis'], row['label']

    query: str = get_query(premise, hypothesis)

    decoding_params = get_decoding_params()

    request: Request = get_request(query, decoding_params)
    print('API Request: {}\n'.format(request))

    request_result: RequestResult = service.make_request(auth, request)
    print('API Result: {}\n'.format(request_result))

    yhat = make_prediction(request_result)

    if 'uid' in row:
        uid = row['uid']
    else:
        uid = i

    predictions.append({
        'uid': uid,
        'y': y,
        'yhat': yhat,
        'correct': y == yhat
    })

metric_name = 'accuracy'
accuracy = sum([pred['correct'] for pred in predictions]) / len(predictions)
print('{} on {}: {}'.format(metric_name, dataset_name, accuracy))

run_name = 'demo'
header = ['uid', 'y', 'yhat']
write_results_csv(predictions, run_name, header)
