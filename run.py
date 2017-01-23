from jmt import JMT
import tensorflow as tf
import pprint

pp = pprint.PrettyPrinter(indent=4)


model = JMT(100, 1e-5, 0.001)
model.load_data()
with tf.Graph().as_default() as graph:
    model.build_model()
    train_desc = {
        'batch_size': 20,
        'entailment': 200,
        'pos': 500,
        'chunk': 500,
        'relatedness': 200
    }
    model.train_model(graph, train_desc, True)

    task_desc = {
        'pos': 'this has increased the risk',
        'chunk': 'this has increased the risk',
        'relatedness': ['two dogs are wrestling and hugging', 'there is no dog wrestling and hugging'],
        'entailment': ['Two dogs are wrestling and hugging', 'There is no dog wrestling and hugging']
    }
    res = model.get_predictions(graph, task_desc)
    pp.pprint(res)
