from jmt import JMT
import tensorflow as tf
import pprint

pp = pprint.PrettyPrinter(indent=4)


model = JMT(100, 1e-5, 20, 0.001)
model.load_data()
with tf.Graph().as_default() as graph:
    model.build_model()
    model.train_model(graph, 300, False)
    task_desc = {
        'pos': 'this has increased the risk',
        'chunk': 'this has increased the risk',
        'relatedness': ['two dogs are wrestling and hugging', 'there is no dog wrestling and hugging'],
        'entailment': ['two dogs are wrestling and hugging', 'there is no dog wrestling and hugging']
    }
    res = model.get_predictions(graph, task_desc)
    pp.pprint(res)
