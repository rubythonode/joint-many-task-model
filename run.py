from jmt import JMT
import tensorflow as tf


model = JMT(200, 1e-5, 20, 0.001)
model.load_data()
with tf.Graph().as_default() as graph:
    model.build_model()
    model.train_model(graph)
