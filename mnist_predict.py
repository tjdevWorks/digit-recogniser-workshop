import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from distutils.version import LooseVersion
import warnings

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

def predict(path_to_image = None):
    if path_to_image is None:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    else:
        print("Changes to mnist_predict.py need to be done yet")
        return

    sess = tf.Session()

    saver=tf.train.import_meta_graph('trained/test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./trained/'))

    graph = tf.get_default_graph()

    output = graph.get_tensor_by_name("output_fc_layer/outputs:0")

    inputs = graph.get_tensor_by_name("Inputs:0")
    targets = graph.get_tensor_by_name("Labels:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    learning_rate = graph.get_tensor_by_name("learning_rate:0")

    pred = tf.nn.softmax(output)

    if path_to_image is None:
        img_predict_index = np.random.randint(mnist.test.images.shape[0])

        predictions = sess.run(pred, feed_dict={inputs:mnist.test.images[img_predict_index].reshape(1,784),
                                        targets: mnist.test.labels[img_predict_index].reshape(1,10),
                                        keep_prob:1.0, learning_rate:0.0001})
        print("Ground Truth Value of image : ", np.argmax(mnist.test.labels[img_predict_index]))
        print("Predicted Value of image : ",np.argmax(predictions[0]))

    sess.close()

if __name__ == '__main__':
    predict()
