import tensorflow as tf
import numpy as np

def predict(image):
    sess=tf.Session()

    labels = np.zeros(10)

    saver=tf.train.import_meta_graph('trained/test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./trained/'))

    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("output_fc_layer/outputs:0")

    inputs = graph.get_tensor_by_name("Inputs:0")
    targets = graph.get_tensor_by_name("Labels:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    learning_rate = graph.get_tensor_by_name("learning_rate:0")

    pred = tf.nn.softmax(output)

    predictions = sess.run(pred, feed_dict={inputs:image.reshape(1,784),
                                        targets: labels.reshape(1,10),
                                        keep_prob:1.0, learning_rate:0.0001})

    #print("Ground Truth Value of image : ", np.argmax(mnist.test.labels[img_predict_index]))
    #print("Predicted Value of image : ",np.argmax(predictions[0]))

    sess.close()

    return np.argmax(predictions[0]), predictions[0][np.argmax(predictions[0])] * 100
