import tensorflow as tf
import facenet
from tensorflow.python.platform import gfile
with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model("20170511-185253.pb")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        persisted_result = sess.graph.get_tensor_by_name("InceptionResnetV1/Bottleneck/weights:0")
        # print(persisted_result.eval())
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model.ckpt")




