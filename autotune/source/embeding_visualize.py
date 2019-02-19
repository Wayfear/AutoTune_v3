import os
import yaml
from os.path import join
import tensorflow as tf
from ggplot import *

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['base_conf']['gpu_num'])


import tensorflow as tf
from libs import facenet
import os
import cv2
import numpy as np
from sklearn.manifold import TSNE
from ggplot import *
import pandas as pd
from os.path import join

middle_data_path = '../../../data/test/2017/self'
peoples = [o for o in os.listdir(middle_data_path)
                    if os.path.isdir(os.path.join(middle_data_path, o))]
pics = {}
for p in peoples:
    pics[p] = list(map(lambda x: join(middle_data_path, p, x), os.listdir(join(middle_data_path, p))))

model_names = [f for f in os.listdir(join(project_dir, cfg['base_conf']['model_path'])) if f.endswith('.pb')]
model_names.sort()
# model_names = ['facenet_v1.pb', '20170511-185253.pb', '20180403-165717_center_loss_factor_0.10.pb',
#                '20180403-165908_center_loss_factor_0.001.pb', '20180406-171209_center_loss_factor_0.10.pb']

# model_names = ['20180402-114759.pb']
result_dict = {}
std_dict = {}

for model_name in model_names:
    # -----------------------------
    # step1: load features
    # -----------------------------
    vectors = []
    labels = []
    cluster_len = []
    result_dict[model_name] = {}
    std_dict[model_name] = {}
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
        with tf.Session() as sess:
            facenet.load_model(join(project_dir, cfg['base_conf']['model_path'],
                                    model_name))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    p_idx = 0
    emd_dim = int(embeddings.get_shape()[1])
    print('embeddings of this model has dim {}'.format(emd_dim))
    for p in pics:
        image_list = []

        data_result_dict = np.empty([0, emd_dim])

        for pic_path in pics[p]:
            im = cv2.imread(pic_path)
            # print(pic_path)
            prewhitened = facenet.prewhiten(im)
            image_list.append(prewhitened)
            if len(image_list) == 1000:
                images = np.stack(image_list)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                data_result_dict = np.vstack((data_result_dict, emb))
                print(emb.shape)
                image_list.clear()
        if len(image_list) != 0:
            images = np.stack(image_list)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            data_result_dict = np.vstack((data_result_dict, emb))
        print(data_result_dict.shape)
        # np.save(join(os.path.dirname(pic_path), 'people%s.npy' % p), result_dict)
        mean = np.mean(data_result_dict, axis=0)
        std = np.std(data_result_dict, axis=0)
        result_dict[model_name][p] = mean
        std_dict[model_name][p] = np.average(std)
        print("%s: "% p)
        print("mean: ", np.linalg.norm(mean))
        print("std: ", np.average(std))
        print("\n")
        vectors += data_result_dict.tolist()
        labels += [p_idx]*data_result_dict.shape[0]
        p_idx += 1
        cluster_len.append(data_result_dict.shape[0])

    # -----------------------------
    # step2: visualize features
    # -----------------------------
    features = np.array(vectors)
    print('...to plot...')
    fig_dir = join('../../fig', str(cfg['visualize_features']['perplexity']))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    feat_cols = ['fea' + str(i) for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df['User'] = np.array(labels)

    # select certain users
    # sel_user = cfg['visualize_features']['sel_user']
    num_classes = len(peoples)
    sel_user = range(1, num_classes, int(np.ceil(num_classes * 0.1)))
    # sel_user = [2] + list(sel_user)
    df = df.loc[df['User'].isin(sel_user)]

    # print label as str
    df['User'] = df['User'].apply(lambda i: str(i))

    perplexity = cfg['visualize_features']['perplexity']
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, random_state=123)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['x-tsne'] = tsne_results[:, 0]
    df['y-tsne'] = tsne_results[:, 1]

    chart = ggplot(df, aes(x='x-tsne', y='y-tsne', color='User')) \
            + geom_point(size=70, alpha=0.8) \
            + labs(x="", y="")

    t = theme_gray()
    t._rcParams['font.size'] = 20  # Legend font size
    t._rcParams['xtick.labelsize'] = 15  # xaxis tick label size
    t._rcParams['ytick.labelsize'] = 15  # yaxis tick label size
    chart += t

    name = model_name.split('.')[0]
    chart.save(fig_dir + '/' + name + '_tnse.png')
