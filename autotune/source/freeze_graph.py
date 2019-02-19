import tools
import os
import yaml
from os.path import join
import tensorflow as tf

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['base_conf']['gpu_num'])

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys
import facenet
from six.moves import xrange


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))


            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')

        # Serialize and dump the output graph to the filesystem
        if args.output_file:
            output_path = join(args.model_dir, args.output_file)
        else:
            output_path = join(args.model_dir, args.model_dir.split('/')[-1] + '.pb')

        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_path))


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or
                node.name.startswith('phase_train') or node.name.startswith('Bottleneck') or node.name.startswith(
                    'Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/home/chris/Dropbox/video_scan/codebase/python/models/facenet/20180403-203839_center_loss_factor_0.01')
    parser.add_argument('--output_file', type=str,
                        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))