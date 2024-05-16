from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif
from PIL import Image

"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir',  default='./exp/checkpoint/experiment_0_batch_6',
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=9, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, default='./exp/data/val.obj', help='the source images for inference')
parser.add_argument('--embedding_ids', dest='embedding_ids', type=str, default='3', help='embeddings involved')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='./exp/result/', help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--interpolate', dest='interpolate', type=int, default=1,
                    help='interpolate between different embedding vectors')
parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
parser.add_argument('--uroboros', dest='uroboros', type=int, default=1,
                    help='Shōnen yo, you have stepped into uncharted territory')
args = parser.parse_args()


def main(_):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    with tf.compat.v1.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=True, inst_norm=args.inst_norm)
        embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
        #if not args.interpolate:
        model.infer(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
                    save_dir=args.save_dir)
        #if 1==1:# len(embedding_ids) == 1:
        #        embedding_ids = embedding_ids[0]
         #       model.infer(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
         #                   save_dir=args.save_dir)
        # else:
        #    if len(embedding_ids) < 2:
        #        raise Exception("no need to interpolate yourself unless you are a narcissist")
        #    chains = embedding_ids[:]
        #    if args.uroboros:
        #        chains.append(chains[0])
        #    pairs = list()
        #    for i in range(len(chains) - 1):
        #        pairs.append((chains[i], chains[i + 1]))
        #    for s, e in pairs:
        #        model.interpolate(model_dir=args.model_dir, source_obj=args.source_obj, between=[s, e],
        #                          save_dir=args.save_dir, steps=args.steps)
           # if args.output_gif:
           #     gif_path = os.path.join(args.save_dir, args.output_gif)
           #     compile_frames_to_gif(args.save_dir, gif_path)
           #     print("gif saved at %s" % gif_path)




if __name__ == '__main__':
    tf.compat.v1.app.run()

