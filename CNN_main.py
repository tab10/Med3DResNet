# based on package from https://github.com/taki0112/ResNet-Tensorflowi

# we want to implement 16 resnets in evenly-spaced blocks averaged in the z-direction
# 1) average 256 z blocks into 16
# 2) each block with [256,256,n_data] trained on a separate ResNet
# 3)
# Tim Burt 12/8/19

from CNN_ResNet import ResNet
import argparse
from CNN_utils import *


"""parsing and configuration"""
def parse_args():
	desc = "Tensorflow implementation of ResNet"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--phase', type=str, default='train', help='train or test ?')
	parser.add_argument('--z_slice', type=int, required=True, help='Which Z-slice averaged ResNet are we train/testing this time? [1, n_slice_blocks]')
	parser.add_argument('--dataset', type=str, default='ACV', help='[ACV] for our project')

	parser.add_argument('--epoch', type=int, default=82, help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=256, help='The size of batch per gpu')
	parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')

	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
	                    help='Directory name to save the checkpoints')
	parser.add_argument('--log_dir', type=str, default='logs',
	                    help='Directory name to save training logs')
	parser.add_argument('--data_folder', type=str, default="acv_image_data", help="Name of image data folder")
	parser.add_argument('--work_path', type=str, default="/Volumes/APPLE SSD", help="Working folder to start in")
	parser.add_argument('--data_type', type=str, default='affine', help="affine or projection (type of image data to train with")
	parser.add_argument('--lung_mask', type=bool, default=True, help="Use masked input data, must batch export first with annotation GUI.")
	parser.add_argument('--n_slice_blocks', type=int, default=16, help="This many ResNets will be trained on averaged evenly-spaced z-direction blocks")
	return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --result_dir
	check_folder(args.log_dir)

	# --epoch
	try:
		assert args.epoch >= 1
	except:
		print('number of epochs must be larger than or equal to one')

	# --batch_size
	try:
		assert args.batch_size >= 1
	except:
		print('batch size must be larger than or equal to one')
	return args


"""main"""
def main():
	# parse arguments
	args = parse_args()
	if args is None:
		exit()

	# open session
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

		os.chdir(args.work_path)

		cnn = ResNet(sess, args)

		# build graph
		cnn.build_model()

		# show network architecture
		show_all_variables()

		print("Z-slice %d of %d-ResNet CNN..." % (args.z_slice, args.n_slice_blocks))

		if args.phase == 'train' :
			# launch the graph in a session
			cnn.train()

			print(" [*] Training finished! \n")

			cnn.test()
			print(" [*] Test finished!")

		if args.phase == 'test' :
			cnn.test()
			print(" [*] Test finished!")

if __name__ == '__main__':
	main()