# based on package from https://github.com/taki0112/ResNet-Tensorflowi

#  implement 16-channel resnet in evenly-spaced blocks averaged in the z-direction
# 1) average 256 z blocks into 16
# 2)
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
	parser.add_argument('--dataset', type=str, default='ACV', help='[ACV] for our project')

	parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=40, help='Minibatch size')
	parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')

	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
	                    help='Directory name to save the checkpoints')
	parser.add_argument('--log_dir', type=str, default='logs',
	                    help='Directory name to save training logs')
	parser.add_argument('--data_folder', type=str, default="acv_image_data", help="Name of image data folder")
	parser.add_argument('--work_path', type=str, default="/Volumes/APPLE SSD/acv_project_team1_data", help="Working folder to start in")
	parser.add_argument('--data_type', type=str, default='affine', help="affine or projection (type of image data to train with")
	parser.add_argument('--use_lung_mask', action='store_true', help="Use masked input data, must batch export first with annotation GUI.")
	parser.add_argument('--n_axial_channels', type=int, default=16, help="Number of averaged evenly-spaced z-direction blocks. (16, 8, 4). Smaller value-better resolution in axial direction.")
	parser.add_argument('--train_test_ratio', type=str, required=True, help="Training/testing data split ('70_30' or '80_20')")
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

		print("%d-slice axial-averaged ResNet CNN..." % args.n_axial_channels)

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