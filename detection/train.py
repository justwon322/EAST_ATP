import argparse
import sys
import numpy as np

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from utils.dataset import custom_dataset
from utils.model import EAST
from utils.loss import Loss
from utils.util import set_seed
import os
import time
import glob
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

def train(args):

	set_seed(args.seed)
	file_num = len(glob.glob(os.path.join(args.train_img_path,"**","**","**","*.JPG")))
	dataset = custom_dataset(img_path=glob.glob(os.path.join(args.train_img_path,"**","**","**","*.JPG")),gt_path=glob.glob(os.path.join(args.train_gt_path,"**","**","**","*.json")))
	data_idxs = np.arange(file_num)
	train_, valid_ = train_test_split(data_idxs,test_size=0.2, random_state = args.seed)
	valid_, test_ = train_test_split(valid_, test_size=.5 , random_state = args.seed)
	trainset, validset, testset = data.Subset(dataset,train_), data.Subset(dataset,valid_) ,data.Subset(dataset,test_)

	train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
	valid_loader = data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
	test_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(w=args.w, d=args.d,pretrained=args.pretrained).to(device)

	if args.distributed & torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # 0.0003
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2], gamma=0.1)
	best_loss = np.inf

	for epoch in range(args.epochs):	
		model.train()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)+model.regularisation().squeeze()
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
              epoch+1, args.epochs, i+1, int(file_num/args.batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/args.batch_size), time.time()-epoch_time))
		args.writer.add_scalar('Train loss',epoch_loss/int(file_num/args.batch_size),epoch)
		print(time.asctime(time.localtime(time.time())))
		print('='*50)

		if (epoch + 1) % args.interval == 0:
			valid_epoch_loss = 0
			# with torch.no_grad():
				# for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			#
			'''
			model.eval()
			for i, (v_img, v_gt_score, v_gt_geo, v_ignored_map) in enumerate(valid_loader):
				v_img, gt_score, gt_geo, ignored_map = v_img.to(device), v_gt_score.to(device), v_gt_geo.to(
					device), v_ignored_map.to(device)
				v_pred_score, v_pred_geo = model(v_img)
				valid_loss = criterion(v_gt_score, v_pred_score, v_gt_geo, v_pred_geo, v_ignored_map) + model.regularisation().squeeze()
				epoch_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


			if valid_loss < best_loss:
				best_loss = valid_loss
				best_state_dict = model.module.state_dict() if args.distributed else model.state_dict()
				torch.save(best_state_dict, os.path.join(args.pths_path, 'model_bestepoch_{}.pth'.format(epoch+1)))
			args.writer.add_scalar('Validation loss',valid_loss,epoch)
			#(+) optional
			#model.train()
			'''
		scheduler.step()








def main():

	# parser
	parser = argparse.ArgumentParser(description="---#---")

	parser.add_argument("--batch_size", default=2, type=int)  # batch size가 성능에도 직접적으로 영향을 끼친다
	parser.add_argument("--lr", default=1e-3, type=float)
	parser.add_argument("--gpu_device", default=0, type=int)
	parser.add_argument('--seed', type=int, default=1) # seed 성능 재연을 위해서 필수적인 부분이고.
	parser.add_argument('--w', type=float, default=0.00009, help='Weight regularization')
	parser.add_argument('--d', type=float, default=1e-7, help='Dropout regularization')
	parser.add_argument('--epochs', type=int, default=600)
	parser.add_argument('--save_interval', type=int, default=5)
	parser.add_argument('--train_img_path', type=str, default="./dataset/jpg")
	parser.add_argument('--train_gt_path', type=str, default="./dataset/json")
	parser.add_argument("--pths_path",type=str, default="./utils/pths")
	parser.add_argument("--log_path",type=str, default="./log")
	parser.add_argument("--exp_name",type=str, default="temp",help="experiment name")
	parser.add_argument("--distributed",action='store_true')
	parser.add_argument("--pretrained",action='store_true')

	args = parser.parse_args()
	args.writer = SummaryWriter(os.path.join(args.log_path,args.exp_name))
	train(args)

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)