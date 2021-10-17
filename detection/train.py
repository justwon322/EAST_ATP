import argparse
import glob
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from earlystop import EarlyStopping
from utils.dataset import custom_dataset
from utils.loss import Loss
from utils.model import EAST
from utils.util import *
import matplotlib.pyplot as plt


def train(args):

	set_seed(args.seed)
	file_num = len(glob.glob(os.path.join(args.train_img_path,"**","**","**","*.JPG")))
	dataset = custom_dataset(img_path=glob.glob(os.path.join(args.train_img_path,"**","**","**","*.JPG")),gt_path=glob.glob(os.path.join(args.train_gt_path,"**","**","**","*.json")))
	train_, valid_ = train_test_split(np.arange(file_num),test_size=0.2, random_state = args.seed)
	valid_, test_ = train_test_split(valid_, test_size=.5 , random_state = args.seed)
	trainset, validset, testset = data.Subset(dataset,train_), data.Subset(dataset,valid_) ,data.Subset(dataset,test_)


	train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
	valid_loader = data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
	test_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

	criterion = Loss()
	device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
	model = EAST(w=args.w, d=args.d).to(device)
	early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.exp_name}.pth')

	if args.pretrained:
		model.load_state_dict(torch.load(f'./parameter/{args.exp_name}.pth'))

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2], gamma=0.2)

	with get_rich_pbar(transient=True, auto_refresh=True) as pg:

		task = pg.add_task(f"[bold red] Training... ", total=(args.epochs*len(train_loader)))
		epoch_list = list(range(args.start_epochs, args.epochs))
		x = pd.Series(epoch_list)
		y_1 = []
		y_2 = []
		for epoch in range(args.start_epochs,args.epochs):

			print('epoch : '+str(epoch))
			model.train()
			epoch_loss = 0
			for _, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
				img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
				pred_score, pred_geo = model(img)
				loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)+model.regularisation().squeeze()
				epoch_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				pg.update(task,advance=1)

			y_1.append(epoch_loss/len(train_loader))
			if args.tensorboard:
				args.writer.add_scalar('Train_loss',epoch_loss/len(train_loader),epoch)
			pg.print('Training loss is {:.8f}'.format(epoch_loss/len(train_loader)))
			scheduler.step()

			model.eval()
			v_temp_loss = 0
			with torch.no_grad():
				valid_epoch_loss = 0

				for _, (v_img, v_gt_score, v_gt_geo, v_ignored_map) in enumerate(valid_loader):
					v_img, v_gt_score, v_gt_geo, v_ignored_map = v_img.to(device), v_gt_score.to(device), v_gt_geo.to(device), v_ignored_map.to(device)
					v_pred_score, v_pred_geo = model(v_img)
					valid_loss = criterion(v_gt_score, v_pred_score, v_gt_geo, v_pred_geo, v_ignored_map).item()
					valid_epoch_loss += valid_loss

				avg_valid_loss = valid_epoch_loss / len(valid_loader)
				y_2.append(avg_valid_loss)
				if args.tensorboard:
					args.writer.add_scalar('Validation_loss',avg_valid_loss,epoch)
				pg.print('Validation loss is {:.8f}'.format(avg_valid_loss))

				model.train()

				early_stopping(avg_valid_loss, model)
				if early_stopping.early_stop:
					pg.print("Early stopped!!")

					model.eval()
					test_epoch_loss = 0

					with torch.no_grad():
						for batch_idx, (t_img, t_gt_score, t_gt_geo, t_ignored_map) in enumerate(test_loader):

							t_img, t_gt_score, t_gt_geo, t_ignored_map = t_img.to(device), t_gt_score.to(device), t_gt_geo.to(device), t_ignored_map.to(device)
							t_pred_score, t_pred_geo = model(t_img)
							test_epoch_loss += criterion(t_gt_score, t_pred_score, t_gt_geo, t_pred_geo,t_ignored_map).item()

							prob_total_u_mu = torch.zeros((30, t_gt_score.size(0), t_gt_score.size(2),t_gt_score.size(3)), device=t_gt_score.device)
							for t in range(30):
								prob_total_u_mu[t] = model(t_img)[0].squeeze(1)

							pred_prob = torch.mean(prob_total_u_mu, axis=0)
							uncertainty = torch.var(prob_total_u_mu,axis=0)

							if batch_idx == 0:
								uncertaintys = pd.DataFrame({'uncertainty': uncertainty.sum(axis=[-1,-2]).cpu().numpy()})
								gt_probs = t_gt_score.squeeze(1).cpu().numpy()
								pred_probs = pred_prob.detach().cpu().numpy()
							else:
								uncertaintys = pd.concat((uncertaintys,pd.DataFrame({'uncertainty': uncertainty.sum(axis=[-1,-2]).cpu().numpy()})))
								gt_probs = np.concatenate([gt_probs,t_gt_score.squeeze(1).cpu().numpy()],axis=0)
								pred_probs = np.concatenate([pred_probs,pred_prob.detach().cpu().numpy()],axis=0)

					avg_test_loss = test_epoch_loss / len(test_loader)
					if args.tensorboard:
						args.writer.add_scalar('Testing_loss',avg_test_loss,epoch)
					pg.print('Testing loss is {:.8f}'.format(avg_test_loss))

					with open(f"./output/{args.run_name}.pkl","rb") as f:
						pickle.dump({"uncertainty":uncertaintys , "gt":gt_probs, "pred_probs":pred_probs}, f)
					break

		plt.plot(x, pd.Series(y_1), x, pd.Series(y_2))
		plt.legend(['training_loss', 'validation_loss'])
		plt.show()

def main():

	# parser
	parser = argparse.ArgumentParser(description="---#---")

	parser.add_argument("--batch_size", default=4, type=int)
	parser.add_argument("--lr", default=1e-4, type=float)
	parser.add_argument("--gpu_device", default=0, type=int)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--w', type=float, default=0.00009, help='Weight regularization')
	parser.add_argument('--d', type=float, default=1e-7, help='Dropout regularization')
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--start_epochs', type=int, default=0)
	parser.add_argument('--train_img_path', type=str, default="./dataset/jpg")
	parser.add_argument('--train_gt_path', type=str, default="./dataset/json")
	parser.add_argument("--log_path",type=str, default="./log")
	parser.add_argument("--exp_name",type=str, default="EAST_baseline",help="experiment name")
	parser.add_argument("--pretrained",action='store_true')
	parser.add_argument("--threshold", default=0.95, type=float)
	parser.add_argument('--tensorboard', type=str, default=True, help='tensorboard run and project name')

	args = parser.parse_args()

	#log name
	log_name = '+'+args.log_path+'/'+args.exp_name
	if args.tensorboard:
		args.writer = SummaryWriter(f"./tensorboard/{str(args.tensorboard)}/{log_name}")
		[args.writer.add_text(f"{k}",str(v),1) for k,v in vars(args).items()]
	args.run_name = log_name
	logfile = os.path.join("./log",f"{args.run_name}.log")
	args.logger = get_rich_logger(logfile)

	train(args)

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
