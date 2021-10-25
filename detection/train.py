import argparse
import glob
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from earlystop import EarlyStopping
from utils.dataset import custom_dataset
from utils.loss import Loss
from utils.model import EAST
from utils.util import *
from utils import epoch_test as et


def train(args):
    set_seed(args.seed)
    file_num = len(glob.glob(os.path.join(args.train_img_path, "**", "**", "**", "*.JPG")))
    dataset = custom_dataset(img_path=glob.glob(os.path.join(args.train_img_path, "**", "**", "**", "*.JPG")),
                             gt_path=glob.glob(os.path.join(args.train_gt_path, "**", "**", "**", "*.json")))
    train_, valid_ = train_test_split(np.arange(file_num), test_size=0.2, random_state=args.seed)
    valid_, test_ = train_test_split(valid_, test_size=.5, random_state=args.seed)
    trainset, validset, testset = data.Subset(dataset, train_), data.Subset(dataset, valid_), data.Subset(dataset,
                                                                                                          test_)

    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    criterion = Loss()

    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    model = EAST(w=args.w, d=args.d)
    if torch.cuda.device_count() > 1:  # multi gpu train
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.run_name}')

    if args.pretrained:
        model.load_state_dict(torch.load(f'./parameter/{args.exp_name}.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.2)

    with get_rich_pbar(transient=True, auto_refresh=True) as pg:

        task = pg.add_task(f"[bold red] Training... ", total=(args.epochs * len(train_loader)))

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            training_classifyloss = 0.0
            training_geoloss = 0.0
            training_iouloss = 0.0
            for _, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
                img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(
                    device), ignored_map.to(device)
                pred_score, pred_geo = model(img)
                classify_loss, geo_loss, iou_loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map,
                                                              args.logger)
                if classify_loss != 0.0:
                    pg.print('classify loss is {:.8f}, geo loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss,
                                                                                                      geo_loss,
                                                                                                      iou_loss))
                trainingloss = classify_loss + geo_loss + model.regularisation().squeeze()
                training_classifyloss += classify_loss
                training_geoloss += geo_loss
                training_iouloss += iou_loss
                epoch_loss += trainingloss.item()
                optimizer.zero_grad()
                trainingloss.backward()
                optimizer.step()
                pg.update(task, advance=1)

            if args.tensorboard:
                args.writer.add_scalar('Training_loss', epoch_loss / len(train_loader), epoch)
                args.writer.add_scalar('Training_classifyloss', training_classifyloss / len(train_loader), epoch)
                args.writer.add_scalar('Training_geoloss', training_geoloss / len(train_loader), epoch)
                args.writer.add_scalar('Training_iouloss', training_iouloss / len(train_loader), epoch)
            pg.print('Training loss is {:.8f}'.format(epoch_loss / len(train_loader)))
            args.logger.info('Training loss is {:.8f}'.format(epoch_loss / len(train_loader)))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                valid_classifyloss = 0
                valid_geoloss = 0
                valid_iouloss = 0

                for _, (v_img, v_gt_score, v_gt_geo, v_ignored_map) in enumerate(valid_loader):
                    v_img, v_gt_score, v_gt_geo, v_ignored_map = v_img.to(device), v_gt_score.to(device), v_gt_geo.to(
                        device), v_ignored_map.to(device)
                    v_pred_score, v_pred_geo = model(v_img)
                    v_classifyloss, v_geoloss, v_iouloss = criterion(v_gt_score, v_pred_score, v_gt_geo, v_pred_geo,
                                                                     v_ignored_map, args.logger)
                    valid_classifyloss += v_classifyloss.item()
                    valid_geoloss += v_geoloss.item()
                    valid_iouloss += v_iouloss.item()
                    pg.print('classify loss is {:.8f}, geo loss is {:.8f}, iou loss is {:.8f}'.format(v_classifyloss,
                                                                                                      v_geoloss,
                                                                                                      v_iouloss))


                avg_valid_loss = (valid_classifyloss + valid_geoloss + valid_iouloss)
                if args.tensorboard:
                    args.writer.add_scalar('Validation_loss', avg_valid_loss / len(valid_loader), epoch)
                    args.writer.add_scalar('Validation_classifyloss', valid_classifyloss / len(valid_loader), epoch)
                    args.writer.add_scalar('Validation_geoloss', valid_geoloss / len(valid_loader), epoch)
                    args.writer.add_scalar('Validation_iouloss', valid_iouloss / len(valid_loader), epoch)
                pg.print('Validation loss is {:.8f}'.format(avg_valid_loss / len(valid_loader)))
                args.logger.info('Validation loss is {:.8f}'.format(avg_valid_loss / len(valid_loader)))
                model.train()
            #################################################
            ###### 1epoch 마다 test image 기록 및 모델 저장#####
            #################################################
            img = et.main(model, device)
            tf = transforms.ToTensor()
            img_t = tf(img)
            img_t = img_t.permute(1, 2, 0)

            fig = plt.figure(figsize=(20,20))
            plt.imshow(img_t)


            if args.tensorboard:
                args.writer.add_figure(f'{epoch}th epoch Test result image', fig, epoch)
            plt.close('all')

            # if epoch % 20 == 0 and args.start_epochs == 0 and epoch != 0: #epoch 20마다 모델 저장
            path = f'./parameter/{args.run_name}'
            if not os.path.exists(path):  # epoch 1마다 모델 저장
                os.makedirs(path)
            torch.save(model.state_dict(), f'{path}/model_epoch_{epoch}.pth')

        # early_stopping(valid_geoloss, model) 임시 주석처리
        # if early_stopping.early_stop:
        # pg.print("Early stopped!!")
        # args.logger.info("Early stopped!!")

        model.eval()

        test_classifyloss = 0
        test_geoloss = 0
        test_iouloss = 0

        with torch.no_grad():
            for batch_idx, (t_img, t_gt_score, t_gt_geo, t_ignored_map) in enumerate(test_loader):

                t_img, t_gt_score, t_gt_geo, t_ignored_map = t_img.to(device), t_gt_score.to(device), t_gt_geo.to(
                    device), t_ignored_map.to(device)
                t_pred_score, t_pred_geo = model(t_img)
                t_classifyloss, t_geoloss, test_iouloss = criterion(t_gt_score, t_pred_score, t_gt_geo, t_pred_geo,
                                                                    t_ignored_map, args.logger)
                test_classifyloss += t_classifyloss.item()
                test_geoloss += t_geoloss.item()
                test_iouloss += t_geoloss.item()

                prob_total_u_mu = torch.zeros((30, t_gt_score.size(0), t_gt_score.size(2), t_gt_score.size(3)),
                                              device=t_gt_score.device)
                for t in range(30):
                    prob_total_u_mu[t] = model(t_img)[0].squeeze(1)

                pred_prob = torch.mean(prob_total_u_mu, axis=0)
                uncertainty = torch.var(prob_total_u_mu, axis=0)

                if batch_idx == 0:
                    uncertaintys = pd.DataFrame({'uncertainty': uncertainty.sum(axis=[-1, -2]).cpu().numpy()})
                    gt_probs = t_gt_score.squeeze(1).cpu().numpy()
                    pred_probs = pred_prob.detach().cpu().numpy()
                else:
                    uncertaintys = pd.concat(
                        (uncertaintys, pd.DataFrame({'uncertainty': uncertainty.sum(axis=[-1, -2]).cpu().numpy()})))
                    gt_probs = np.concatenate([gt_probs, t_gt_score.squeeze(1).cpu().numpy()], axis=0)
                    pred_probs = np.concatenate([pred_probs, pred_prob.detach().cpu().numpy()], axis=0)

        avg_test_loss = (test_classifyloss + test_geoloss + test_iouloss)
        if args.tensorboard:
            args.writer.add_scalar('Testing_loss', avg_test_loss / len(test_loader), epoch)
            args.writer.add_scalar('Testing_classify_loss', test_classifyloss / len(test_loader), epoch)
            args.writer.add_scalar('Testing_geo_loss', test_geoloss / len(test_loader), epoch)
            args.writer.add_scalar('Testing_iou_loss', test_iouloss / len(test_loader), epoch)
        pg.print('Testing loss is {:.8f}'.format(avg_test_loss / len(test_loader)))

        if not os.path.exists(f"./output/{args.run_name}.pkl"):
            os.makedirs(f"./output/{args.run_name}.pkl")
        with open(f"./output/{args.run_name}.pkl", "wb") as f:
            pickle.dump({"uncertainty": uncertaintys, "gt": gt_probs, "pred_probs": pred_probs}, f)
    # break


def main():
    # parser
    parser = argparse.ArgumentParser(description="---#---")

    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.000009, help='Weight regularization')
    parser.add_argument('--d', type=float, default=1e-10, help='Dropout regularization')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--train_img_path', type=str, default="./dataset/jpg")
    parser.add_argument('--train_gt_path', type=str, default="./dataset/json")
    parser.add_argument("--pths_path", type=str, default="./utils/pths")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--exp_name", type=str, default="EAST_baseline", help="experiment name")
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument('--tensorboard', type=str, default='None', help='tensorboard run and project name')

    args = parser.parse_args()

    # log name
    log_name = '+'.join([f'{v}' for _, v in vars(args).items()])
    if args.tensorboard:
        args.writer = SummaryWriter(f"./tensorboard/{str(args.tensorboard)}/{log_name}")
        [args.writer.add_text(f"{k}", str(v), 1) for k, v in vars(args).items()]
    args.run_name = log_name
    logfile = os.path.join("./log", f"{args.run_name}.log")
    args.logger = get_rich_logger(logfile)

    # Learning start
    start = time.time()
    args.logger.info(f"Start time : {start}")
    train(args)

    # Finish
    if args.tensorboard:
        args.writer.close()
    elapsed_sec = time.time() - start
    elapsed_mins = elapsed_sec / 60
    elapsed_hours = elapsed_mins / 60
    args.logger.info(f"Total training time: {elapsed_mins:,.2f} minutes ({elapsed_hours:,.2f} hours).")
    args.logger.info(f"Finish time : {time.time()}")
    args.logger.handlers.clear()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
