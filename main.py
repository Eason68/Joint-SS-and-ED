from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import MyModel
from loss import Loss
from dataLoader import S3DISDataset
import torch
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from metrics import Metrics
import argparse
from utils import LogColor


def train(args):

    print("Creating network at GPU " + str(args.gpu_id), end="...")
    device = torch.device("cuda:{}".format(args.gpu_id))

    net = MyModel(points=args.num_points, in_channel=args.in_channel)
    net.to(device)
    net = torch.nn.DataParallel(net)
    if args.pretrain:
        net.load_state_dict(torch.load(os.path.join(args.save_dir, "pretrain.pth")))

    criterion = Loss()
    criterion.to(device)
    print("Done!")

    print("Loading dataset", end="...")
    train = S3DISDataset(split="train", test_area=args.test_area, num_points=args.num_points, transform=args.transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)

    test = S3DISDataset(split="test", test_area=args.test_area, num_points=args.num_points, transform=args.transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    print("Done!")

    print("Creating optimizer and scheduler", end="...")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    print("Done!")

    print("Creating results folder", end="...")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_dir, "Train_area{}_{}".format(args.test_area, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("Done at", root_folder)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    maxIOU = 0.0
    maxBIoU = 0.0

    # iterate over epochs
    for epoch in range(args.epochs):

        # training
        net.train()
        cm = np.zeros((args.num_classes, args.num_classes))
        train_biou = 0
        train_loss = 0

        t = tqdm(enumerate(train_loader), ncols=100, desc="Train {}".format(epoch))
        for i, (points, labels) in t:

            points = points.permute(0, 2, 1)
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output, coords, feats, preds, indexs = net(points)

            loss = criterion(labels, indexs, output, preds, coords, feats)
            loss.backward()
            optimizer.step()

            output_np = np.argmax(output.cpu().detach().numpy(), axis=2).copy()
            target_np = labels.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(args.num_classes)))
            cm += cm_

            oa = f"{Metrics.stats_overall_accuracy(cm):.3f}"
            # aa = f"{Metrics.stats_accuracy_per_class(cm)[0]:.3f}"
            iou = f"{Metrics.stats_iou_per_class(cm)[0]:.3f}"

            train_biou += Metrics.stats_boundary_iou(points, labels, output)
            train_loss += loss.detach().cpu().item()

            t.set_postfix(IOU=LogColor.wblue(iou), BIoU=LogColor.wblue(f"{train_biou / (i + 1):.3f}"),
                          LOSS=LogColor.wblue(f"{train_loss / cm.sum():.3e}"))

        if optimizer.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.9e-5

        # validation
        net.eval()
        cm_test = np.zeros((args.num_classes, args.num_classes))
        test_biou = 0
        test_loss = 0

        t = tqdm(enumerate(test_loader), ncols=100, desc="Test {}".format(epoch))
        with torch.no_grad():
            for i, (points, labels) in t:

                points = points.permute(0, 2, 1)
                points = points.to(device)
                labels = labels.to(device)

                output, coords, feats, preds, indexs = net(points)
                loss = criterion(labels, indexs, output, preds, coords, feats)

                output_np = np.argmax(output.cpu().detach().numpy(), axis=2).copy()
                target_np = labels.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(args.num_classes)))
                cm_test += cm_

                oa_val = f"{Metrics.stats_overall_accuracy(cm_test):.3f}"
                # aa_val = f"{Metrics.stats_accuracy_per_class(cm_test)[0]:.3f}"
                iou_val = f"{Metrics.stats_iou_per_class(cm_test)[0]:.3f}"

                test_biou += Metrics.stats_boundary_iou(points, labels, output)
                test_loss += loss.detach().cpu().item()

                t.set_postfix(IOU=LogColor.wgreen(iou_val), BIoU=LogColor.wgreen(f"{test_biou / (i + 1):.3f}"),
                              LOSS=LogColor.wgreen(f"{test_loss / cm_test.sum():.3e}"))

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))
        if maxIOU < float(iou_val):
            torch.save(net.state_dict(), os.path.join(root_folder, "IoU_" + str(iou_val) + ".pth"))
            maxIOU = float(iou_val)
        if maxBIoU < float(test_biou):
            torch.save(net.state_dict(), os.path.join(root_folder, "B-IoU_" + str(test_biou) + ".pth"))
            maxBIoU = float(maxBIoU)

        # write the logs
        logs.write(f"{epoch} OA: {oa} IoU: {iou} B-IoU: {train_biou} OA: {oa_val} IoU: {iou_val} B-IoU: {test_biou}\n")
        logs.flush()

    logs.close()


def test(args):

    print("Creating network at " + str(args.gpu_id),end="...")
    device = torch.device("cuda:{}".format(args.gpu_id))

    net = MyModel()
    net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(args.save_dir, "state_dict.pth")))

    criterion = Loss()
    criterion.to(device)
    print("Done!")

    print("Loading dataset", end="...")
    test = S3DISDataset(split="test", test_area=args.test_area, num_points=args.num_points, transform=args.transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    print("Done!")

    print("Creating results folder", end="...")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_dir, "Test_area{}_{}".format(args.test_area, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("Done at", root_folder)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")

    net.eval()
    cm_test = np.zeros((args.num_classes, args.num_classes))
    test_biou = 0
    test_loss = 0

    t = tqdm(enumerate(test_loader), ncols=100, desc="Test")
    with torch.no_grad():
        for i, (points, labels) in t:
            points = points.permute(0, 2, 1)
            points = points.to(device)
            labels = labels.to(device)

            output, coords, feats, preds, indexs = net(points)
            loss = criterion(labels, indexs, output, preds, coords, feats)

            output_np = np.argmax(output.cpu().detach().numpy(), axis=2).copy()
            target_np = labels.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(args.num_classes)))
            cm_test += cm_

            oa_val = f"{Metrics.stats_overall_accuracy(cm_test):.3f}"
            # aa_val = f"{Metrics.stats_accuracy_per_class(cm_test)[0]:.3f}"
            iou_val = f"{Metrics.stats_iou_per_class(cm_test)[0]:.3f}"

            test_biou += Metrics.stats_boundary_iou(points, labels, output)
            test_loss += loss.detach().cpu().item()

            t.set_postfix(IOU=iou_val, BIoU=f"{test_biou / (i + 1):.3f}", LOSS=f"{test_loss / cm_test.sum():.3e}")

    # write the logs
    logs.write(f"{epoch} OA: {oa} IoU: {iou} B-IoU: {train_biou} OA: {oa_val} IoU: {iou_val} B-IoU: {test_biou}\n")
    logs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",    default="results/",   type=str)
    parser.add_argument("--data_path",   default="data",       type=str)
    parser.add_argument("--batch_size",  default=2,            type=int)
    parser.add_argument("--num_points",  default=2048,         type=int)
    parser.add_argument("--test_area",   default=5,            type=int)
    parser.add_argument("--threads",     default=1,            type=int)
    parser.add_argument("--pretrain",    default=False,        type=bool)
    parser.add_argument("--optimizer",   default="Adam",       type=str)
    parser.add_argument("--lr",          default=0.003,        type=float)
    parser.add_argument("--epochs",      default=350,          type=int)
    parser.add_argument("--model",       default="MyModel",    type=str)
    parser.add_argument("--num_classes", default=13,           type=int)
    parser.add_argument("--transform",   default=False,        type=bool)
    parser.add_argument("--gpu_id",      default=0,            type=int)
    parser.add_argument("--in_channel",  default=6,            type=int)
    parser.add_argument("--eval",        default=True,         type=bool)
    parser.add_argument("--test",        default=False,        type=bool)
    args = parser.parse_args()

    train(args) if not args.test else test(args)


if __name__ == '__main__':
    main()
