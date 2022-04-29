from model import PointMLP
from loss import Loss
from data import S3DISDataset
import torch
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from metrics import Metrics
import argparse


def train(args):

    # create the network
    print("Creating network at GPU " + str(args.gpu_id))
    net = PointMLP(points=args.num_points)
    net.cuda(args.gpu_id)
    net = torch.nn.DataParallel(net)
    if args.pretrain:
        net.load_state_dict(torch.load(os.path.join(args.save_dir, "pretrain.pth")))

    criterion = Loss()
    criterion.cuda(args.gpu_id)

    print("Loading dataset...")
    train = S3DISDataset(split="train", data_path=args.data_path, test_area=args.test_area, num_points=args.num_points, transform=args.transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)

    test = S3DISDataset(split="test", data_path=args.data_path, test_area=args.test_area, num_points=args.num_points, transform=args.transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    print("Creating optimizer and scheduler...")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    print("Creating results folder")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_dir, "{}_area{}_{}_{}".format(args.model, args.test_area, args.num_points, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("done at", root_folder)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    maxIOU = 0.0

    # iterate over epochs
    for epoch in range(args.epochs):

        # training
        net.train()

        lr = optimizer.param_groups[0]['lr']
        print('LearningRate:', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = 0
        cm = np.zeros((args.num_classes, args.num_classes))
        t = tqdm(train_loader, ncols=100, desc="Train {}".format(epoch))
        for points, labels in t:

            points = points.permute(0, 2, 1)
            points = points.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)

            optimizer.zero_grad()

            output, coords, feats, preds, indexs = net(points)

            loss = criterion(labels, indexs, output, preds, coords, feats)
            loss.backward()
            optimizer.step()
            scheduler.step()

            output_np = np.argmax(output.cpu().detach().numpy(), axis=2).copy()
            target_np = labels.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(args.num_classes)))
            cm += cm_

            oa = f"{Metrics.stats_overall_accuracy(cm):.3f}"
            aa = f"{Metrics.stats_accuracy_per_class(cm)[0]:.3f}"
            iou = f"{Metrics.stats_iou_per_class(cm)[0]:.3f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=oa, IOU=iou, LOSS=f"{train_loss / cm.sum():.3e}")

        # validation
        net.eval()
        cm_test = np.zeros((args.num_classes, args.num_classes))
        test_loss = 0

        t = tqdm(test_loader, ncols=100, desc="Test {}".format(epoch))

        with torch.no_grad():
            for points, labels in t:

                points = points.permute(0, 2, 1)
                points = points.cuda(args.gpu_id)
                labels = labels.cuda(args.gpu_id)

                output, coords, feats, preds, indexs = net(points)
                loss = criterion(labels, indexs, output, preds, coords, feats)

                output_np = np.argmax(output.cpu().detach().numpy(), axis=2).copy()
                target_np = labels.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(args.num_classes)))
                cm_test += cm_

                oa_val = f"{Metrics.stats_overall_accuracy(cm_test):.3f}"
                aa_val = f"{Metrics.stats_accuracy_per_class(cm_test)[0]:.3f}"
                iou_val = f"{Metrics.stats_iou_per_class(cm_test)[0]:.3f}"

                test_loss += loss.detach().cpu().item()

                t.set_postfix(OA=oa_val, IOU=iou_val, LOSS=f"{test_loss / cm_test.sum():.3e}")

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))
        if maxIOU < float(iou_val):
            # save the model
            torch.save(net.state_dict(), os.path.join(root_folder, "state_dict" + str(iou_val) + ".pth"))
            maxIOU = float(iou_val)
        # write the logs
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.flush()

    logs.close()


def test(args):

    # create the network
    print("Creating network...")
    net = PointMLP()
    net.cuda(args.gpu_id)
    net = torch.nn.DataParallel(net)

    net.load_state_dict(torch.load(os.path.join(args.save_dir, "state_dict.pth")))
    # net.cuda(args.gpu_id)
    net.eval()

    # TODO: test the model
    pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",    default="results/",   type=str)
    parser.add_argument("--data_path",   default="data",       type=str)
    parser.add_argument("--batch_size",  default=2,            type=int)
    parser.add_argument("--num_points",  default=2048,         type=int)
    parser.add_argument("--test_area",   default=5,            type=int)
    parser.add_argument("--threads",     default=1,            type=int)
    parser.add_argument("--pretrain",    default=False,        type=bool)
    parser.add_argument("--lr",          default=0.0001,       type=float)
    parser.add_argument("--epochs",      default=350,          type=int)
    parser.add_argument("--model",       default="PointMLP",   type=str)
    parser.add_argument("--num_classes", default=13,           type=int)
    parser.add_argument("--transform",   default=False,        type=bool)
    parser.add_argument("--gpu_id",      default=0,            type=int)
    args = parser.parse_args()


    train(args)
    # test(args)


if __name__ == '__main__':
    main()