import argparse
import logging
import torch
import datetime
import os
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim

from dataset import *
from utils.metrics import *
from utils.plot import loss_plot, metrics_plot

from models.Unet import Unet, resnet34_unet

# from attention_unet import AttU_Net
# from channel_unet import myChannelUnet
# from r2unet import R2U_Net
# from segnet import SegNet
# from unetpp import NestedUNet
# from fcn import get_fcn8s


def getArgs():
    parse = argparse.ArgumentParser()

    parse.add_argument("--deepsupervision", default=0)
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument(
        "--batch_size", type=int, default=5
    )  # set small for saving cuda memory
    parse.add_argument(
        "--action", type=str, help="train/test/train&test", default="train&test"
    )
    parse.add_argument(
        "--model",
        metavar="model_structure",
        default="Unet",
        help="Unet/ResNet34_Unet",
        # unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s
    )
    parse.add_argument(
        "--dataset",
        default="liver",
        help="dataset name:liver/lung",
        # esophagus/dsb2018Cell/corneal/driveEye/isbiCell
    )
    parse.add_argument("--res", default="results", help="results dir")
    parse.add_argument("--threshold", type=float, default=None)
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()
    return args


def gettime():
    i = datetime.datetime.now()
    return "{}-{}-{}-{}_{}_{}".format(
        i.year, i.month, i.day, i.hour, i.minute, i.second
    )


def getLog(args):
    # dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    dirname = os.path.join(current_dirname, "log")
    filename = dirname + "/log.log"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )
    return logging


def getModel(args):
    if args.model == "Unet":
        model = Unet(3, 1).to(device)
    if args.model == "resnet34_unet":
        model = resnet34_unet(1, pretrained=False).to(device)
    return model


def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None

    if args.dataset == "liver":
        train_dataset = LiverDataset(
            r"train", transform=x_transforms, target_transform=y_transforms
        )
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)

        val_dataset = LiverDataset(
            r"val", transform=x_transforms, target_transform=y_transforms
        )
        val_dataloaders = DataLoader(val_dataset, batch_size=1)

        test_dataloaders = val_dataloaders

    if args.dataset == "lung":
        train_dataset = LungDataset(
            r"train", transform=x_transforms, target_transform=y_transforms
        )
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)

        val_dataset = LungDataset(
            r"val", transform=x_transforms, target_transform=y_transforms
        )
        val_dataloaders = DataLoader(val_dataset, batch_size=1)

        test_dataset = LungDataset(
            r"test", transform=x_transforms, target_transform=y_transforms
        )
        test_dataloaders = DataLoader(test_dataset, batch_size=1)

    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        # i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数
        # print(num)
        for x, _, pic, mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)  # 获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(mask[0], img_y)
            # if i < num:
                # i += 1  # 处理验证集下一张图

        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total / num

        print("Miou = %f, Aver_hd = %f, Aver_dice = %f" % (aver_iou, aver_hd, aver_dice))
        logging.info(
            "Miou=%f, Aver_hd=%f, Aver_dice=%f" % (aver_iou, aver_hd, aver_dice)
        )

        if aver_iou > best_iou:
            print("aver_iou: {} > best_iou: {}".format(aver_iou, best_iou))
            logging.info("aver_iou:{} > best_iou:{}".format(aver_iou, best_iou))
            logging.info("===========> save best model!")
            best_iou = aver_iou
            print("===========> save best model!")
            torch.save(
                model.state_dict(),
                r"./weights/"
                + str(args.model)
                + "_"
                + str(args.dataset)
                + "_batchsize"
                + str(args.batch_size)
                + "_epoch"
                + str(args.epoch)
                + ".pth",
            )
        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    for epoch in range(num_epochs):
        model = model.train()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output, labels)
            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                "%d/%d, train_loss: %0.3f"
                % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item())
            )
            logging.info(
                "%d/%d, train_loss: %0.3f"
                % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item())
            )
        loss_list.append(epoch_loss)

        print("validating...")
        best_iou, aver_iou, aver_dice, aver_hd = val(model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss: %0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss: %0.3f" % (epoch, epoch_loss))

    loss_plot(args, current_dirname, loss_list)
    metrics_plot(args, current_dirname, "iou & dice", iou_list, dice_list)
    metrics_plot(args, current_dirname, "hd", hd_list)
    return model


def test(val_dataloaders, save_predict=False):
    logging.info("final test........")
    if save_predict == True:
        dir = os.path.join(current_dirname, "predict")
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print("dir already exist!")
    model.load_state_dict(
        torch.load(
            r"./weights/"
            + str(args.model)
            + "_"
            + str(args.dataset)
            + "_batchsize"
            + str(args.batch_size)
            + "_epoch"
            + str(args.epoch)
            + ".pth",
            # map_location="cpu",
        )
    )  # 载入训练好的模型
    model.eval()

    # plt.ion() #开启动态模式
    with torch.no_grad():
        # i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数
        for pic, _, pic_path, mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = (
                    torch.squeeze(predict).cpu().numpy()
                )  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou = get_iou(mask_path[0], predict)
            miou_total += iou  # 获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0], predict)
            dice_total += dice

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title("input")
            plt.imshow(Image.open(pic_path[0]))
            # print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title("predict")
            plt.imshow(predict, cmap="Greys_r")
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title("mask")
            plt.imshow(Image.open(mask_path[0]), cmap="Greys_r")
            # print(mask_path[0])
            if save_predict == True:
                if args.dataset == "driveEye":
                    saved_predict = dir + "/" + mask_path[0].split("\\")[-1]
                    saved_predict = "." + saved_predict.split(".")[1] + ".tif"
                    plt.savefig(saved_predict)
                else:
                    plt.savefig(dir + "/" + mask_path[0].split("\\")[-1])

            # plt.pause(0.01)
            print(mask_path[0].split("\\")[-1])
            print("iou = {}, dice = {}".format(iou, dice))
            # if i < num:
            #     i += 1  # 处理验证集下一张图

        print(
            "Miou = %f, Aver_hd = %f, M_dice = %f"
            % (miou_total / num, hd_total / num, dice_total / num)
        )
        logging.info(
            "Miou = %f, Aver_hd = %f, M_dice = %f"
            % (miou_total / num, hd_total / num, dice_total / num)
        )
        plt.close()
        # print('M_dice=%f' % (dice_total / num))


if __name__ == "__main__":
    x_transforms = transforms.Compose(
        [
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # ->[-1,1]
        ]
    )

    y_transforms = transforms.ToTensor()  # mask只需要转换为tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    args = getArgs()
    current_dirname = os.path.join("results", args.model, gettime())
    print("Path: ", current_dirname)
    logging = getLog(args)
    print("============ task info ============")
    print(
        "models:%s,\n \
        epoch:%s,\n \
        batch size:%s\n \
        dataset:%s"
        % (args.model, args.epoch, args.batch_size, args.dataset)
    )
    logging.info(
        "\n=======\n \
        models:%s,\n \
        epoch:%s,\n \
        batch size:%s\n \
        dataset:%s\n \
        ========"
        % (args.model, args.epoch, args.batch_size, args.dataset)
    )
    print("============ task start ============")

    model = getModel(args)
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    if "train" in args.action:
        train(model, criterion, optimizer, train_dataloaders, val_dataloaders, args)
    if "test" in args.action:
        test(test_dataloaders, save_predict=True)
