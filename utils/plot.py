import matplotlib.pyplot as plt
import os


def loss_plot(args, current_dirname, loss):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = os.path.join(current_dirname, "plot")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = (
        plot_save_path
        + str(args.model)
        + "_"
        + str(args.dataset)
        + "_batch_size"
        + str(args.batch_size)
        + "_epoch"
        + str(args.epoch)
        + "_loss.jpg"
    )
    plt.figure()
    plt.plot(x, loss, label="loss")
    plt.legend()
    plt.savefig(save_loss)


def metrics_plot(arg, current_dirname, name, *args):
    num = arg.epoch
    names = name.split("&")
    metrics_value = args
    i = 0
    x = [i for i in range(num)]
    plot_save_path = os.path.join(current_dirname, "plot")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = (
        plot_save_path
        + str(arg.model)
        + "_"
        + str(arg.dataset)
        + "_batch_size"
        + str(arg.batch_size)
        + "_epoch"
        + str(arg.epoch)
        + "_"
        + name
        + ".jpg"
    )
    plt.figure()
    for l in metrics_value:
        plt.plot(x, l, label=str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)
