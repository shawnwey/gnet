import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import ticker
import json
import os


def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = np.exp(-(x - mean) ** 2 / (2 * std ** 2))  # / sigma
    return x_out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_guass():
    # sin & cos曲线
    x = np.arange(1, 10, 0.1)
    # y1 = 0.1 * x
    y1 = x
    y2 = sigmoid(x)
    y3 = 10 * GaussProjection(x, x.mean(), x.std())

    plt.plot(x, y1, label="Origin", linestyle="--")
    plt.plot(x, y3, label="Gauss")
    # plt.plot(x,y2,label="Sigmoid",linestyle='dotted')

    # plt.ylim((-0.1,1.1))
    # plt.ylim()

    plt.xlabel("Before rectification")
    plt.ylabel("After rectification")
    plt.legend()  # 打上标签

    plt.savefig('./OriginAndGauss.svg', format='svg')
    plt.show()


def mse(x, gt):
    return np.sum((x - gt) ** 2)


def afi(x, gt1, gt2):
    return mse(x, gt1) + max(mse(x, gt1) - mse(x, gt2), 0)


def draw_afi():
    x = np.arange(-10, 30, 1)
    g1 = 10
    g2 = 20
    gt1 = np.full(x.shape, g1)
    gt2 = np.full(x.shape, g2)

    y = [afi(x[i], gt1[i], gt2[i]) for i in range(len(x))]
    y_mse = [mse(x[i], gt1[i]) for i in range(len(x))]

    infinit = np.arange(-1000, 1000, 1)
    g1x = np.full(infinit.shape, g1)
    g2x = np.full(infinit.shape, g2)
    plt.plot(x, y_mse, linestyle=':', marker='*', label="MSE")
    plt.plot(x, y, label="AFI")
    plt.plot(g1x, infinit, label="GT1", linestyle="--", color='g')
    plt.plot(g2x, infinit, label="GT2", linestyle="--", color='r')

    plt.ylim((-10, 420))

    plt.xlabel("Predicted Value")
    plt.ylabel("Loss")
    plt.legend()  # 打上标签

    plt.savefig('./drawAFI.png', format='png')
    plt.show()


def draw_person_num():
    COCO = [98.7, 1.2, 0.1]
    CrowdPose = [92.2, 7.7, 0.1]
    OCHuman = [18.8, 80.3, 0.9]
    x = [1, 2, 3]
    width = 1.0 / 3.77
    x1 = [i - width for i in x]
    x2 = [i + width for i in x]
    total_width, n = 3.5, 13
    width = total_width / n

    bar1 = plt.bar(x1, COCO, width=width, label='COCO')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    bar2 = plt.bar(x, CrowdPose, width=width, label='CrowdPose')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    bar3 = plt.bar(x2, OCHuman, width=width, label='OCHuman')

    plt.xlabel("Number of Persons in BB(IoU > 0.5)")
    plt.ylabel("Number of BBs(%)")
    plt.legend()

    for bar in [bar1, bar2, bar3]:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.6, str(height), ha="center", va="bottom")

    plt.savefig('./IouNumPerson.svg', format='svg')
    plt.show()


def draw_subplot():
    fig = plt.figure()
    # -----------------
    coco = fig.add_subplot(2, 2, 1)
    x = np.arange(0, 1, 1.0 / 14)
    y = [68, 6.7, 5.8, 4.5, 4, 4.5, 3.5, 1, 1, 0.5, 0.5, 0.5, 0.4, 0.1]
    plt.bar(x, y, width=0.064)
    coco.yaxis.set_major_formatter(ticker.PercentFormatter())
    coco.set_title('MS COCO')

    # -----------------
    mpii = fig.add_subplot(2, 2, 2)
    x = np.arange(0, 1, 1.0 / 14)
    y = [87, 3, 2, 2, 1, 1, 1, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
    plt.bar(x, y, width=0.066)
    mpii.yaxis.set_major_formatter(ticker.PercentFormatter())
    mpii.set_title('MPII')

    # -----------------
    ai = fig.add_subplot(2, 2, 3)
    x = np.arange(0, 1, 1.0 / 14)
    y = [62, 3, 6, 4, 2.5, 5.5, 3.5, 2.5, 3, 2.5, 1.9, 1.4, 1.1, 1.1]
    plt.bar(x, y, width=0.066)
    ai.yaxis.set_major_formatter(ticker.PercentFormatter())
    ai.set_title('AI Challenger')

    # -----------------
    crowdpose = fig.add_subplot(2, 2, 4)
    x = np.arange(0, 1, 0.1)
    y = [10, 10.2, 10.3, 9.7, 10, 10.3, 9.8, 9.7, 9.5, 10.5]
    plt.bar(x, y, width=0.066)
    crowdpose.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.xlim((-0.075, 1.01))
    crowdpose.set_title('CrowdPose')

    fig.tight_layout()

    plt.savefig('./CrowdIndex.svg', format='svg')
    plt.show()


def count_afi():
    afi_json = 'G:\Wei\grad\ccc\Rccc\exp\Train_two_2_32_AFILoss/run-log-tag-interference_point_count.json'
    x = []
    y = []
    with open(afi_json, 'r') as f:
        x_ys = json.load(f)
        s = set()
        for xy in x_ys:
            if xy[1] in s:
                continue
            s.add(xy[1])
            x.append(xy[1])
            y.append(xy[2])

    # infinit = np.arange(-1000, 1000, 1)
    plt.plot(x, y, label="interference keypoint num")

    # plt.ylim((-10, 420))

    plt.xlabel("epoch")
    # plt.ylabel("interference keypoint num")
    plt.legend()  # 打上标签
    # plt.title("num")
    plt.tight_layout()

    plt.savefig('./count_AFI.svg', format='svg')
    plt.show()


def multiSubFillPlot():
    # R -----------------
    r_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    r_real_y = [0.001, 0.01, 0.04, 0.11, 0.49, 0.35]
    r_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    r_fake_y = [0.0001, 0.005, 0.03, 0.10, 0.49, 0.35]
    r = [r_real_x, r_real_y, r_fake_x, r_fake_y]
    # G -----------------
    g_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    g_real_y = [0.001, 0.01, 0.05, 0.17, 0.55, 0.23]
    g_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    g_fake_y = [0.0001, 0.005, 0.035, 0.14, 0.57, 0.24]
    g = [g_real_x, g_real_y, g_fake_x, g_fake_y]
    # B -----------------
    b_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    b_real_y = [0.001, 0.01, 0.05, 0.18, 0.53, 0.23]
    b_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    b_fake_y = [0.0001, 0.005, 0.035, 0.15, 0.56, 0.24]
    b = [b_real_x, b_real_y, b_fake_x, b_fake_y]
    # y -----------------
    y_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    y_real_y = [0.001, 0.01, 0.04, 0.14, 0.54, 0.26]
    y_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    y_fake_y = [0.0001, 0.005, 0.03, 0.13, 0.57, 0.27]
    y = [y_real_x, y_real_y, y_fake_x, y_fake_y]
    # cb -----------------
    cb_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    cb_real_y = [0.001, 0.02, 0.06, 0.16, 0.43, 0.34]
    cb_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    cb_fake_y = [0.001, 0.03, 0.08, 0.22, 0.49, 0.19]
    cb = [cb_real_x, cb_real_y, cb_fake_x, cb_fake_y]
    # cr -----------------
    cr_real_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    cr_real_y = [0.001, 0.01, 0.02, 0.09, 0.39, 0.48]
    cr_fake_x = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    cr_fake_y = [0.001, 0.015, 0.025, 0.08, 0.50, 0.37]
    cr = [cr_real_x, cr_real_y, cr_fake_x, cr_fake_y]

    font_size = 16
    colors = [r, g, b, y, cb, cr]
    for i, color in enumerate(colors):
        # plot
        plt.fill_between(color[0], color[1], color="darkgreen", alpha=0.3, label='Real')  #
        plt.fill_between(color[2], color[3], color='red', alpha=0.3, label='Fake')  # color="skyblue"

        plt.plot(color[0], color[1], color="darkgreen")
        plt.plot(color[2], color[3], color='red', linestyle="--")

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        ax = plt.axes()
        plt.ylim((0, 0.57))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        plt.xlabel("relative", size=font_size + 3)
        plt.ylabel("proportion", size=font_size + 3)
        plt.legend(loc='upper left', prop={'size'   : font_size})
        # plt.title(str(i))

        plt.tight_layout()
        plt.savefig(str(i) + '.png', bbox_inches='tight')
        plt.show()


def drawFf():
    figs = ['RAW', 'High Quality', 'Low Quality']

    methods =['Steg.\nFeatures+SVM', 'Cozzolino\net al.', 'Bayar and\nStamm', 'Rahmouni\net al.', 'MesoNet', ' XceptionNet']
    colorBuf = ['skyblue','bisque','lightgreen','lightcoral','darkcyan', 'violet']
    font_size = 13
    # RAW -----------------
    raw_x = list( np.array(range(len(methods))) + 1 )
    raw_y = [97.63, 98.57, 98.74, 97.03, 95.23, 99.26]
    # high quality -----------------
    hq_x = list( np.array(range(len(methods))) + 1 )
    hq_y = [70.97, 78.45, 82.97, 79.08, 83.10, 95.73]
    # Low quality-----------------
    l_x = list( np.array(range(len(methods))) + 1 )
    l_y = [55.98, 58.69, 66.84, 61.18, 70.47, 81.00]
    accs = [[raw_x, raw_y], [hq_x, hq_y], [l_x, l_y]]

    for i, acc in enumerate(accs):
        plt.figure(figsize=(8, 5))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.grid(axis='y', zorder=0)   # linestyle="-.",
        ax.axhline(80, linewidth=1.5, linestyle='--', color='r')

        plt.xticks(acc[0], methods)
        plt.tick_params(labelsize=font_size)
        plt.bar(acc[0], acc[1], width=0.5, color=colorBuf, zorder=10)

        plt.ylim((0, 100))
        plt.ylabel("Accuracy", size=font_size)
        # plt.legend()

        plt.title(figs[i], size=font_size+1)
        plt.savefig(figs[i] + '.png')
        plt.show()

def drawLoss():
    jsonDir = r'G:\Wei\grad\ccc\Rccc\deepfake\img_data'
    lossJsonList = [
        '1-run-Xception_log-tag-train_loss.json',
        '2-run-SgeMspNet-endpoints_log-tag-train_loss.json',
        '3-run-EfficientNetAutoAttB4-spatial_log-tag-train_loss.json',
        '4-run-SgeNet-groups8_endpoints_log-tag-train_loss.json'
    ]
    lossJsonList = lossJsonList[::-1]
    methods = ['Baseline+MSP+GSA', 'Baseline+GSA', 'Baseline+MSP', 'Baseline']
    methods = methods[::-1]

    font_size = 15
    ax = plt.axes()
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    for i, lossJsonFile in enumerate(lossJsonList):
        lossJsonFile = os.path.join(jsonDir, lossJsonFile)
        x = []
        y = []
        with open(lossJsonFile, 'r') as f:
            x_ys = json.load(f)
            for xy in x_ys:
                if xy[1] < 30001:
                    x.append(xy[1])
                    y.append(xy[2])
        plt.plot(x, y, label=methods[i])

    plt.xlabel("iter", size=font_size)
    plt.ylabel("loss", size=font_size)
    plt.legend(prop={'size': font_size})  # 打上标签
    # plt.title("num")
    plt.tight_layout()

    plt.savefig('trainLoss.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')

def drawAUC():
    jsonDir = r'G:\Wei\grad\ccc\Rccc\deepfake\img_data'
    aucJsonList = [
        '1-run-Xception_log-tag-train_roc_auc.json',
        '2-run-SgeMspNet-endpoints_log-tag-train_roc_auc.json',
        '3-run-EfficientNetAutoAttB4-spatial_log-tag-train_roc_auc.json',
        '4-run-SgeNet-groups8_endpoints_log-tag-train_roc_auc.json'
    ]
    aucJsonList = aucJsonList[::-1]
    methods = ['Baseline+MSP+GSA', 'Baseline+GSA', 'Baseline+MSP', 'Baseline']
    methods = methods[::-1]

    font_size = 15
    ax = plt.axes()
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    for i, lossJsonFile in enumerate(aucJsonList):
        lossJsonFile = os.path.join(jsonDir, lossJsonFile)
        x = []
        y = []
        with open(lossJsonFile, 'r') as f:
            x_ys = json.load(f)
            for xy in x_ys:
                if xy[1] < 30001:
                    x.append(xy[1])
                    y.append(xy[2])
        plt.plot(x, y, label=methods[i])

    plt.xlabel("iter", size=font_size)
    plt.ylabel("auc", size=font_size)
    plt.legend(prop={'size': font_size})  # 打上标签
    # plt.title("num")
    plt.tight_layout()

    plt.savefig('trainAUC.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')

if __name__ == '__main__':
    # count_afi()
    # draw_guass()
    # draw_subplot()
    # draw_person_num()
    # x = 10
    # gt1 = 10
    # gt2 = 20
    # a = mse(x, gt1) - mse(x, gt2)
    # print(max(a, 0))
    multiSubFillPlot()
    # drawFf()
    # drawLoss()
    # drawAUC()
