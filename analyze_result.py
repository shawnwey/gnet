import argparse
from core.analyze import analyze_net_pklResults


def main():
    parser = argparse.ArgumentParser()

    # 命名规定：网络名称（用net拼接）-其他实验因素，下划线区分
    parser.add_argument('--exp_id', type=str, default='EfficientNetAutoAttB4-spatial')
    args, _ = parser.parse_known_args()

    analyze_net_pklResults(args.exp_id)

if __name__=='__main__':
    main()