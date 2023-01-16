from utils import parse_args
from train import GANSSL
import torch


# main to run
def main():
    # parse arguments
    args = parse_args()
    # print(vars(args))
    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
    
    if args.change_nlabels:
        list_num_of_labels = [50, 100, 600, 1000, 3000]
        for num_labels in list_num_of_labels:
            args.num_labels = num_labels
            if num_labels == 50:
                args.batch_size = 32
            print("number of labels = ", args.num_labels)
            gan = GANSSL(args)
            gan.train()
            gan.visualize_results(args.epoch)
            print("="*30)

    if args.change_alpha:
        args.num_labels = 100
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            args.alpha = alpha
            print("alpha = ", alpha)
            gan = GANSSL(args)
            gan.train()
            gan.visualize_results(args.epoch)
            print("="*30)

    print("Training finished!")


if __name__ == "__main__":
    main()