import argparse
from email.policy import default
from urllib.parse import ParseResult

def create_parser():
    description = 'Relativistic Margin Cosine - Relativistic Large Margin Softmax GAN-SSL'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--device', type=str, default='cuda')

    # save results
    parser.add_argument('--acc_time', type=str, default='acc_time')
    parser.add_argument('--generated_images_dir', type=str, default='generated_images')

    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='rmcos', choices=['rmcos', 'rlmsoftmax','rmarc', 'margingan'], required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--z_dim', type=int, default=64)

    parser.add_argument('--lrC', type=float, default=0.1)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--momentum', type=str, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--scheduler', type=bool, default=False)
    
    # m, s
    parser.add_argument('--m', type=float, default=0.15)
    parser.add_argument('--s', type=float, default=10.0)

    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=40)
    parser.add_argument('--num_labeled', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.9)
    
    parser.add_argument('--checkpoint_epochs', type=int, default=1)
    # visualize
    parser.add_argument('--num_samples', type=int, default=100)

    # resume
    parser.add_argument('--resume', type=str, default='')
    # labeled_indexes
    parser.add_argument('--labeled_indexes', type=str, default='')

    return parser.parse_args()

# if use str2bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# if use str2epoch
def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs