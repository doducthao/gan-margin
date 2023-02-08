from cProfile import label
from dataloader import dataloader_given_indexes
from net import generator, discriminator, classifier, initialize_weights
from utils import save_images, generate_animation, train_loss_plot, test_loss_plot, acc_plot
from losses import clf_loss, inverted_cross_entropy, d_loss_cosine_margin, g_loss_cosine_margin, d_loss_multi_angular_2k, g_loss_multi_angular_2k, g_loss_additive_angular_arccos, d_loss_additive_angular_arccos
from ramps import linear_rampup, cosine_rampdown
from parser import create_parser 
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import numpy as np
import pickle
import time
from datetime import datetime
import logging

def save_checkpoint(state, dirpath):
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, best_path)

def adjust_learning_rate(args, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualize_generated_images(epoch, sample_z_, args, fix=True):
    G.eval()
    tot_num_samples = min(args.num_samples, args.batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    if fix:
        """ fixed noise """
        samples = G(sample_z_)
    else:
        """ random noise """
        sample_z_ = torch.rand((args.batch_size, args.z_dim)).to(args.device)
        samples = G(sample_z_)

    if args.device == 'cuda':
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        os.path.join(generated_images_dir, 'epoch%03d' % epoch + '.png'))

def load_checkpoint(args, C, G, D, C_optimizer):
    LOG.info("=> loading checkpoint '{}'".format(args.resume))
    best_file = os.path.join(args.resume, 'best.ckpt')
    G_file = os.path.join(args.resume, 'G.pkl')
    D_file = os.path.join(args.resume, 'D.pkl')

    assert os.path.isfile(best_file), "=> no checkpoint found at '{}'".format(best_file)
    assert os.path.isfile(G_file), "=> no checkpoint found at '{}'".format(G_file)
    assert os.path.isfile(D_file), "=> no checkpoint found at '{}'".format(D_file)

    checkpoint = torch.load(best_file)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    C.load_state_dict(checkpoint['state_dict'])
    C_optimizer.load_state_dict(checkpoint['optimizer'])

    G.load_state_dict(torch.load(G_file))
    D.load_state_dict(torch.load(D_file))

    LOG.info("=> loaded checkpoint '{}' | Best accuracy: {} (epoch {})".format(args.resume, round(best_acc,4), checkpoint['epoch']))
    return best_acc, start_epoch

if __name__ == '__main__':
    args = create_parser()
    if args.num_labeled == 50:
        args.batch_size = 32 
    id_txt = 'id-' + args.labeled_indexes.split("/")[-1].replace(".txt", "")
    if args.resume:
        checkpoint_path = args.resume
        date_time_now = datetime.now()
        date_time_now = "{:%Y-%m-%d_%H:%M:%S}".format(date_time_now)
    else:
        date_time_now = datetime.now()
        date_time_now = "{:%Y-%m-%d_%H:%M:%S}".format(date_time_now)
        if args.mode == "rmcos":
            checkpoint_path = os.path.join("out_rmcos", args.dataset, str(args.num_labeled), str(args.alpha), str(args.m), id_txt,
                                        date_time_now)
        elif args.mode == "rmlsoftmax":
            checkpoint_path = os.path.join("out_rmlsoftmax", args.dataset, str(args.num_labeled), str(args.alpha), str(args.m), id_txt,
                                        date_time_now)
        elif args.mode == "rmarc":
            checkpoint_path = os.path.join("out_rmarc", args.dataset, str(args.num_labeled), str(args.alpha), str(args.m), id_txt,
                                        date_time_now)

    generated_images_dir = os.path.join(checkpoint_path, args.generated_images_dir)

    acc_time_file = os.path.join(checkpoint_path, args.acc_time + '.txt')
    acc_time_best_file = os.path.join(checkpoint_path, args.acc_time + '_best.txt')
    
    os.makedirs(checkpoint_path, exist_ok=True) 
    os.makedirs(generated_images_dir, exist_ok=True)

    logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(checkpoint_path, "logging.log"),
            filemode="a")
            
    LOG = logging.getLogger("main")
    LOG.info("Start training at {}".format(date_time_now))
    LOG.info(f"args: {args}")

    # create nets
    G = generator(input_dim=args.z_dim, output_dim=1, input_size=args.input_size).to(args.device)
    D = discriminator(input_size=args.input_size).to(args.device)
    C = classifier().to(args.device)

    # use multiple gpus
    if torch.cuda.device_count() > 1:
        LOG.info(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        G = torch.nn.DataParallel(G)
        C = torch.nn.DataParallel(C)
        D = torch.nn.DataParallel(D)

    # initialize weights
    G.apply(initialize_weights)
    D.apply(initialize_weights)
    C.apply(initialize_weights)

    # create optimizers
    C_optimizer = optim.SGD(C.parameters(), lr=args.lrC, momentum=args.momentum)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    
    # create schedulers
    if args.scheduler:
        C_scheduler_warmup = StepLR(C_optimizer, step_size=10, gamma=0.9)
        C_scheduler = StepLR(C_optimizer, step_size=10, gamma=0.9)

    # fixed noise
    sample_z_ = torch.rand(args.batch_size, args.z_dim).to(args.device)

    # load dataset
    labeled_loader, unlabeled_loader, test_loader, labeled_indexes = dataloader_given_indexes(args)
    LOG.info(f'Labeled size: {len(labeled_indexes)}, Unlabeled size: {50000 - len(labeled_indexes)}')

    train_hist = {}
    train_hist['D_loss'] = []
    train_hist['G_loss'] = []
    train_hist['C_loss'] = []
    train_hist['test_loss'] = []

    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    train_hist['test_accuracy'] = []

    labeled_steps = len(labeled_loader.dataset) // args.batch_size
    unlabeled_steps = len(unlabeled_loader.dataset) // args.batch_size
    test_steps = len(test_loader.dataset) // args.batch_size

    y_real = torch.ones(args.batch_size, 1).to(args.device)

    # load checkpoints
    if args.resume:
        best_acc, args.start_epoch = load_checkpoint(args, C, G, D, C_optimizer)
    else:
        best_acc, args.start_epoch = 0, 1
    best_time = None 

    # train D
    D.train()
    start_time = time.time()
    for epoch in range(args.start_epoch, args.num_epoch+1):
        # print learning rate after each epoch
        LOG.info('Epoch: {}  || lrC: {}, lrD: {}, lrG: {}'.format(
            epoch,
            C_optimizer.param_groups[0]['lr'],
            D_optimizer.param_groups[0]['lr'],
            G_optimizer.param_groups[0]['lr']))
        # train G
        G.train()
        epoch_start_time = time.time()

        # warms up for C
        if not args.resume:
            if epoch == 1:
                if args.num_labeled == 50:
                    gate = 0.6
                elif args.num_labeled == 100:
                    gate = 0.8
                elif args.num_labeled == 600:
                    gate = 0.90
                elif args.num_labeled == 1000:
                    gate = 0.93
                elif args.num_labeled == 3000:
                    gate = 0.95

                LOG.info(f'Training C to archieve at least {gate} accuracy')
                correct_rate = 0
                count = 0 # support adjust learning rate of C 
                while True:
                    for iter, (x_labeled, y_labeled) in enumerate(labeled_loader):
                        x_labeled, y_labeled = x_labeled.to(args.device), y_labeled.to(args.device)
                        C.train() # train C
                        C_optimizer.zero_grad()
                        _, output = C(x_labeled)
                        C_loss = clf_loss(output, y_labeled)
                        C_loss.backward()
                        C_optimizer.step()
                        count += 1

                        # evaluate
                        if iter == labeled_steps:
                            C.eval()
                            test_loss = 0
                            correct = 0
                            with torch.no_grad():
                                for x_test, y_test in test_loader:
                                    x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                                    _, output = C(x_test)
                                    test_loss += clf_loss(output, y_test).item() # reduction = 'mean'
                                    pred = torch.argmax(output, dim=1, keepdim=True) # get the index of the max log-probability
                                    correct += pred.eq(y_test.view_as(pred)).sum().item()
                            test_loss /= test_steps

                            LOG.info('Test loss: {:.4f} || Accuracy: {}/{} ({:.0f}%)\n'.format(
                                test_loss, correct, len(test_loader.dataset),
                                100. * correct / len(test_loader.dataset)
                                ))
                            correct_rate = correct / len(test_loader.dataset)
                            train_hist['test_accuracy'].append(correct_rate)
                        
                        if args.scheduler:
                            if count % 20 == 0:
                                #  update learning rate
                                C_scheduler_warmup.step()
                                # LOG.info(f'lrC: {C_optimizer.param_groups[0]["lr"]}')
                                LOG.info(f'Step: {count} | Update lrC :{C_scheduler_warmup.get_last_lr()[0]}')

                    if correct_rate >= gate:
                        break

        C_optimizer.param_groups[0]['lr'] = args.lrC # recover C learning rate
        correct_wei = 0
        number = 0
        labeled_iter = labeled_loader.__iter__()

        # train C on unlabeled data
        C.train()
        for iter, (x_u, y_u) in enumerate(unlabeled_loader):
            # adjust_learning_rate(args, C_optimizer, epoch, iter, len(unlabeled_loader)) # thu nghiem 

            if iter == len(unlabeled_loader.dataset) // args.batch_size:
                if epoch > 0:
                    LOG.info('\nPseudo tag || Accuracy: {}/{} ({:.0f}%)\n'.format(
                        correct_wei, number,
                        100. * correct_wei / number))
                break

            try:
                x_labeled, y_labeled = labeled_iter.__next__()
                if len(x_labeled) != args.batch_size: # number of labels < batch size -> generate a new iter
                    labeled_iter = labeled_loader.__iter__()
                    x_labeled, y_labeled = labeled_iter.__next__()
            except StopIteration:
                labeled_iter = labeled_loader.__iter__()
                x_labeled, y_labeled = labeled_iter.__next__()

            z_ = torch.rand(args.batch_size, args.z_dim, device=args.device)
            x_labeled, y_labeled, x_u, y_u = x_labeled.to(args.device), y_labeled.to(args.device), x_u.to(args.device), y_u.to(args.device)

            # update C
            C_optimizer.zero_grad()

            _, C_labeled_pred = C(x_labeled)
            C_labeled_loss = clf_loss(C_labeled_pred, y_labeled)

            _, C_unlabeled_pred = C(x_u)
            C_unlabeled_true = torch.argmax(C_unlabeled_pred, dim=1) # make pseudo labels for unlabeled data 
            C_unlabeled_loss = clf_loss(C_unlabeled_pred,  C_unlabeled_true)

            correct_wei += C_unlabeled_true.eq(y_u).sum().item()
            number += len(y_u)

            G_ = G(z_)
            C_fake_pred, _  = C(G_)
            C_fake_true = torch.argmax(C_fake_pred, dim=1) # make pseudo labels for fake data  
            C_fake_true = F.one_hot(C_fake_true, 10) # make one-hot for y true
            C_fake_loss = inverted_cross_entropy(C_fake_pred, C_fake_true)

            C_loss = C_labeled_loss + C_unlabeled_loss + C_fake_loss
        
            train_hist['C_loss'].append(C_loss.item())
            C_loss.backward()
            C_optimizer.step()

            # update D network
            D_optimizer.zero_grad()
            D_labeled = D(x_labeled)
            D_unlabeled = D(x_u)
            D_real = (D_labeled + D_unlabeled)/2 

            G_ = G(z_)
            D_fake = D(G_)
            if args.mode == "rmcos":
                D_loss = d_loss_cosine_margin(D_real, D_fake, torch.ones_like(D_real), args.m, args.s)
            elif args.mode == "rmlsoftmax":
                D_loss = d_loss_multi_angular_2k(D_real, D_fake, torch.ones_like(D_real), args.m, args.s)
            elif  args.mode == "rmarc":
                D_loss = d_loss_additive_angular_arccos(D_real, D_fake, torch.ones_like(D_real), args.m, args.s)

            train_hist['D_loss'].append(D_loss.item())
            D_loss.backward()
            D_optimizer.step()

            # update G
            G_optimizer.zero_grad()
            z_ = torch.rand(args.batch_size, args.z_dim, device=args.device)
            G_ = G(z_)
            D_fake = D(G_)

            D_labeled = D(x_labeled)
            D_unlabeled = D(x_u)
            D_real = (D_labeled + D_unlabeled)/2

            if args.mode == "rmcos":
                G_loss_D = g_loss_cosine_margin(D_real, D_fake, torch.ones_like(D_fake), args.m, args.s)
            elif args.mode == "rmlsoftmax":
                G_loss_D = g_loss_multi_angular_2k(D_real, D_fake, torch.ones_like(D_fake), args.m, args.s)
            elif args.mode == "rmarc":
                G_loss_D = g_loss_additive_angular_arccos(D_real, D_fake, torch.ones_like(D_fake), args.m, args.s)

            _, C_fake_pred = C(G_)
            C_fake_true = torch.argmax(C_fake_pred, dim=1) # (vals, indices)
            G_loss_C  = clf_loss(C_fake_pred, C_fake_true)

            G_loss = args.alpha * G_loss_D + (1-args.alpha) * G_loss_C # consider to adjust alpha

            train_hist['G_loss'].append(G_loss.item())
            G_loss_D.backward(retain_graph=True)
            G_loss_C.backward()

            G_optimizer.step()

            if (iter % 100 == 0) and iter > 0:
                LOG.info("Epoch: {:3d} || {}/{} || D_loss: {:.4f}, G_loss: {:.4f}, C_loss: {:.4f}".format(
                    epoch, iter, unlabeled_steps,
                    D_loss.item(),
                    G_loss.item(),
                    C_loss.item())
                )
        # evaluate C 
        C.eval()
        average_loss = 0
        correct = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                _, output = C(x_test)
                test_loss = clf_loss(output, y_test).item()  # reduction = 'mean'
                train_hist['test_loss'].append(test_loss)
                pred = torch.argmax(output, dim=1, keepdim=True)  # get the index of the max log-probability
                average_loss += test_loss
                correct += pred.eq(y_test.view_as(pred)).sum().item()
        average_loss /= test_steps

        correct_rate = correct / len(test_loader.dataset)
        cur_time = time.time() - epoch_start_time
        with open(acc_time_file, 'a') as f:
            f.write(str(cur_time) + ' ' + str(correct_rate) + '\n')

        if correct_rate > best_acc:
            best_acc = correct_rate
            best_time = cur_time
            is_best = True
        else:
            is_best = False
        
        # save model C, G, D
        if args.checkpoint_epochs and epoch % args.checkpoint_epochs == 0 and is_best:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': C.state_dict(),
                'best_acc': best_acc,
                'optimizer': C_optimizer.state_dict()},
                checkpoint_path)
            torch.save(D.state_dict(), os.path.join(checkpoint_path, 'D.pkl'))
            torch.save(G.state_dict(), os.path.join(checkpoint_path, 'G.pkl'))

        LOG.info('\nTest set || Test loss: {:.4f} || Accuracy: {}/{} ({:.4f}%) || Best accuraccy: {}\n'.format(
            average_loss, correct, len(test_loader.dataset),
            100. * correct_rate,
            best_acc))
        
        train_hist['test_accuracy'].append(correct_rate)
        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        # visualize generated images
        with torch.no_grad():
            visualize_generated_images(epoch, sample_z_, args)
        
        if args.scheduler:
            C_scheduler.step(average_loss)

    # best accuracy
    with open(acc_time_best_file, 'a') as f:
        f.write(str(best_time) + ',' + str(best_acc) + '\n')

    train_hist['total_time'].append(time.time() - start_time)
    LOG.info("Average one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                    epoch,
                                                                    train_hist['total_time'][0]))

    with open(os.path.join(checkpoint_path, 'history.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)

    # make animation image (gif)
    generate_animation(generated_images_dir, epoch)
    # save loss img
    train_loss_plot(train_hist, checkpoint_path)
    test_loss_plot(train_hist, checkpoint_path)
    # save acc img
    acc_plot(train_hist, checkpoint_path)
    

