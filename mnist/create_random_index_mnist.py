def dataloader(args):
    transform = transform_func(args.input_size)

    if args.dataset == 'mnist':
        training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

        indexes = np.arange(len(training_set))
        np.random.shuffle(indexes)

        mask = np.zeros(shape=indexes.shape, dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indexes], dtype = np.int64)

        for i in range(10):
            mask[np.where(labels[indexes] == i)[0][: args.num_labeled // 10]] = True # choosen labeled data
        
        labeled_indexes, unlabeled_indexes = indexes[mask], indexes[~ mask]
        labeled_indexes = list(labeled_indexes)

        labeled_set = get_dataset(labeled_indexes, training_set)
        unlabeled_set = get_dataset(unlabeled_indexes, training_set)
        labeled_set = MyDataset(labeled_set[0], labeled_set[1])
        unlabeled_set = MyDataset(unlabeled_set[0], unlabeled_set[1])

        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False
            )
    
    return labeled_loader, unlabeled_loader, test_loader, labeled_indexes