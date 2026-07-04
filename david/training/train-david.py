

if __name__ == "__main__":



    for epoch in range(args.epochs):                                        # pass over the data N times
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   # fresh shuffled batches
    train_epoch(epoch, loader, len(loader))                             # run the inner loop