fpath = 'datasets/CROHME/train_labels.txt'
out_fpath = 'datasets/CROHME/half_train_labels.txt'

with open(fpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
labels = labels[::2]
with open(out_fpath, 'w') as f:
    f.write('\n'.join(labels)+'\n')