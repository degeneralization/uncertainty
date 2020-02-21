wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
gunzip -c train.tar.gz | tar xopf -
cd data
wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
gunzip -c notMNIST_small.tar.gz | tar xopf -
python make_data.py
