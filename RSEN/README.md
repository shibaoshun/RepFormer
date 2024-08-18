# Network architecture
![](https://github.com/shibaoshun/RepFormer/blob/b2f1353e3b3600a34649f8c0c92c8f5807705665/figs/RSEN.png)
# Training
+ Download the Datasets
+ Train the model with default arguments by running
```
python train.py
```
# Testing
+ Download the model and place it in ./checkpoints/
+ Run and get the rain streak images
```
python test.py
```
## To obtain the rain streak masks from binarization, run
```
python test1.py
```

