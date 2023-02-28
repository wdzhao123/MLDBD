# MLDBD in PyTorch

Implementation of "IEEE Transactions on Multimedia: Full-scene Defocus Blur Detection with DeFBD+ via Multi-Level Distillation Learning" in PyTorch.

# Datasets DeFBD+
* `train_data`:
   * `source`: Contains 1924 training images.
   * `gt`: Contains 1924 ground truth images corresponding to source images.
* `test_data`:
   * `CUHK+`: Contains 160 testing images and it's GT.
   * `DUT+`: Contains 800 testing images and it's GT.
   
Download and unzip datasets from 

# Test
You can use the following command to test：

>python test.py

You can use the following model to output results directly.Here is our parameters:
baidu link: https://pan.baidu.com/s/1gXcObnd8Ya0i4tNR7JmO-A?pwd=qumx password: qumx

Put "checkpoint.pth" in "./saved_models".

# Eval
If you want to use Fmax and MAE to evaluate the results, you can run the following code in MATLAB. It shows the PR curve and F-measure curve at the same time.

>./evaluate_dbd/evaluate.m
