# Distillation-Based Training for Multi-Exit Architectures ([Mary Phuong](https://mary-phuong.github.io), [Christoph H. Lampert](http://pub.ist.ac.at/~chl/), ICCV 2019)

We present a new method for training *multi-exit architectures*.
A multi-exit architecture looks like this:

![Multi-exit architecture](https://github.com/mary-phuong/multiexit-distillation/blob/master/images/multiexit_architecture.png)

We propose to train such architectures by transferring knowledge from late exits (<img src="https://github.com/mary-phuong/multiexit-distillation/blob/master/images/ynhat.png" height="15">) to early exits (<img src="https://github.com/mary-phuong/multiexit-distillation/blob/master/images/y1hat.png" height="15">, <img src="https://github.com/mary-phuong/multiexit-distillation/blob/master/images/y2hat.png" height="15">, ...), via so-called *distillation*.

Read more [here](https://mary-phuong.github.io/multiexit_distillation.pdf).
