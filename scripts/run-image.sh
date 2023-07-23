python experiment.py -m resnet18 -me outliers kurtosis rotated-kurtosis top1 top5 rms -d imagenet $@
python experiment.py -m resnet34 -me outliers kurtosis rotated-kurtosis top1 top5 rms -d imagenet $@
python experiment.py -m resnet50 -me outliers kurtosis rotated-kurtosis top1 top5 rms -d imagenet $@
python experiment.py -m resnet101 -me outliers kurtosis rotated-kurtosis top1 top5 rms -d imagenet $@
python experiment.py -m resnet152 -me outliers kurtosis rotated-kurtosis top1 top5 rms -d imagenet $@