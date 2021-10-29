## Cut-Thumbnail(Accepted at ACM MULTIMEDIA 2021)
**[Tianshu Xie](mailto:tianshuxie@std.uestc.edu.cn), [Xuan Cheng](mailto:cs_xuancheng@uestc.edu.cn), Xiaomin Wang, 
Minghui Liu, Jiali Deng, Tao Zhou, Ming Liu**
---------------------
This is the official Pytorch implementation of Cut-Thumbnail in the paper [Cut-Thumbnail:A Novel Data Augmentation for Convolutional 
Neural Network](https://doi.org/10.1145/3474085.3475302).

This implementation is based on these repositories:
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
- [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch/)
- [Pytorch-classification](https://github.com/bearpaw/pytorch-classification/)

### Main Requirements
- torch == 1.0.1
- torch == 0.2.0
- Python 3

## Training Examples
- Mixed Single Thumbnail
```
python train.py -d [datasetlocation] --depth 50 --mode mst --size 112 --lam 0.25 --participation_rate 0.8
```
- Single Thumbnail
```
python train.py -d [datasetlocation] --depth 50 --mode st --size 112 --lam 0.25 --participation_rate 0.8
```

## Citation
If you find our paper and this repo useful, please cite as
```
@inproceedings{xie20cut-thumbnail,
    author = {Xie, Tianshu and Cheng, Xuan and Wang, Xiaomin and Liu, Minghui and Deng, Jiali and Zhou, Tao and Liu, Ming},
    title = {Cut-Thumbnail: A Novel Data Augmentation for Convolutional Neural Network},
    year = {2021},
    isbn = {9781450386517},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3474085.3475302},
    doi = {10.1145/3474085.3475302},
    booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
    pages = {1627–1635},
    numpages = {9},
    location = {Virtual Event, China},
    series = {MM '21}
}
```