# 运行命令

## 训练代码

swin_base_rf 
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py \
--cfg configs/swin/swin_base_rf.yaml --data-path /disk/datasets/rf_data/train_data/time_frequency --batch-size 32
```
swin_small_rf 
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py \
--cfg configs/swin/swin_small_rf.yaml --data-path /disk/datasets/rf_data/train_data/time_frequency --batch-size 32
```

## 测试代码

swin_base_rf 
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 1245 main.py --eval \
--cfg configs/swin/swin_base_rf.yaml --resume output/swin_base_rf/minmax/ckpt_epoch_52.pth --data-path /disk/datasets/rf_data/train_data/time_frequency
```

swin_small_rf  
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 1235 main.py --eval \
--cfg configs/swin/swin_small_rf.yaml --resume output/swin_small_rf/minmax/ckpt_epoch_52.pth --data-path /disk/datasets/rf_data/train_data/time_frequency
```


python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py \
--cfg configs/swin/swin_small_rf_adjust.yaml --data-path /disk/datasets/rf_data/train_data/time_frequency --batch-size 64