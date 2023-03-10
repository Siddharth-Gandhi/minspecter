python scripts/pytorch_lightning_training_script/train.py --save_dir data/output --gpus 1 --train_file data/preprocessed/data-train.p --dev_file data/preprocessed/data-val.p --test_file data/preprocessed/data-test.p --batch_size 2 --num_workers 8 --num_epochs 8 --grad_accum 256

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

python scripts/pytorch_lightning_training_script/train.py --save_dir output --gpus 1 --train_file data/preprocessed/data-train.p --dev_file data/preprocessed/data-val.p --test_file data/preprocessed/data-test.p --batch_size 8 --num_workers 8 --num_epochs 2 --grad_accum 256 --checkpoint_path output/model_checkpoint

1. conda create --name minspecter python=3.7 setuptools
2. pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
3. pip install -r requirements.txt
