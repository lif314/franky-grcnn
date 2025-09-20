conda create -n grcnn python==3.10
conda activate grcnn

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# # libfranka 0.15.0
pip install franky-control
