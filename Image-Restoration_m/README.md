Multi-Stage Progressive Image Restoration (MPRNet)

MPRNet is a state-of-the-art model for image restoration tasks, including deblurring, deraining, and denoising. It employs a novel multi-stage architecture that progressively refines degraded images. Key features include:

Synergistic Design: Balances spatial details with high-level contextual information.
Per-Pixel Adaptive Design: Uses supervised attention to reweight local features.
Two-Faceted Information Exchange: Combines sequential and lateral connections for robust feature propagation.

Highlights
Achieves significant performance gains on benchmarks like Rain100L, GoPro, and DND.
Supports tasks such as Deblurring, Deraining, and Denoising with pre-trained models.

Installation
Set up a Python 3.7 environment:
bash
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install the warmup scheduler:
bash
Copy code
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..

Quick Run
Test pre-trained models on custom images:


python demo.py --task <Task_Name> --input_dir <path_to_images> --result_dir <save_images_here>

Example for Deblurring:


python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
Training and Evaluation
Training and testing scripts are available for all tasks in their respective directories.

Results
MPRNet delivers high-quality restored images. Pre-generated results can be downloaded for:

Deblurring
Deraining
Denoising


