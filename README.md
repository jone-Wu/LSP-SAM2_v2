# LSP-SAM2_v2
A Lightweight Self-Prompting SAM2 Model for Robust Video Object Segmentation.

![image](./figs/Structure.png)


## Highlights


- Problem Addressed:
    - In **challenging** segmentation scenarios, SAM2 still relies on manual prompts or decoupled self-prompting methods that require complex feature extraction, severely limiting its practical applicability.

- Proposed Solution: 
	- **LSP-SAM2**: more efficient self-prompting model for fully automatic VOS, without additional feature extraction.

	- **Two-way Multi-scale box Predictor (TMP)** fuses SAM2’s multi-scale features with temporal sampling, robust to object variations.

	- **End-to-end training** optimizes TMP’s box prompts for segmentation, not just spatial accuracy—unlike conventional decoupled methods.
	- LSP-SAM2: +**12.21%** *J&F* score over SAM2 on hard videos, meanwhile matching its performance on easy videos.

## Visualization Demo:

https://github.com/user-attachments/assets/80a7e0f1-0743-4590-a25a-2fd4f12ed152.mp4

https://github.com/user-attachments/assets/f634d4a1-1964-41a9-9a1e-336c12933bf6.mp4

## Getting Started
### Installation
   LSP-SAM2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`.

You can install it on a GPU machine using:

	git clone https://github.com/jone-Wu/LSP-SAM2_v2.git && cd LSP-SAM2_v2
	pip install -e .
>**Note:** It's recommended to create a new Python environment via Anaconda for this installation.

### Download Checkpoints
- First, we need to download some model checkpoints. 
- You can download these checkpoint files through the following links：

| SAM2 checkpoint | TMP (davis)| TMP (mose)|
| - | - | - |
| [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) | [best_davis_model_epoch625.pth](https://pan.baidu.com/s/1VMsBhOlTrV5j4Mr45avIww) | [best_mose_model_epoch310.pth](https://pan.baidu.com/s/1n_9-805FwdKE8heiw2Zbnw) |
**Access code: lsp2**

**Directory Structure:**

	checkpoints/
	|-- davis/
	|	|-- best_davis_model_epoch625.pth
	|-- mose/
	|  |-- best_mose_model_epoch310.pth
	|-- sam2.1_hiera_base_plus.pt

### Data
- MOSE 2023 ([Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MOSE_A_New_Dataset_for_Video_Object_Segmentation_in_Complex_ICCV_2023_paper.pdf))([Data_avaliable](https://mose.video/)）
	- Since most videos in MOSE are short clips typically containing only around a dozen frames, we construct a subset named **MOSE-long** to enable TMP to learn long-range motion trajectories. 
	- Specifically, we selected **423 videos** longer than 60 frames from the MOSE dataset. Then, we calculated the *J&F* score for each video using the SAM2 model and categorized these videos into three difficulty levels: **easy** (*J&F* >= 90%), **medium** (70% <= *J&F* < 90%), and **hard** (*J&F* < 70%). 
	- To comprehensively evaluate the performance of each self-prompting method on videos of varying difficulty levels, ‌we randomly selected 20 videos from each category to construct a [**validation set**](https://github.com/jone-Wu/LSP-SAM2_v2/blob/main/data/mose_long_val.txt) of 60 videos.  The remaining 363 videos were used as the [**training set**](https://github.com/jone-Wu/LSP-SAM2_v2/blob/main/data/mose_long_train.txt). 

- DAVIS 2017 ([Paper](https://arxiv.org/pdf/1704.00675))([Data_avaliable](https://davischallenge.org/))
	- Our model uses the original settings of the DAVIS 2017 Semi superviced dataset.
	- [**training set**](https://github.com/jone-Wu/LSP-SAM2_v2/blob/main/data/davis_2017_train.txt) & [**validation set**](https://github.com/jone-Wu/LSP-SAM2_v2/blob/main/data/davis_2017_val.txt)

**Directory Structure:**

	LSP-SAM2_v2/DAVIS-2017-trainval-480p/
	|-- DAVIS/
	|	|-- ImageSets/
	|	|	|-- 2017/
	|	|	|	|-- train.txt (./data/davis_2017_train.txt)
	|	|	|	|-- val.txt (./data/davis_2017_val.txt)
	|	|-- Annotations/
	|	|	|-- 480p
	|	|	|	|-- bear(video_name)
	|	|	|	|	|-- 00000.png
	|	|	|	|	|-- ...
	|	|-- JPEGImages/
	|	|	|-- 480p
	|	|	|	|-- bear(video_name)
	|	|	|	|	|-- 00000.jpg
	|	|	|	|	|-- ...

	LSP-SAM2_v2/mose2023/
	|-- train.txt(.data/mose_long_train.txt)
	|-- val.txt(.data/mose_long_val.txt)
	|	|-- train/
	|	|	|-- Annotations/
	|	|	|	|-- ad1881d1(video_name)
	|	|	|	|	|-- 00000.png
	|	|	|	|	|-- ...
	|	|	|-- JPEGImages/
	|	|	|	|-- ad1881d1(video_name)
	|	|	|	|	|-- 00000.jpg
	|	|	|	|	|-- ...

## Evaluation
You can run the `./eval/valid_LSP_SAM2_v6_final.py` according to the following configuration:

	--sam2_cfg
	configs/sam2.1/sam2.1_hiera_b+.yaml
	--sam2_checkpoint
	./checkpoints/sam2.1_hiera_base_plus.pt
	--base_video_dir
	./DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/
	--input_mask_dir
	./DAVIS-2017-trainval-480p/DAVIS/Annotations/480p/
	--val_video_list_file
	./DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/val.txt
	--output_mask_dir
	./eval/output/
	--model_path
	./checkpoints/davis/best_davis_model_epoch625.pth
	--extend_box
	0.0
	--local_rate
	6
	--middle_rate
	4
	--local_weight
	0.2
	--middle_weight
	0.4
	--large_weight
	0.4
	--backtrack_range
	3


## Acknowledgements
- We gratefully acknowledge the publicly available code & datasets from the articles ([SAM2](https://github.com/facebookresearch/sam2), [DAVIS](https://arxiv.org/pdf/1704.00675), [MOSE](https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MOSE_A_New_Dataset_for_Video_Object_Segmentation_in_Complex_ICCV_2023_paper.pdf)) that enabled this work.
