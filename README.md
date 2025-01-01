# General Multimodal Embedding for Image Search

This project is developed based on the [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) model and is used for testing image retrieval under arbitrary inputs.
- Paper: [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855)

## How to Use

1.  **Prepare the database** for retrieval, use [build_index.py](build_index.py) for feature extraction and index building.
2.  **run** [retrieval_app.py](retrieval_app.py) for online retrieval.

## Setup

``` bash
# Set Environment
conda create -n gme python=3.10
conda activate gme
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
pip install transformers                               # test with 4.47.1
pip install gradio                                     # test with 5.9.1
```

``` bash
# Get Model
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Alibaba-NLP/gme-Qwen2-VL-2B-Instruct --local-dir gme-Qwen2-VL-2B-Instruct
```


## Results

<details>
  <summary><strong>Image(+Text) -> Image</strong></summary>

  <video src="https://github.com/user-attachments/assets/b92e9782-5873-4f2d-9fe9-4d6aecd2ccfc"></video>
  
</details>
<details>
  <summary><strong>Text -> Image[Chinese input]</strong></summary>

<video src="https://github.com/user-attachments/assets/c8efe5fb-4d0d-46dc-9a17-b1aa0bd88572"></video>

</details>

<details>
  <summary><strong>Text -> Image[English input]</strong></summary>

<video src="https://github.com/user-attachments/assets/492b6aa2-3ba2-4337-8d5e-1ba0ab5b997e"></video>

</details>
<details>
  <summary><strong>Text(long) -> Image</strong></summary>

<video src="https://github.com/user-attachments/assets/49e57772-8846-4cf4-bc28-004337234228"></video>

</details>

