# ------------------------------------------------------------------------
# GME Search
# Copyright (c) 2025 Wei Li. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

import faiss
import argparse
import numpy as np
import gradio as gr
from gme_model import GmeQwen2VL

def load_model_and_index(args):
    model = GmeQwen2VL(args.model_path)
    
    with open(args.image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    image_embeddings = np.load(args.image_embeddings_file)
    
    index = faiss.read_index(args.faiss_index_file)
    
    return model, image_paths, image_embeddings, index

def get_query_embedding(model, text=None, image=None):
    if text and image:
        embeddings = model.get_fused_embeddings(texts=[text], images=[image])
    elif text:
        embeddings = model.get_text_embeddings(texts=[text])
    elif image:
        embeddings = model.get_image_embeddings(images=[image])
    else:
        return None
    return embeddings.cpu().numpy().astype('float32')

def search_faiss(index, query_embedding, topk):
    similarities, indices = index.search(query_embedding, topk)
    return similarities, indices

def reset_interface(default_topk):
    return "", None, default_topk, []

def process_input(text, image, topk, model, image_paths, index):
    if not text and not image:
        gr.Warning('Please provide either text or image, or both.')
        return []

    query_embedding = get_query_embedding(model, text, image)
    if query_embedding is None:
        gr.Warning('Invalid input.')
        return []

    similarities, indices = search_faiss(index, query_embedding, topk)
    topk_image_paths = [image_paths[i] for i in indices[0]]
    topk_similarities = similarities[0].tolist()
    
    # Format result for display
    return [(topk_image_paths[i], f"Similarity: {topk_similarities[i]:.3f}") for i in range(topk)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GME Search Demo')
    parser.add_argument('--model_path', type=str, default='./models/gme-Qwen2-VL-2B-Instruct',
                        help='Path to the GME model.')
    parser.add_argument('--image_embeddings_file', type=str, default='data_img_ebd.npy',
                        help='Path to the image embeddings file.')
    parser.add_argument('--faiss_index_file', type=str, default='data_faiss_idx.index',
                        help='Path to the FAISS index file.')
    parser.add_argument('--image_paths_file', type=str, default='data_img_path.txt',
                        help='Path to the file containing image paths.')
    args = parser.parse_args()
    
    model, image_paths, image_embeddings, index = load_model_and_index(args)

    with gr.Blocks() as demo:
        gr.Markdown("# General Multimodal Embedding (GME) Search Demo")
        gr.Markdown("""
        <img src="https://user-images.githubusercontent.com/7837172/44953557-0fb54e80-aec9-11e8-9d38-2388bc70c5c5.png" width="13%" style="float: right;" />

        - This project is developed based on the [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) model and is used for testing image retrieval under arbitrary inputs.
        - Paper: [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855)
        - Usage:
            1. **Prepare the database** for retrieval, use `build_index.py` for feature extraction and index building.
            2. **Enter a query** in the "Input Text" field or upload an image in the "Input Image" field, or use both.
                - If both **text(T)** and **image(I)** are provided, the model will perform **multimodal retrieval(T+I -> I)**.
                - Click "Reset" to clear all inputs and reset to default settings.
                - Use the "Select top-k" slider to choose how many top results to retrieve.
            3. **Click "Search"** to perform retrieval based on your input(s).
                - The results will be displayed in the "Top-k Results" gallery.
        """)

        default_topk = 10
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Input Text", placeholder="Enter the query...")
                topk_input = gr.Slider(minimum=1, maximum=50, step=1, label="Select top-k", value=default_topk)
                with gr.Row():
                    submit_btn = gr.Button("Search", variant="primary")
                    reset_btn = gr.Button("Reset")
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Input Image")
        
        output_gallery = gr.Gallery(label="Top-k Results", columns=5, object_fit="contain")
        
        reset_btn.click(fn=lambda: reset_interface(default_topk),
                        inputs=None,
                        outputs=[text_input, image_input, topk_input, output_gallery])
        
        submit_btn.click(fn=lambda text, image, topk: process_input(text, image, topk, model, image_paths, index),
                         inputs=[text_input, image_input, topk_input],
                         outputs=[output_gallery])

    demo.launch(server_port=12306)