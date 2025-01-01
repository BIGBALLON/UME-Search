# ------------------------------------------------------------------------
# GME Search
# Copyright (c) 2025 Wei Li. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

import glob
import torch
import faiss
import argparse
import numpy as np
from gme_model import GmeQwen2VL

def extract_image_embeddings(vlm, image_paths, batch_size=2):
    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_images = image_paths[i:i + batch_size]
        try:
            batch_embeddings = vlm.get_image_embeddings(images=batch_images)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])

def load_existing_data(embeddings_file, index_file, image_paths_file):
    try:
        embeddings = np.load(embeddings_file)
        index = faiss.read_index(index_file)
        with open(image_paths_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return embeddings, index, image_paths
    except Exception:
        return None, None, []

def save_updated_data(embeddings, index, image_paths, embeddings_file, index_file, image_paths_file):
    np.save(embeddings_file, embeddings)
    faiss.write_index(index, index_file)
    with open(image_paths_file, 'w') as f:
        for img_path in image_paths:
            f.write(img_path + '\n')

def main(args):
    gme = GmeQwen2VL(args.model_path)

    existing_embeddings, existing_index, existing_image_paths = load_existing_data(
        args.embeddings_output, args.index_output, args.image_paths_output
    )

    # Extract embeddings for new images if needed
    new_image_paths = glob.glob(f"{args.image_dir}/**/*", recursive=True)
    new_image_paths = [img for img in new_image_paths if img not in existing_image_paths]
    if not new_image_paths:
        print("No new images to index.")
        return

    new_image_embeddings = extract_image_embeddings(gme, new_image_paths, batch_size=args.batch_size)
    if new_image_embeddings.nelement() == 0:
        print("No embeddings extracted for new images.")
        return

    # Convert embeddings to float32 numpy array
    new_image_embeddings_np = new_image_embeddings.cpu().numpy().astype('float32')

    # Update FAISS index and embeddings
    if existing_index is not None:
        existing_index.add(new_image_embeddings_np)
        updated_embeddings = np.vstack([existing_embeddings, new_image_embeddings_np])
        updated_image_paths = existing_image_paths + new_image_paths
    else:
        # If no existing index, create a new one
        dimension = new_image_embeddings_np.shape[1]
        existing_index = faiss.IndexFlatIP(dimension)
        existing_index.add(new_image_embeddings_np)
        updated_embeddings = new_image_embeddings_np
        updated_image_paths = new_image_paths

    # Save updated data
    save_updated_data(
        updated_embeddings, existing_index, updated_image_paths,
        args.embeddings_output, args.index_output, args.image_paths_output
    )
    print(f"Index updated with {len(new_image_paths)} new images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Incrementally update FAISS index for image embeddings.')
    parser.add_argument('--model_path', type=str, default='./models/gme-Qwen2-VL-2B-Instruct',
                        help='Path to the GmeQwen2VL model.')
    parser.add_argument('--image_dir', type=str, default='./gallery',
                        help='Path to the directory containing new images.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for embedding extraction.')
    parser.add_argument('--embeddings_output', type=str, default='data_img_ebd.npy',
                        help='Output file for saving image embeddings.')
    parser.add_argument('--index_output', type=str, default='data_faiss_idx.index',
                        help='Output file for saving FAISS index.')
    parser.add_argument('--image_paths_output', type=str, default='data_img_path.txt',
                        help='Output file for saving image paths.')
    args = parser.parse_args()
    main(args)