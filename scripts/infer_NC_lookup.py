import sys
sys.path.append('.')

import sys
sys.path.append('.')
import pandas as pd
import os, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import commons
from utils.losses import pairwise_cosine_distance
import torch
import numpy as np
from models.mlp import *
from tqdm import tqdm
from scripts.eval_NC_multilabel import evaluate
import faiss

def load_model(model_dir, device, logger, ckpt_name='best_checkpoints.pt'):
    train_config = commons.load_config(os.path.join(model_dir, 'config.yml'))
    # model = MLPModel(train_config.model)
    model = globals()[train_config.model.model_type](train_config.model)
    ckpt = torch.load(os.path.join(model_dir, 'checkpoints', ckpt_name), map_location='cpu')
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(model_dir, "checkpoints", ckpt_name)}')
    model.to(device)
    model.eval()
    
    return model

def get_pid_embedding_labels(data_file, model, ont, device, logger, tag, save_path=None, model_dir=None):
    data = torch.load(data_file)
    pids, inputs, labels = zip(*[(k, v['embedding'], v[ont]) for k, v in data.items()])
    inputs = torch.vstack(inputs)
    logger.info(f'Loaded {len(inputs)} sequences from {data_file}')
    batch_size = 1000
    num_batches = int(np.ceil(len(inputs) / batch_size))
    embeddings = []
    if tag == 'lookup' and save_path is not None and os.path.exists(os.path.join(model_dir, save_path)):
        embeddings = torch.load(os.path.join(model_dir, save_path))
        logger.info(f'Loaded embeddings from {os.path.join(model_dir, save_path)}')
        return pids, embeddings, labels
    for i in tqdm(range(num_batches), desc='Embedding', dynamic_ncols=True):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size].to(device)
        with torch.no_grad():
            _, batch_embeddings = model(batch_inputs)
        embeddings.append(batch_embeddings.cpu())
    try:
        embeddings = torch.cat(embeddings, dim=0).to(device)
    except:
        embeddings = torch.cat(embeddings, dim=0)
    logger.info(f'Embedding shape: {embeddings.shape}')
    if tag == 'lookup' and save_path is not None:
        torch.save(embeddings, os.path.join(model_dir, save_path))
        logger.info(f'Saved embeddings to {os.path.join(model_dir, save_path)}')
    
    return pids, embeddings, labels

def pairwise_cosine_distance_batch(x, y, batch_size=1024):
    '''
    x: (N, D)
    y: (M, D)
    return: (N, M)
    '''
    N, D = x.size()
    M, _ = y.size()
    
    # Initialize the result tensor
    result = torch.empty((N, M), device=x.device)
    
    # Normalize x and y to have unit norm
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-10)
    y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-10)
    
    # Compute the cosine distance in batches for y
    for j in range(0, M, batch_size):
        y_batch = y_norm[j:j+batch_size]
        result[:, j:j+batch_size] = 1 - torch.mm(x_norm, y_batch.t())
    
    return result

def load_database(lookup_database):
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return(index)


def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return(D, I)

def infer_lookup_query(lookup_emb, lookup_labels, query_emb, query_labels, query_pids, topk, logger):
    # try:
    #     cos_dist = pairwise_cosine_distance(query_emb, lookup_emb)
    # except:
    #     cos_dist = pairwise_cosine_distance_batch(query_emb.cpu(), lookup_emb.cpu())
    # smallest_k_values, smallest_k_indices = commons.n_smallest(cos_dist, topk)
    # normalize lookup_emb and query_emb
    lookup_emb = lookup_emb.cpu()
    query_emb = query_emb.cpu()
    lookup_emb = lookup_emb / (lookup_emb.norm(dim=1, keepdim=True) + 1e-10)
    query_emb = query_emb / (query_emb.norm(dim=1, keepdim=True) + 1e-10)
    lookup_emb = lookup_emb.cpu().numpy()
    query_emb = query_emb.cpu().numpy()
    lookup_database = load_database(lookup_emb)
    D, smallest_k_indices = query(lookup_database, query_emb, topk)
    pred_labels = []
    n_query = len(query_labels)
    for i in tqdm(range(n_query)):
        pred = []
        for j in smallest_k_indices[i]:
            pred.extend(lookup_labels[j])
        pred_labels.append(pred)
    
    pid2pred = {pid: pred for pid, pred in zip(query_pids, pred_labels)}
    pid2gt = {pid: gt for pid, gt in zip(query_pids, query_labels)}
    
    return pid2pred, pid2gt

def save_predictions(pid2pred, pid2gt, save_path):
    pids = list(pid2pred.keys())
    predictions = [';'.join(pid2pred[pid]) for pid in pids]
    ground_truths = [';'.join(pid2gt[pid]) for pid in pids]
    df = pd.DataFrame({'Entry': pids, 'Predictions': predictions, 'Ground Truth': ground_truths})
    df.to_csv(save_path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='configs/infer_NC_lookup.yml')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--lookup_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--ont', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='predictions_lookup.csv')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoints.pt')
    parser.add_argument('--cache_lookup', action='store_true')
    
    args = parser.parse_args()
    
    return args

def main():
    args = get_args()
    config = commons.load_config(args.config)
    config.model_dir = args.model_dir if args.model_dir is not None else config.model_dir
    config.lookup_data = args.lookup_data if args.lookup_data is not None else config.lookup_data
    config.test_data = args.test_data if args.test_data is not None else config.test_data
    config.label_file = args.label_file if args.label_file is not None else config.label_file
    config.topk = args.topk if args.topk is not None else config.topk
    config.ont = args.ont if args.ont is not None else config.ont
    
    logdir = os.path.join(config.model_dir, 'infer_NC_lookup')
    os.makedirs(logdir, exist_ok=True)
    logger = commons.get_logger('infer', logdir)
    
    model = load_model(config.model_dir, args.device, logger, ckpt_name=args.ckpt_name)
    lookup_pids, lookup_emb, lookup_labels = get_pid_embedding_labels(config.lookup_data, model, config.ont, args.device, logger, tag='lookup' if args.cache_lookup else '', save_path='lookup_emb.pt', model_dir=config.model_dir)
    test_pids, test_emb, test_labels = get_pid_embedding_labels(config.test_data, model, config.ont, args.device, logger, tag='test')
    pid2pred, pid2gt = infer_lookup_query(lookup_emb, lookup_labels, test_emb, test_labels, test_pids, config.topk, logger)
    evaluate(pid2pred, pid2gt, config.label_file, logger)
    
    save_predictions(pid2pred, pid2gt, os.path.join(config.model_dir, args.output))
    
    
if __name__ == '__main__':
    main()


