import torch
import argparse
import os
from model import DiGemo
from thop import profile


def calculte_complextity(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = checkpoint['args']
    
    if args.dataset == "IEMOCAP":
        embedding_dims = [1024, 342, 1582]
        n_classes_emo = 6

    if args.dataset == "MELD":
        embedding_dims = [1024, 342, 300]
        n_classes_emo = 7

    model = DiGemo(args, embedding_dims, n_classes_emo).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    seq_len = 50
    batch_size = 1

    t_feat = torch.randn(seq_len, batch_size, embedding_dims[0]).to(device)
    v_feat = torch.randn(seq_len, batch_size, embedding_dims[1]).to(device)
    a_feat = torch.randn(seq_len, batch_size, embedding_dims[2]).to(device)


    umask = torch.ones(seq_len, batch_size).to(device)
    n_speakers = 2 if n_classes_emo == 6 else 9
    qmask = torch.zeros(seq_len, batch_size, n_speakers).to(device)
    qmask[:, :, 0] = 1 # all speaker 1
    dia_lengths = [seq_len] * batch_size

    inputs = (t_feat, v_feat, a_feat, umask, qmask, dia_lengths)
    
    flops, params = profile(model, inputs=inputs, verbose=True)

    print(f"Model params: {params / 1e6:.3f} M")
    print(f"Flops: {flops / 1e9:.3f} GFLOPs")


if __name__ == "__main__":
    path = "./checkpoints/best_model_IEMOCAP_9161.pth"
    calculte_complextity(path)    

    
