from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def plot_tsne(features, labels, dataset, file_name, save_dir='results/tnse'):

    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'IEMOCAP':
        label_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    if dataset == 'MELD':
        label_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for i, label_name in enumerate(label_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, s=10, alpha=0.7)

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{file_name}.pdf')
    plt.savefig(save_path, format="pdf")
    plt.close()





