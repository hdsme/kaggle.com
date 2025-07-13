import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_loss(unique_id, loss_name):
    with open(f'loss_histories_{unique_id}.pkl', 'rb') as f:
        loss_history = pickle.load(f)

    train_losses = loss_history['train_loss']
    val_losses = loss_history['val_loss']

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss ({loss_name})')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'loss_plot_{unique_id}.png')
    plt.show()