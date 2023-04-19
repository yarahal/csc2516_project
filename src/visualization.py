import matplotlib.pyplot as plt

def plot_learning_curves(loss_log):
    plt.figure
    plt.plot(loss_log['train'],label='train_loss')
    plt.plot(loss_log['val'],label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss');