import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_curve():
    labels = {
        'Base': 'c3d_gru.csv',
        'Base+CA': 'c3d_ca_gru.csv',
        'Base+STA': 'c3d_sta_gru.csv',
        'Base+SATA': 'c3d_sata_gru.csv',
        'Base+CA+STA': 'c3d_ca_sta_gru.csv',
        'Base+CA+SATA': 'c3d_ca_sata_gru.csv',
        'Base+CA+STA+SATA': 'c3d_ca_sta_sata_gru.csv'
    }
    for i, (key, value) in enumerate(labels.items()):
        df = pd.read_csv(os.path.join('D:/xyc/workspace/csv/', value))
        cer = df.loc[:, ['cer']].values
        plt.plot(cer, label='({}) {}'.format(i+1, key), marker='.')
        
    plt.grid(linestyle='-.')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('character error rate')
    plt.savefig('fig2.pdf')

plot_curve()