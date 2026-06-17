from Data import load_data
import matplotlib.pyplot as plt



def plot_data(path):
    X_data, _ = load_data(path)
    plt.imshow(X_data, aspect='auto', origin='lower')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time Channels')
    plt.ylabel('Frequency Channels')
    plt.title('Preprocessed Pulsar Data')
    plt.show()


# Edit the path here
plot_data("EXAMPLE/PATH/PULSAR.T")