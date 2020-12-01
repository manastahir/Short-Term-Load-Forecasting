import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os 
import numpy as np
import datetime as dt
from matplotlib.dates import DateFormatter

def mean_absolute_percentage_error(y_true, y_pred, epsilon = 10e-10):
    y_true, y_pred = np.array(y_true), y_pred[:, 0]
    return np.mean(np.abs((y_true - y_pred) / (y_true+epsilon))) * 100

def train(model, train_data, test_data, weights_file, plots_file, doTrain = True, verbose = 1):    
    x_train, y_train = train_data
    x_test, y_test = test_data

    if(doTrain):
        #callbacks
        checkpoint = ModelCheckpoint(weights_file, monitor='val_mean_absolute_percentage_error', verbose = verbose, save_best_only=True, mode='min')
        lr_scheduler = ReduceLROnPlateau('val_loss', patience=10, verbose = verbose)
        
        # Train
        History = model.fit(x_train, y_train, batch_size=32,
                            epochs = 100, validation_split=0.2,
                            verbose = verbose, callbacks=[checkpoint, lr_scheduler])

        # Get training and test val history
        training_loss = History.history['loss']
        val_loss = History.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.clf()
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, val_loss, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(plots_file)

    #TO make the evaluation on the est weights saved
    model.load_weights(weights_file)

    predicted = model.predict(x_test)
    actual = y_test
    
    mape = mean_absolute_percentage_error(actual, predicted)

    return mape


def create_graphs(data, labels, graph_data, dates, graph_file):
    formatter = DateFormatter('%Y-%m-%d')
    fig, ax = plt.subplots(figsize=(18,8))

    for i in range(len(data)):
      ax.plot(dates, data[i], graph_data[i][0], color=graph_data[i][1], linestyle=graph_data[i][2])

    plt.ylabel ('KW per 30 minutes', size = 20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)

    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

    for label in plt.gcf().axes[0].xaxis.get_ticklabels()[::2]:
      label.set_visible(False)
    
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.13),
            ncol=3, fancybox=True, shadow=True, fontsize=18)
    
    plt.savefig(graph_file, format='pdf', bbox_inches='tight')
    plt.show()
