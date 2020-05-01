"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log, type):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        accuracy = []
        cnn_benchmark = []  # this is ridiculous
        if type == 'lstm':
            for epoch,acc,loss,val_accuracy,val_loss in reader:
                accuracies.append(float(val_accuracy))
                # top_5_accuracies.append(float(val_top_k_categorical_accuracy))
                accuracy.append(float(acc))
        else:
            for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
                accuracies.append(float(val_acc))
                # top_5_accuracies.append(float(val_top_k_categorical_accuracy))
                accuracy.append(float(acc))
        plt.plot(accuracy)
        plt.plot(accuracies)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.plot(cnn_benchmark)
        plt.show()

if __name__ == '__main__':
    # training_log = 'data/logs/cnn-training-1586816403.0097952.log'
    # training_log ="data/logs/conv_3d-training-1586839810.2295735.log"
    # main(training_log, 'cnn')
    training_log = 'data/logs/cnn-training-1588144072.2978652.log'
    main(training_log, 'lstm')

