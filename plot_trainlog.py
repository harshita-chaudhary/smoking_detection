"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        val_accuracies = []
        accuracies = []
        val_losses = []
        losses = []
        cnn_benchmark = []  # this is ridiculous

        for epoch,acc,loss,val_accuracy,val_loss in reader:
            val_accuracies.append(float(val_accuracy))
            accuracies.append(float(acc))
            # val_losses.append(float(val_loss))
            # losses.append(float(loss))


        plt.plot(accuracies)
        plt.plot(val_accuracies)
        # plt.plot(losses)
        # plt.plot(val_losses)
        plt.title('model accuracy and loss')
        # plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'test_acc'], loc='upper left')
        # plt.plot(cnn_benchmark)
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/cnn-training-1587866460.33326.log'
    # training_log ="data/logs/conv_3d-training-1586839810.2295735.log"
    # main(training_log, 'cnn')
    # training_log = 'data/logs/lstm-training-1588036406.0306637.log'
    main(training_log)

