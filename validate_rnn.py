"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data_processor import DataSet
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn import svm, linear_model
from sklearn.model_selection import GridSearchCV


def validate(data_type, model, seq_length=50, saved_model=None,
             class_limit=None, image_shape=None, train_test='test'):
    # batch_size = 32
    batch_size = 1


    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # _, test = data.split_train_test()
    # size = len(test)
    # val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    X, y = data.get_data_train_test(data_type, train_test)
    size = len(X)

    # Evaluate!
    # results = rm.model.evaluate_generator(
    #     generator=val_generator,
    #     val_samples=3200)
    #
    # print(results)
    # print(rm.model.metrics_names)

    # results = rm.model.predict_generator(
    #     generator=val_generator,
    #     val_samples=size,
    #     # val_samples=3200,
    #     verbose=1)

    results = rm.model.predict(
        X,
        # val_samples=size,
        # val_samples=3200,
        verbose=1)

    print(results.shape)

    return (results, y)
    # print(results)
    # print(rm.model.metrics_names)

def main(train_test):

    model = 'conv_flow_3d'
    saved_model = 'data/checkpoints/conv_flow_3d-flow.035-0.715.hdf5'

    if model == 'conv_flow_3d' or model == 'lrcn':
        data_type = 'flow'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None


    conv_flow_3d_results, y_conv_flow = validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=4, train_test=train_test)

    model = 'conv_3d'
    saved_model = 'data/checkpoints/conv_3d-images.018-0.616.hdf5'

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None


    conv3d_results, y_conv = validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=4, train_test=train_test)

    model = 'lstm'
    saved_model = 'data/checkpoints/lstm-features.004-0.614.hdf5'
    # saved_model = 'data/checkpoints/lstm-features.017-0.867.hdf5'

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    lstm_results, y_lstm = validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=4, train_test=train_test)




    results = []
    combined_result = []
    for i in range(len(lstm_results)):
        results.append([lstm_results[i][0],lstm_results[i][1], conv3d_results[i][0], conv3d_results[i][1], conv_flow_3d_results[i][0], conv_flow_3d_results[i][1]])
        combined_result.append([lstm_results[i][0]+conv3d_results[i][0]+conv_flow_3d_results[i][0], 3*lstm_results[i][1]+conv3d_results[i][1]+conv_flow_3d_results[i][1]])
        # combined_result = np.argmax(lstm_results[i][0]+conv3d_results[i][0], lstm_results[i][1]+conv3d_results[i][1])
        # Print f1, precision, and recall scores
    np.save('output_' + train_test + '.npy', results)
    np.save('output_labels_' + train_test + '.npy' , y_lstm)

    print("Conv Flow 3d")
    print(precision_score(y_lstm, np.argmax(conv_flow_3d_results, axis=1), average="macro"))
    print(recall_score(y_lstm, np.argmax(conv_flow_3d_results, axis=1), average="macro"))
    print(f1_score(y_lstm, np.argmax(conv_flow_3d_results, axis=1), average="macro"))

    print("LSTM")
    print(precision_score(y_lstm, np.argmax(lstm_results, axis=1) , average="macro"))
    print(recall_score(y_lstm, np.argmax(lstm_results, axis=1) , average="macro"))
    print(f1_score(y_lstm, np.argmax(lstm_results, axis=1) , average="macro"))

    print("Conv 3d")
    print(precision_score(y_lstm, np.argmax(conv3d_results, axis=1), average="macro"))
    print(recall_score(y_lstm, np.argmax(conv3d_results, axis=1), average="macro"))
    print(f1_score(y_lstm, np.argmax(conv3d_results, axis=1), average="macro"))

    print("Combined")
    print(precision_score(y_lstm, np.argmax(combined_result, axis=1), average="macro"))
    print(recall_score(y_lstm, np.argmax(combined_result, axis=1), average="macro"))
    print(f1_score(y_lstm, np.argmax(combined_result, axis=1), average="macro"))

        # results.append((lstm_results[i][0]+conv3d_results[i][0], lstm_results[i][1]+conv3d_results[i][1]))
        # print(str(lstm_results[i]),str(conv3d_results[i]))
        # print(str(y_conv[i]),str(y_lstm[i]))


    # y_pred = np.argmax(y_pred1, axis=1)
    #
    # # Print f1, precision, and recall scores
    # print(precision_score(y_test, y_pred , average="macro"))
    # print(recall_score(y_test, y_pred , average="macro"))
    # print(f1_score(y_test, y_pred , average="macro"))

    return results, y_lstm

if __name__ == '__main__':
    results, labels = main('train')
    results_test, labels_test = main('test')
    # svc = svm.SVC(kernel='rbf', C=1, gamma='auto')
    parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 1, 10]}
    svc = svm.SVC()
    # Create regularization penalty space
    # penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    # C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    # parameters = dict(C=C, penalty=penalty)
    # linear = linear_model.LogisticRegression()
    clf = GridSearchCV(svc, parameters)
    # clf.fit(iris.data, iris.target)
    clf.fit(results, labels)
    print("Accuracy for train: ", clf.score(results, labels))
    print("Accuracy for test: ", clf.score(results_test, labels_test))
