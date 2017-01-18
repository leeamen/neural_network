import numpy as np
import NeuronNetwork as nn
import os.path
import pandas as pd

if __name__ == '__main__':
    data_folder = 'D:\\PRML\\HousePrice\\data'
    path_test = os.path.join(data_folder, 'test.csv')
    test_data = pd.read_csv(path_test)

    X_train = np.load(os.path.join(data_folder, 'features_train.npy'))
    X_test = np.load(os.path.join(data_folder, 'features_test.npy'))
    y = np.load(os.path.join(data_folder, 'outcome.npy'))
    network = nn.NeuralNetWork([288, 144, 1])
    network.fit(X_train, y, learning_rate=0.05, epochs=100000)
    print('train done.')
    prediction = np.array([])
    for item in X_test:
        prediction = np.row_stack((prediction, network.predict(item)))
    print('predict done.')
    solution = pd.DataFrame({"SalePrice": np.expm1(prediction), 'id': test_data.Id})

    col = solution.columns.tolist()
    col = col[-1:] + col[:-1]
    solution[col].to_csv("prediction.csv", index=False)

