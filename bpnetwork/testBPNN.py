import numpy as np
import NeuronNetwork as nn


if __name__ == '__main__':
    print('Test start.')
    train_data = np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                     [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                     [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                     [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                     [0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    np.set_printoptions(precision=3, suppress=True)

    network = nn.NeuralNetWork([8, 6, 8])
    network.fit(train_data, train_data, learning_rate=0.05, epochs=10000)
    print('Train done.')
    print('\n\n', 'Result:')
    for inner_item in train_data:
        print(inner_item, network.predict(inner_item))

