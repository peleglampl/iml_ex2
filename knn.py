
import numpy as np
import faiss
import helpers
import matplotlib.pyplot as plt


class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        D, I = self.knn_distance(X)
        neighbors_labels = self.Y_train[I]  # shape: (num_test, k)

        y_pred = []  # collect predictions here

        for i in range(len(neighbors_labels)):
            freq = {}  # count frequency of each label
            for label in neighbors_labels[i]:
                freq[label] = freq.get(label, 0) + 1

            # find label with max frequency
            max_label = max(freq, key=freq.get)
            y_pred.append(max_label)

        return np.array(y_pred)


    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        # the faiss function takes the closest k elements
        D, I = self.index.search(X, self.k)
        return D, I



if __name__ == '__main__':
    # load data:

    # labels as the last column:
    train_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1].astype(np.float32)
    Y_test = test_data[:, -1].astype(int)

    # parameters:
    k_values = [1, 10, 100, 1000, 3000]
    metrics = ['l1', 'l2']

    results = np.zeros((len(k_values), len(metrics)))  # 5x2 accuracy table

    for j, metric in enumerate(metrics):
        for i, k in enumerate(k_values):
            model = KNNClassifier(k, distance_metric=metric)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == Y_test)
            results[i, j] = acc

    # print results as table:
    # Print results as table
    print("\nAccuracy Table (rows = k, columns = metrics):")
    print("           L1        L2")
    for i, k in enumerate(k_values):
        print(f"k={k:<6}  {results[i, 0]:.4f}    {results[i, 1]:.4f}")

    # --- find the best amd worst models:
    best_idx = np.unravel_index(np.argmax(results), results.shape)
    worst_idx = np.unravel_index(np.argmin(results), results.shape)

    best_k = k_values[best_idx[0]]
    worst_k = k_values[worst_idx[0]]

    model_best_1 = KNNClassifier(k=best_k, distance_metric='l2')
    model_best_1.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_best_1, X_test, Y_test)

    model_worst_2 = KNNClassifier(k=worst_k, distance_metric='l2')
    model_worst_2.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_worst_2, X_test, Y_test)

    model_3 = KNNClassifier(k=best_k, distance_metric='l1')
    model_3.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_3, X_test, Y_test)

    # --- Anomaly detection using kNN:
    test_AD_data = np.genfromtxt('AD_test.csv', delimiter=',', skip_header=1)
    X_test_AD = test_AD_data.astype(np.float32)  # take both columns

    # build the model on the train set:
    model = KNNClassifier(k=5, distance_metric='l2')
    model.fit(X_train, Y_train)

    neighbors_distances, neighbors_indices = model.knn_distance(X_test_AD)

    # sum the 5 distances
    anomaly_sum = np.sum(neighbors_distances, axis=1)

    # --- Find 50 most anomalous points ---
    num_anomalies = 50

    # Get indices that sort the scores in descending order
    sorted_indices = np.argsort(anomaly_sum)[::-1]
    anomalies = anomaly_sum[:num_anomalies]
    normal = anomaly_sum[num_anomalies:]

    X_anomalies = X_test_AD[:num_anomalies]
    X_normal = X_test_AD[num_anomalies:]

    plt.figure(figsize=(8,6))
    plt.scatter(X_train[:, 0], X_train[:, -1],
                color='black', alpha=0.01, label='Train data')
    plt.scatter(X_normal[:, 0], X_normal[:, -1],
                color='blue', label='Normal test points')
    plt.scatter(X_anomalies[:, 0], X_anomalies[:, -1],
                color='red', label='Anomaly test points')

    # Labels and legend
    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()




