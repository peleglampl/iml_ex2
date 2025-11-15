import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ==========================
# 1. DATA LOADING
# ==========================
def load_data():
    train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

    X_train = train[:, :-1].astype(np.float32)
    Y_train = train[:, -1].astype(int)
    X_test = test[:, :-1].astype(np.float32)
    Y_test = test[:, -1].astype(int)

    return X_train, Y_train, X_test, Y_test


# ==========================
# 2. EVALUATION OF kNN MODELS
# ==========================
def evaluate_knn_models(X_train, Y_train, X_test, Y_test):
    k_values = [1, 10, 100, 1000, 3000]
    metrics = ['l1', 'l2']
    results = np.zeros((len(k_values), len(metrics)))

    for j, metric in enumerate(metrics):
        for i, k in enumerate(k_values):
            model = KNNClassifier(k, distance_metric=metric)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            results[i, j] = np.mean(y_pred == Y_test)
            print(metric, k, results[i, j])

    return results, k_values, metrics


# ==========================
# 3. DECISION TREES (24 models)
# ==========================
def decision_trees():
    X_train_full, Y_train_full, X_test, Y_test = load_data()

    # split for validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=0
    )

    max_depth = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes = [50, 100, 1000]

    models = []

    for d in max_depth:
        for l in max_leaf_nodes:
            tree = DecisionTreeClassifier(max_depth=d, max_leaf_nodes=l, random_state=0)
            tree.fit(X_train, Y_train)

            train_acc = np.mean(tree.predict(X_train) == Y_train)
            val_acc   = np.mean(tree.predict(X_val)   == Y_val)
            test_acc  = np.mean(tree.predict(X_test)  == Y_test)

            models.append({
                'model': tree,
                'depth': d,
                'leaf': l,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            })

            print(f"depth={d} leaf={l} | train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")

    return models


# ==========================
# 4. ANOMALY DETECTION
# ==========================
def run_anomaly_detection(X_train):
    test_AD = np.genfromtxt('AD_test.csv', delimiter=',', skip_header=1)
    X_test_AD = test_AD.astype(np.float32)

    model = KNNClassifier(k=5, distance_metric='l2')
    model.fit(X_train, np.zeros(len(X_train)))

    distances, _ = model.knn_distance(X_test_AD)
    anomaly_scores = distances.sum(axis=1)

    sorted_idx = np.argsort(anomaly_scores)[::-1]
    anomaly_idx = sorted_idx[:50]
    normal_idx = sorted_idx[50:]

    X_anom = X_test_AD[anomaly_idx]
    X_norm = X_test_AD[normal_idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], color='black', alpha=0.01)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], color='blue')
    plt.scatter(X_anom[:, 0], X_anom[:, 1], color='red')
    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# ==========================
# MAIN
# ==========================
def main():
    X_train, Y_train, X_test, Y_test = load_data()

    # === PART 1: KNN evaluation ===
    results, k_values, metrics = evaluate_knn_models(X_train, Y_train, X_test, Y_test)
    print("\nResults table:\n", results)

    # === PART 2: Decision Trees (FAST) ===
    models = decision_trees()

    # === PART 3: Anomaly Detection ===
    run_anomaly_detection(X_train)


if __name__ == "__main__":
    main()


def decision_tree_demo():
    # Create random data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Feature matrix with 100 samples and 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels based on a simple condition

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = tree_classifier.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")


def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)


def loading_xgboost():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)


def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def knn_examples(X_train, Y_train, X_test, Y_test):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k=5, distance_metric='l2')

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names


if __name__ == '__main__':
    decision_tree_demo()
