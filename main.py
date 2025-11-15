import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier
import helpers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
# 2. EVALUATE KNN MODELS
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
            accuracy = np.mean(y_pred == Y_test)
            results[i, j] = accuracy

            print(f"{metric}, k={k}  accuracy={accuracy:.4f}")

    return results, k_values, metrics


# ==========================
# 3. PLOT BEST & WORST MODELS
# ==========================
def plot_knn_best_worst(results, k_values, X_train, Y_train, X_test, Y_test):
    # Use L2 column
    best_idx = np.argmax(results[:, 1])
    worst_idx = np.argmin(results[:, 1])

    best_k = k_values[best_idx]
    worst_k = k_values[worst_idx]

    print("\n--- Plotting Best and Worst Models ---")
    print(f"Best L2 model: k={best_k}")
    print(f"Worst L2 model: k={worst_k}")

    # BEST L2
    model_best_l2 = KNNClassifier(best_k, distance_metric='l2')
    model_best_l2.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_best_l2, X_test, Y_test)

    # WORST L2
    model_worst_l2 = KNNClassifier(worst_k, distance_metric='l2')
    model_worst_l2.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_worst_l2, X_test, Y_test)

    # BEST L1 (same k as best L2)
    model_best_l1 = KNNClassifier(best_k, distance_metric='l1')
    model_best_l1.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(model_best_l1, X_test, Y_test)


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
    plt.scatter(X_train[:, 0], X_train[:, 1], color='black', alpha=0.01, label="Train Data")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], color='blue', label="Normal Points")
    plt.scatter(X_anom[:, 0], X_anom[:, 1], color='red', label="Anomalies")

    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    # plt.show()


# ==========================
# 5. DECISION TREES (24 MODELS)
# ==========================
def decision_trees():
    X_train_full, Y_train_full, X_test, Y_test = load_data()

    # Validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=0
    )

    max_depths = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes_list = [50, 100, 1000]

    models = []

    print("\n=== Training 24 Decision Trees ===")
    for d in max_depths:
        for leaf in max_leaf_nodes_list:
            tree = DecisionTreeClassifier(
                max_depth=d,
                max_leaf_nodes=leaf,
                random_state=0
            )
            tree.fit(X_train, Y_train)

            train_acc = np.mean(tree.predict(X_train) == Y_train)
            val_acc = np.mean(tree.predict(X_val) == Y_val)
            test_acc = np.mean(tree.predict(X_test) == Y_test)

            print(f"depth={d}, leaf={leaf} | train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")

            models.append({
                "depth": d,
                "leaf": leaf,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "model": tree
            })

    return models

##################
### QUESTION 7 ###
##################


def random_forest(X_train, Y_train, X_test, Y_test):
    max_depth = 6
    random_forest = RandomForestClassifier(n_estimators=300, max_depth=max_depth, random_state=0)
    random_forest.fit(X_train, Y_train)

    # evaluate accuracy:
    test_acc = np.mean(random_forest.predict(X_test) == Y_test)
    print(f"Random Forest with depth={max_depth} | test={test_acc:.4f}")
    helpers.plot_decision_boundaries(random_forest, X_test, Y_test)


def plotting_decision_boundaries(model, X_test, Y_test, X_train, Y_train):
    max_depth, max_leaf_nodes = model
    tree = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=0)
    tree.fit(X_train, Y_train)
    helpers.plot_decision_boundaries(tree, X_test, Y_test)



def main():
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    # Evaluate ALL 10 kNN models
    results, k_values, metrics = evaluate_knn_models(X_train, Y_train, X_test, Y_test)

    # Plot best/worst boundaries
    # plot_knn_best_worst(results, k_values, X_train, Y_train, X_test, Y_test)
    #
    # # Run anomaly detection on AD_test.csv
    # run_anomaly_detection(X_train)

    # Train 24 decision trees
    decision_trees()

    # Ques 4:
    # plotting_decision_boundaries((20, 1000), X_test, Y_test, X_train, Y_train)

    # # Ques 5:
    # plotting_decision_boundaries((10, 50), X_test, Y_test, X_train, Y_train)
    #
    # # Ques 6:
    # plotting_decision_boundaries((6, 1000), X_test, Y_test, X_train, Y_train)

    # Ques 7:
    random_forest(X_train, Y_train, X_test, Y_test)

    plt.show()


if __name__ == "__main__":
    main()
