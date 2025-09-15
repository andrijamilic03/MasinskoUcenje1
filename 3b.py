# TODO popuniti kodom za problem 3b

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Implementacija k-NN klasifikatora
class KNN:
    def __init__(self, nb_features, nb_classes, data, k):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.k = k
        self.X = tf.convert_to_tensor(data['x'], dtype=tf.float32)
        self.Y = tf.convert_to_tensor(data['y'], dtype=tf.int32)

    def predict(self, query_data):
        nb_queries = len(query_data['x'])
        predictions = []

        for i in range(nb_queries):
            dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, query_data['x'][i])), axis=1))
            _, idxs = tf.nn.top_k(-dists, self.k)
            classes = tf.gather(self.Y, idxs)
            hyp = tf.argmax(np.bincount(classes.numpy()))
            predictions.append(hyp)

        return np.array(predictions)

# Učitavanje iris skupa podataka
data = np.loadtxt('data/iris.csv', delimiter=',', dtype=str, skiprows=1)
X = data[:, :2].astype(float)  # Samo prve dve kolone
y_labels = data[:, -1]  # Poslednja kolona sadrži imena klasa

unique_classes = np.unique(y_labels)
class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

y = np.array([class_mapping[label] for label in y_labels])

# Podela skupa na trening i test deo
nb_samples = len(y)
indices = np.arange(nb_samples)
np.random.seed(44) # Nije neophodno, ali omogocuava da se pri svakom pokretanju dobijaju isti rezultati
np.random.shuffle(indices)

train_size = int(nb_samples * 0.7)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Kreiranje i treniranje KNN modela
accuracies = []
k_values = range(1, 16)

for k in k_values:
    knn = KNN(nb_features=2, nb_classes=3, data={'x': X_train, 'y': y_train}, k=k)
    predictions = knn.predict({'x': X_test})
    accuracy = np.sum(predictions == y_test) / len(y_test)
    accuracies.append(accuracy)
    print(f"k={k}, Test set accuracy: {accuracy:.2f}")

# Prikazivanje grafika
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.xlabel("Vrednost parametra k")
plt.ylabel("Tačnost")
plt.title("Zavisnost tačnosti od vrednosti parametra k")
plt.xticks(k_values)
plt.grid()
plt.show()


# Najniza tacnost se javlja za k=2 (0.71), sto ukazuje na overfitting.
# Najvisa tacnost je postignuta za k=9 (0.84), sto znaci da je ova vrednost potencijalno najbolji izbor za model.
# Relativno visoka tacnost je zabelezena za k=4,6, gde je vrednost 0.82