# TODO popuniti kodom za problem 2b

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Pomoćna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp


# Funkcija troška i optimizacija.
def loss(x, y, w, b, lmbd, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    # Regularizacija

    if reg == 'l1':
        l1_reg = lmbd * tf.reduce_mean(tf.abs(w))
        loss = tf.add(mse, l1_reg)
    elif reg == 'l2':
        l2_reg = lmbd * tf.reduce_mean(tf.square(w))
        loss = tf.add(mse, l2_reg)
    else:
        loss = mse

    return loss


# Računanje gradijenta
def calc_grad(x, y, w, b, lmbd):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, lmbd, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val


# Trening korak
def train_step(x, y, w, b, adam, lmbd):
    w_grad, b_grad, loss = calc_grad(x, y, w, b, lmbd)

    adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss


# Učitavanje podataka
filename = 'data/bottle.csv'
all_data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(5, 6), dtype='float32', invalid_raise=False)
valid_data = all_data[~np.isnan(all_data).any(axis=1)]  # Uklanja NaN vrednosti

data = dict()
data['x'] = valid_data[:700, 0]  # Salinitet
data['y'] = valid_data[:700, 1]  # Temperatura

# Nasumično mešanje (nije obavezno, ali može pomoći).
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalizacija podataka (obratiti pažnju na axis=0).
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

# Parametri modela
learning_rate = 0.001
nb_epochs = 10
nb_features = 4
lmbd_values = [0, 0.001, 0.01, 0.1, 1, 10, 100]

# Čuvamo vrednosti funkcije troška
losses = []

# Prvi grafik
plt.scatter(data['x'], data['y'], color='blue', label='Data', alpha=0.5)  # svi podaci iz skupa

# Iteracija kroz stepen polinoma
for lmbd in lmbd_values:
    print(f"Training model with lambda = {lmbd} ...")

    # Kreiranje feature matrice.
    x_poly = create_feature_matrix(data['x'], nb_features)

    # Model i parametri.
    w = tf.Variable(tf.zeros(nb_features))
    b = tf.Variable(0.0)

    # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
    # slozenijim funkcijama.
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Trening.
    for epoch in range(nb_epochs):

        # Stochastic Gradient Descent.
        epoch_loss = 0
        for sample in range(nb_samples):
            x_sample = x_poly[sample].reshape((1, nb_features))  # x_sample je sada u obliku (1, degree)
            y = data['y'][sample]

            curr_loss = train_step(x_sample, y, w, b, adam, lmbd)
            epoch_loss += curr_loss

        # U svakoj stotoj epohi ispisujemo prosečan loss.
        epoch_loss /= nb_samples
        print(f'Epoch: {epoch + 1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

    # Ispisujemo i plotujemo finalnu vrednost parametara.
    print(f'w = {w.numpy()}, bias = {b.numpy()}')
    xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'), nb_features)
    hyp_val = pred(xs, w, b)
    plt.plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), label=f'lambda {lmbd}') # regresiona kriva

    # Finalna funkcija troska
    final_loss = loss(x_poly, data['y'], w, b, lmbd, reg='l2').numpy()
    losses.append(final_loss)
    print(f'Final Loss for lambda = {lmbd}: {final_loss:.5f}')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.show()

# Drugi grafik
plt.plot(lmbd_values, losses, marker='o', linestyle='-', color='red') # zavisnost finalne funkcije troska na celom skupu od stepena polinoma
plt.xlabel("Lambda")
plt.ylabel("Final Loss")
plt.title("Loss vs Lambda")
plt.show()

# Model je overfitted za male vrednosti lambde (0, 0.001)
# Model je underfitted za velike vrednosti lambde (10, 100)
# Kada se lambda prilizava 1, greska se naglo povecava
# Dobijamo najbolje rezultate za vrednosti izmedju 0.01 i 0.1