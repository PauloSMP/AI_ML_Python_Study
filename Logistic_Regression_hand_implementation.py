"""
σ(z)=1/ (1+e^−z1​)
where z = w*x + b
and σ(0) = 0.5

We cant use MSE as a loss function because it it will have multiple local minimum, vanishing gradients near the sigmoid limits.

A loss function that is most suitable for binary classification problems is the cross-entropy loss function.
L = - [y*log[y_hat] + (1-y) * log (1-y_hat)]

where y_hat = σ(z)

dl/dw = x*(y_hat-y)
dl/db = (y_hat-y)

#Exemplos
x = 2
y = 1
Classe verdadeira é 1.

w = 0
b = 0 

z = w * x + b => z = 0
if -> y_hat = σ(z) -> y_hat = 0.5

dL/dw = x * (y_hat - y)
dL/dw = 2 * (0.5 - 1)
dL/dw = -1

dL/db = y_hat - y
dL/db = 0.5 - 1
dL/db = -0.5

#Actualizar pesos e vies
 w = (w-lr*dl_dw)
 b = (b-lr*dl_db)

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------- 1. Dados ----------
# 2 features (para poderes visualizar depois), 2 classes
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1, random_state=42)
print (X)
print (y)
# X tem shape (500, 2) -- duas colunas, uma por feature
# y tem shape (500,) -- 0 ou 1

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------- 2. Inicialização ----------
# Atenção: agora tens DUAS features, logo DOIS pesos (w1, w2) -- ou um vector w de tamanho 2
w = np.zeros(2)  # TODO: np.zeros(2) -- um peso por feature
b = 0.0  # TODO: 0.0
lr = 0.01  # TODO
epochs = 1000  # TODO

loss_history = []

# ---------- 3. Funções auxiliares ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # TODO: 1 / (1 + np.exp(-z))

# ---------- 4. Loop de treino ----------
for i in range(epochs):
    # z = X @ w + b  -- multiplicação matriz-vector, substitui o "w*x" de antes
    z =  X_train @ w + b  # TODO
    y_hat = sigmoid(z)  # TODO: sigmoid(z)

    # Cross-entropy loss (média sobre todos os pontos)
    # cuidado: log(0) é -inf, por isso normalmente clipa-se y_hat para evitar isso
    y_hat_clipped = np.clip(y_hat, 1e-10, 1 - 1e-10)
    # Adicionar valor de lambda para regularização L2
    lambda_reg = 0.1
    loss = -np.mean(y_train*np.log(y_hat_clipped) + (1-y_train)*np.log(1-y_hat_clipped)) + (lambda_reg * np.sum(w**2)) # TODO: -np.mean(y*np.log(y_hat_clipped) + (1-y)*np.log(1-y_hat_clipped))
    loss_history.append(loss)

    # Gradientes -- mesma fórmula que derivaste, agora vectorizada
    # dL/dw = X.T @ (y_hat - y) / N   (repara: X.T @ erro faz a soma ponderada por feature)
    dL_dw = (np.transpose(X_train) @ (y_hat - y_train)/X_train.shape[0]) + 2*(lambda_reg * w)  # TODO
    dL_db = np.mean (y_hat - y_train)  # TODO: np.mean(y_hat - y)

    # Update
    w = (w - lr * dL_dw)   # TODO
    b = (b - lr * dL_db)  # TODO

# ---------- 5. Avaliação ----------
y_pred_train = (y_hat > 0.5).astype(int) # TODO: classifica como 1 se y_hat > 0.5, senão 0 (dica: (y_hat > 0.5).astype(int))
accuracy_train = np.mean(y_pred_train == y_train)

# Previsão em Teste
z_test = X_test @ w + b
y_hat_test = sigmoid(z_test)
y_pred_test = (y_hat_test > 0.5).astype(int)
accuracy_test = np.mean(y_pred_test == y_test)

print(f"Accuracy treino: {accuracy_train:.4f}")
print(f"Accuracy teste:  {accuracy_test:.4f}")

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Cross-Entropy Loss")
plt.show()