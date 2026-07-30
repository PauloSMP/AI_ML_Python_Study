import numpy as np
import matplotlib.pyplot as plt


# Gerar dados random
np.random.seed(1)
#Dados 
# Gerar x e y com uma relação conhecida + ruido, ex: y = 3x + 5 + ruido
x = np.random.uniform (0, 10, 1000)
x = (x - x.mean()) / x.std()
y = (3*x + 5 ) + (np.random.rand(1000)*0.5)

#print (x)
#print (y)

#Inicialização
w = 0.0
b = 0.0
lr = 0.01
epochs = 1000

loss_history = []

#Loop de treino

for i in range (epochs):
    y_hat = w * x + b
    loss = np.mean((y - y_hat)**2)
    loss_history.append(loss)
    
    #Gradientes
    dl_dw = np.mean (2*x*(y_hat - y))
    dl_db = np.mean (2*(y_hat - y))
    
    #Actualizar pesos e vies
    w = (w-lr*dl_dw)
    b = (b-lr*dl_db)
    
    if i % 1 ==0:
        print (f'Epoch {i}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}')

#Resultados
print (f'w = {w:.4f}, b = {b:.4f}')

plt.plot (loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss History')
plt.show()
