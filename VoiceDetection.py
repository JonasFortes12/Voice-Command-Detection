import scipy.io
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Carregar os sinais de áudios para o treinamento
audiosMatData = scipy.io.loadmat('./data/InputDataTrain.mat')

# Pegar a matriz de dados dos sinais de áudio
audiosMatrix = audiosMatData['InputDataTrain']

# Separar os sinais de áudio 'NÃO'
audio01Data = audiosMatrix[:, 0]
audio02Data = audiosMatrix[:, 1]
audio03Data = audiosMatrix[:, 2]
audio04Data = audiosMatrix[:, 3]
audio05Data = audiosMatrix[:, 4]

# Separae os sinais de áudio 'SIM'
audio06Data = audiosMatrix[:, 5]
audio07Data = audiosMatrix[:, 6]
audio08Data = audiosMatrix[:, 7]
audio09Data = audiosMatrix[:, 8]
audio10Data = audiosMatrix[:, 9]


# ________________________________Questão 01________________________________________

# Definir valores do eixo X
x = np.arange(0, audiosMatrix.shape[0])


# Criar uma figura para o gráfico de áudios 'NÃO'
plt.figure()

# Plotar os sinais de áudio 'NÃO' 
plt.plot(x, audio01Data, label='audio01', color='red', linewidth=0.6)
plt.plot(x, audio02Data, label='audio02', color='blue', linewidth=0.6)
plt.plot(x, audio03Data, label='audio03', color='green', linewidth=0.6)
plt.plot(x, audio04Data, label='audio04', color='black', linewidth=0.6)
plt.plot(x, audio05Data, label='audio05', color='orange', linewidth=0.6)

# Adicionar rótulos aos eixos
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Sinais do áudio "NÃO"')
# Adicionar uma legenda
plt.legend()


# Criar uma figura para o gráfico de áudios 'SIM'
plt.figure()

# Plotar os sinais de áudio 'SIM' 
plt.plot(x, audio06Data, label='audio06', color='red', linewidth=0.6)
plt.plot(x, audio07Data, label='audio07', color='blue', linewidth=0.6)
plt.plot(x, audio08Data, label='audio08', color='green', linewidth=0.6)
plt.plot(x, audio09Data, label='audio09', color='black', linewidth=0.6)
plt.plot(x, audio10Data, label='audio10', color='orange', linewidth=0.6)

# Adicionar rótulos aos eixos
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Sinais do áudio "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()







