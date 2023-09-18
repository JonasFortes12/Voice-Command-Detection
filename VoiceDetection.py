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

# Separar os sinais de áudio 'SIM'
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

# ________________________________Questão 02________________________________________


# Dividir os sinais de áudio 'SIM' e 'NÃO' em 80 blocos de N/80 amostras
divisionNumber = 80
audio01Divided = np.array_split(audio01Data, divisionNumber)
audio02Divided = np.array_split(audio02Data, divisionNumber)
audio03Divided = np.array_split(audio03Data, divisionNumber)
audio04Divided = np.array_split(audio04Data, divisionNumber)
audio05Divided = np.array_split(audio05Data, divisionNumber)
audio06Divided = np.array_split(audio06Data, divisionNumber)
audio07Divided = np.array_split(audio07Data, divisionNumber)
audio08Divided = np.array_split(audio08Data, divisionNumber)
audio09Divided = np.array_split(audio09Data, divisionNumber)
audio10Divided = np.array_split(audio10Data, divisionNumber)

# Instânciar vetores para armazenar as energias dos blocos de sinais
audio01Energies = []
audio02Energies = []
audio03Energies = []
audio04Energies = []
audio05Energies = []
audio06Energies = []
audio07Energies = []
audio08Energies = []
audio09Energies = []
audio10Energies = []

# Calcular a energia de cada bloco nos 10 sinais de áudio
for i in range(divisionNumber):
    audio01Energies.append(np.sum(np.square(audio01Divided[i])))
    audio02Energies.append(np.sum(np.square(audio02Divided[i])))
    audio03Energies.append(np.sum(np.square(audio03Divided[i])))
    audio04Energies.append(np.sum(np.square(audio04Divided[i])))
    audio05Energies.append(np.sum(np.square(audio05Divided[i])))
    audio06Energies.append(np.sum(np.square(audio06Divided[i])))
    audio07Energies.append(np.sum(np.square(audio07Divided[i])))
    audio08Energies.append(np.sum(np.square(audio08Divided[i])))
    audio09Energies.append(np.sum(np.square(audio09Divided[i])))
    audio10Energies.append(np.sum(np.square(audio10Divided[i])))

# Definir valores do eixo X
x = np.arange(0, divisionNumber)


# Criar uma figura para o gráfico de energias do áudios 'NÃO'
plt.figure()

# Plotar os sinais de áudio 'NÃO' 
plt.plot(x, audio01Energies, label='audio01', color='red')
plt.plot(x, audio02Energies, label='audio02', color='blue')
plt.plot(x, audio03Energies, label='audio03', color='green')
plt.plot(x, audio04Energies, label='audio04', color='black')
plt.plot(x, audio05Energies, label='audio05', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# Adicionar um título ao gráfico
plt.title('Energia dos sinais de áudio "NÃO"')
# Adicionar uma legenda
plt.legend()


# Criar uma figura para o gráfico de energias do áudios 'SIM'
plt.figure()

# Plotar os sinais de áudio 'SIM' 
plt.plot(x, audio06Energies, label='audio06', color='red')
plt.plot(x, audio07Energies, label='audio07', color='blue')
plt.plot(x, audio08Energies, label='audio08', color='green')
plt.plot(x, audio09Energies, label='audio09', color='black')
plt.plot(x, audio10Energies, label='audio10', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# Adicionar um título ao gráfico
plt.title('Energia dos sinais de áudio "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()
