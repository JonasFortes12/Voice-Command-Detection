import scipy.io
import numpy as np
import sounddevice as sd #sd.play(audio08Data, 44100)
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


# ________________________________Questão 03________________________________________

# Calcular o módulo ao quadrado da transformada de Fourier de cada sinal de áudio
audio01fft = np.abs(np.fft.fftshift(np.fft.fft(audio01Data)))**2
audio02fft = np.abs(np.fft.fftshift(np.fft.fft(audio02Data)))**2
audio03fft = np.abs(np.fft.fftshift(np.fft.fft(audio03Data)))**2
audio04fft = np.abs(np.fft.fftshift(np.fft.fft(audio04Data)))**2
audio05fft = np.abs(np.fft.fftshift(np.fft.fft(audio05Data)))**2
audio06fft = np.abs(np.fft.fftshift(np.fft.fft(audio06Data)))**2
audio07fft = np.abs(np.fft.fftshift(np.fft.fft(audio07Data)))**2
audio08fft = np.abs(np.fft.fftshift(np.fft.fft(audio08Data)))**2
audio09fft = np.abs(np.fft.fftshift(np.fft.fft(audio09Data)))**2
audio10fft = np.abs(np.fft.fftshift(np.fft.fft(audio10Data)))**2

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, audiosMatrix.shape[0])

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO'
plt.figure()

# Plotar fft dos sinais de áudio 'NÃO' 
plt.plot(x, audio01fft, label='audio01', color='red')
plt.plot(x, audio02fft, label='audio02', color='blue')
plt.plot(x, audio03fft, label='audio03', color='green')
plt.plot(x, audio04fft, label='audio04', color='black')
plt.plot(x, audio05fft, label='audio05', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier dos sinais de áudio "NÃO"')
# Adicionar uma legenda
plt.legend()


# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'
plt.figure()

# Plotar fft dos sinais de áudio 'SIM' 
plt.plot(x, audio06fft, label='audio06', color='red')
plt.plot(x, audio07fft, label='audio07', color='blue')
plt.plot(x, audio08fft, label='audio08', color='green')
plt.plot(x, audio09fft, label='audio09', color='black')
plt.plot(x, audio10fft, label='audio10', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier dos sinais de áudio "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()


# ________________________________Questão 04________________________________________


# Definir os indices das freqências no intervalo de 0 a pi/2 
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

# Copiar os sinais da transformada de Fourier dos áudios
audio01fft_filtered = audio01fft.copy()
audio02fft_filtered = audio02fft.copy()
audio03fft_filtered = audio03fft.copy()
audio04fft_filtered = audio04fft.copy()
audio05fft_filtered = audio05fft.copy()
audio06fft_filtered = audio06fft.copy()
audio07fft_filtered = audio07fft.copy()
audio08fft_filtered = audio08fft.copy()
audio09fft_filtered = audio09fft.copy()
audio10fft_filtered = audio10fft.copy()

# Anular as frequências fora do intervalo de 0 a pi/2 
audio01fft_filtered[~x_filtered] = 0
audio02fft_filtered[~x_filtered] = 0
audio03fft_filtered[~x_filtered] = 0
audio04fft_filtered[~x_filtered] = 0
audio05fft_filtered[~x_filtered] = 0
audio06fft_filtered[~x_filtered] = 0
audio07fft_filtered[~x_filtered] = 0
audio08fft_filtered[~x_filtered] = 0
audio09fft_filtered[~x_filtered] = 0
audio10fft_filtered[~x_filtered] = 0

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO' (Filtada)
plt.figure()

# Plotar fft dos sinais de áudio 'NÃO' 
plt.plot(x, audio01fft_filtered, label='audio01', color='red')
plt.plot(x, audio02fft_filtered, label='audio02', color='blue')
plt.plot(x, audio03fft_filtered, label='audio03', color='green')
plt.plot(x, audio04fft_filtered, label='audio04', color='black')
plt.plot(x, audio05fft_filtered, label='audio05', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier dos sinais de áudio "NÃO"')
# Adicionar uma legenda
plt.legend()

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'(Filtada)
plt.figure()

# Plotar fft dos sinais de áudio 'SIM' 
plt.plot(x, audio06fft_filtered, label='audio06', color='red')
plt.plot(x, audio07fft_filtered, label='audio07', color='blue')
plt.plot(x, audio08fft_filtered, label='audio08', color='green')
plt.plot(x, audio09fft_filtered, label='audio09', color='black')
plt.plot(x, audio10fft_filtered, label='audio10', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier dos sinais de áudio "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()


# ________________________________Questão 05________________________________________

# Dividir os sinais da TF dos áudios 'SIM' e 'NÃO' em 80 blocos de N/320 amostras
divisionNumber = 80
audio01fftDivided = np.array_split(audio01fft_filtered, divisionNumber)
audio02fftDivided = np.array_split(audio02fft_filtered, divisionNumber)
audio03fftDivided = np.array_split(audio03fft_filtered, divisionNumber)
audio04fftDivided = np.array_split(audio04fft_filtered, divisionNumber)
audio05fftDivided = np.array_split(audio05fft_filtered, divisionNumber)
audio06fftDivided = np.array_split(audio06fft_filtered, divisionNumber)
audio07fftDivided = np.array_split(audio07fft_filtered, divisionNumber)
audio08fftDivided = np.array_split(audio08fft_filtered, divisionNumber)
audio09fftDivided = np.array_split(audio09fft_filtered, divisionNumber)
audio10fftDivided = np.array_split(audio10fft_filtered, divisionNumber)

# Instânciar vetores para armazenar as energias dos blocos de sinais
audio01fft_filteredEnergies = []
audio02fft_filteredEnergies = []
audio03fft_filteredEnergies = []
audio04fft_filteredEnergies = []
audio05fft_filteredEnergies = []
audio06fft_filteredEnergies = []
audio07fft_filteredEnergies = []
audio08fft_filteredEnergies = []
audio09fft_filteredEnergies = []
audio10fft_filteredEnergies = []

# Calcular a energia de cada bloco nos 10 sinais de áudio
for i in range(divisionNumber):
    audio01fft_filteredEnergies.append(np.sum(np.square(audio01fftDivided[i])))
    audio02fft_filteredEnergies.append(np.sum(np.square(audio02fftDivided[i])))
    audio03fft_filteredEnergies.append(np.sum(np.square(audio03fftDivided[i])))
    audio04fft_filteredEnergies.append(np.sum(np.square(audio04fftDivided[i])))
    audio05fft_filteredEnergies.append(np.sum(np.square(audio05fftDivided[i])))
    audio06fft_filteredEnergies.append(np.sum(np.square(audio06fftDivided[i])))
    audio07fft_filteredEnergies.append(np.sum(np.square(audio07fftDivided[i])))
    audio08fft_filteredEnergies.append(np.sum(np.square(audio08fftDivided[i])))
    audio09fft_filteredEnergies.append(np.sum(np.square(audio09fftDivided[i])))
    audio10fft_filteredEnergies.append(np.sum(np.square(audio10fftDivided[i])))
    
# Definir valores do eixo X
x = np.arange(0, divisionNumber)


# Criar uma figura para o gráfico de energias da TF dos áudios 'NÃO'
plt.figure()

# Plotar os sinais de áudio 'NÃO' 
plt.plot(x, audio01fft_filteredEnergies, label='audio01', color='red')
plt.plot(x, audio02fft_filteredEnergies, label='audio02', color='blue')
plt.plot(x, audio03fft_filteredEnergies, label='audio03', color='green')
plt.plot(x, audio04fft_filteredEnergies, label='audio04', color='black')
plt.plot(x, audio05fft_filteredEnergies, label='audio05', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# Adicionar um título ao gráfico
plt.title('Energia da Transformada de Fourier dos áudios "NÃO"')
# Adicionar uma legenda
plt.legend()


# Criar uma figura para o gráfico de energias da TF dos áudios 'SIM'
plt.figure()

# Plotar os sinais de áudio 'SIM' 
plt.plot(x, audio06fft_filteredEnergies, label='audio06', color='red')
plt.plot(x, audio07fft_filteredEnergies, label='audio07', color='blue')
plt.plot(x, audio08fft_filteredEnergies, label='audio08', color='green')
plt.plot(x, audio09fft_filteredEnergies, label='audio09', color='black')
plt.plot(x, audio10fft_filteredEnergies, label='audio10', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# Adicionar um título ao gráfico
plt.title('Energia da Transformada de Fourier dos áudios "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()
