import scipy.io
from scipy.spatial import distance
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
audio01fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01Data))))
audio02fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02Data))))
audio03fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03Data))))
audio04fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04Data))))
audio05fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05Data))))
audio06fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06Data))))
audio07fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07Data))))
audio08fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio08Data))))
audio09fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio09Data))))
audio10fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio10Data))))

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

#Definindo os intervalos de corte do sinal (0 a pi/2 )
x_freqCutStart = x_filtered[0]
x_freqCutEnd = x_filtered[len(x_filtered)-1] + 1

# Filtrando os sinais FT para as baixas frequências (0 a pi/2 ) 
audio01fft_filtered = audio01fft[x_freqCutStart:x_freqCutEnd]
audio02fft_filtered = audio02fft[x_freqCutStart:x_freqCutEnd]
audio03fft_filtered = audio03fft[x_freqCutStart:x_freqCutEnd]
audio04fft_filtered = audio04fft[x_freqCutStart:x_freqCutEnd]
audio05fft_filtered = audio05fft[x_freqCutStart:x_freqCutEnd]
audio06fft_filtered = audio06fft[x_freqCutStart:x_freqCutEnd]
audio07fft_filtered = audio07fft[x_freqCutStart:x_freqCutEnd]
audio08fft_filtered = audio08fft[x_freqCutStart:x_freqCutEnd]
audio09fft_filtered = audio09fft[x_freqCutStart:x_freqCutEnd]
audio10fft_filtered = audio10fft[x_freqCutStart:x_freqCutEnd]



# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO' (Filtrada)
plt.figure()

# Plotar fft dos sinais de áudio 'NÃO' 
plt.plot(x_filtered, audio01fft_filtered, label='audio01', color='red')
plt.plot(x_filtered, audio02fft_filtered, label='audio02', color='blue')
plt.plot(x_filtered, audio03fft_filtered, label='audio03', color='green')
plt.plot(x_filtered, audio04fft_filtered, label='audio04', color='black')
plt.plot(x_filtered, audio05fft_filtered, label='audio05', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier dos sinais de áudio "NÃO"')
# Adicionar uma legenda
plt.legend()

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'(Filtrada)
plt.figure()

# Plotar fft dos sinais de áudio 'SIM' 
plt.plot(x_filtered, audio06fft_filtered, label='audio06', color='red')
plt.plot(x_filtered, audio07fft_filtered, label='audio07', color='blue')
plt.plot(x_filtered, audio08fft_filtered, label='audio08', color='green')
plt.plot(x_filtered, audio09fft_filtered, label='audio09', color='black')
plt.plot(x_filtered, audio10fft_filtered, label='audio10', color='orange')

# Adicionar rótulos aos eixos
plt.xlabel('Amostra')
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
    audio01fft_filteredEnergies.append(np.sum(audio01fftDivided[i]))
    audio02fft_filteredEnergies.append(np.sum(audio02fftDivided[i]))
    audio03fft_filteredEnergies.append(np.sum(audio03fftDivided[i]))
    audio04fft_filteredEnergies.append(np.sum(audio04fftDivided[i]))
    audio05fft_filteredEnergies.append(np.sum(audio05fftDivided[i]))
    audio06fft_filteredEnergies.append(np.sum(audio06fftDivided[i]))
    audio07fft_filteredEnergies.append(np.sum(audio07fftDivided[i]))
    audio08fft_filteredEnergies.append(np.sum(audio08fftDivided[i]))
    audio09fft_filteredEnergies.append(np.sum(audio09fftDivided[i]))
    audio10fft_filteredEnergies.append(np.sum(audio10fftDivided[i]))
    
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


# ________________________________Questão 06________________________________________

# Dividir os sinais de áudio 'SIM' e 'NÃO' em 10 blocos de N/10 amostras
divisionNumber = 10
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


# Calcular o módulo ao quadrado da transformada de Fourier de cada bloco dos sinais de áudio
# Transformada de Fourier de tempo curto (short-time Fourier transform – STFT)
audio01_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01Divided))))
audio02_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02Divided))))
audio03_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03Divided))))
audio04_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04Divided))))
audio05_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05Divided))))
audio06_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06Divided))))
audio07_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07Divided))))
audio08_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio08Divided))))
audio09_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio09Divided))))
audio10_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio10Divided))))


# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, int(audiosMatrix.shape[0]/divisionNumber))

# Definir os indices das freqências no intervalo de 0 a pi/2 
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

#Definir os índices dos blocos da STFT
N_blocs = np.arange(audio01_STFT.shape[0])

# Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) 
audio01_STFT_filtered = audio01_STFT[N_blocs[:, np.newaxis], x_filtered]
audio02_STFT_filtered = audio02_STFT[N_blocs[:, np.newaxis], x_filtered]
audio03_STFT_filtered = audio03_STFT[N_blocs[:, np.newaxis], x_filtered]
audio04_STFT_filtered = audio04_STFT[N_blocs[:, np.newaxis], x_filtered]
audio05_STFT_filtered = audio05_STFT[N_blocs[:, np.newaxis], x_filtered]
audio06_STFT_filtered = audio06_STFT[N_blocs[:, np.newaxis], x_filtered]
audio07_STFT_filtered = audio07_STFT[N_blocs[:, np.newaxis], x_filtered]
audio08_STFT_filtered = audio08_STFT[N_blocs[:, np.newaxis], x_filtered]
audio09_STFT_filtered = audio09_STFT[N_blocs[:, np.newaxis], x_filtered]
audio10_STFT_filtered = audio10_STFT[N_blocs[:, np.newaxis], x_filtered]

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO' (Filtrada)
plt.figure()

#Definir cores para as linhas dos gráficos de STFT
lineColors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'cyan', 'brown', 'gray', 'olive']

# Plotar fft dos sinais de áudio 'NÃO'
for i in range(divisionNumber): 
    color = lineColors[i % len(lineColors)]
    plt.plot(x_filtered, audio01_STFT_filtered[i], label=f'bloco {i+1}', color=color)

# Adicionar rótulos aos eixos
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier de tempo curto do sinal de áudio "NÃO"')
# Adicionar uma legenda
plt.legend()

# Criar uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'(Filtrada)
plt.figure()

# Plotar fft dos sinais de áudio 'SIM'
for i in range(divisionNumber): 
    color = lineColors[i % len(lineColors)]
    plt.plot(x_filtered, audio06_STFT_filtered[i], label=f'bloco {i+1}', color=color)    

# Adicionar rótulos aos eixos
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# Adicionar um título ao gráfico
plt.title('Transformada de Fourier de tempo curto do sinal de áudio "SIM"')
# Adicionar uma legenda
plt.legend()

# Exibir os gráficos
plt.show()



# ________________________________Questão 07________________________________________

# Dividir as STFT dos sinais de áudio 'SIM' e 'NÃO' em 8 blocos de N/320 amostras
divisionNumber = 8

# Instânciar vetores para armazenar de cada bloco da STFT dividido por 8  
STFT01DividedBlocs = []
STFT02DividedBlocs = []
STFT03DividedBlocs = []
STFT04DividedBlocs = []
STFT05DividedBlocs = []
STFT06DividedBlocs = []
STFT07DividedBlocs = []
STFT08DividedBlocs = []
STFT09DividedBlocs = []
STFT10DividedBlocs = []

# Armazenar cada bloco da STFT dividido por 8 (10x8x730)
for i in range(10):
    STFT01DividedBlocs.append(np.array_split(audio01_STFT[i], divisionNumber))
    STFT02DividedBlocs.append(np.array_split(audio02_STFT[i], divisionNumber))
    STFT03DividedBlocs.append(np.array_split(audio03_STFT[i], divisionNumber))
    STFT04DividedBlocs.append(np.array_split(audio04_STFT[i], divisionNumber))
    STFT05DividedBlocs.append(np.array_split(audio05_STFT[i], divisionNumber))
    STFT06DividedBlocs.append(np.array_split(audio06_STFT[i], divisionNumber))
    STFT07DividedBlocs.append(np.array_split(audio07_STFT[i], divisionNumber))
    STFT08DividedBlocs.append(np.array_split(audio08_STFT[i], divisionNumber))
    STFT09DividedBlocs.append(np.array_split(audio09_STFT[i], divisionNumber))
    STFT10DividedBlocs.append(np.array_split(audio10_STFT[i], divisionNumber))


# Instânciar vetores para armazenar as energias de cada bloco (N/320 amostras)
# Energias: 8 energias para cada uma das 10 STFTs
STFT01BlocsEnergy = []
STFT02BlocsEnergy = []
STFT03BlocsEnergy = []
STFT04BlocsEnergy = []
STFT05BlocsEnergy = []
STFT06BlocsEnergy = []
STFT07BlocsEnergy = []
STFT08BlocsEnergy = []
STFT09BlocsEnergy = []
STFT10BlocsEnergy = []

# Calcular as 80 energias: 8 energias para cada uma das 10 partes dos STFT
for i in range(10):
    for j in range(8):
        STFT01BlocsEnergy.append(np.sum(STFT01DividedBlocs[i][j]))
        STFT02BlocsEnergy.append(np.sum(STFT02DividedBlocs[i][j]))
        STFT03BlocsEnergy.append(np.sum(STFT03DividedBlocs[i][j]))
        STFT04BlocsEnergy.append(np.sum(STFT04DividedBlocs[i][j]))
        STFT05BlocsEnergy.append(np.sum(STFT05DividedBlocs[i][j]))
        STFT06BlocsEnergy.append(np.sum(STFT06DividedBlocs[i][j]))
        STFT07BlocsEnergy.append(np.sum(STFT07DividedBlocs[i][j]))
        STFT08BlocsEnergy.append(np.sum(STFT08DividedBlocs[i][j]))
        STFT09BlocsEnergy.append(np.sum(STFT09DividedBlocs[i][j]))
        STFT10BlocsEnergy.append(np.sum(STFT10DividedBlocs[i][j]))
  
    
    
# ________________________________Questão 08________________________________________
    # Há 80 energias em cada domínio do sinal: tempo, TF e STFT
    
    # CENTROIDES:
    
# Calcular média das energias do áudio "NAO" para o domínio do tempo
meanTimeEnergy_NO = np.mean(np.array([audio01Energies, audio02Energies, audio03Energies, audio04Energies, audio05Energies]), axis=0)

# Calcular média das energias do áudio "SIM" para o domínio do tempo
meanTimeEnergy_YES = np.mean(np.array([audio06Energies, audio07Energies, audio08Energies, audio09Energies, audio10Energies]), axis=0)

# Calcular média das energias do áudio "NAO" para o domínio de TF
meanTFEnergy_NO = np.mean(np.array([audio01fft_filteredEnergies, audio02fft_filteredEnergies, audio03fft_filteredEnergies, audio04fft_filteredEnergies, audio05fft_filteredEnergies]), axis=0)

# Calcular média das energias do áudio "SIM" para o domínio de TF
meanTFEnergy_YES = np.mean(np.array([audio06fft_filteredEnergies, audio07fft_filteredEnergies, audio08fft_filteredEnergies, audio09fft_filteredEnergies, audio10fft_filteredEnergies]), axis=0)

# Calcular média das energias do áudio "NAO" para o domínio de STFT
meanSTFTEnergy_NO = np.mean(np.array([STFT01BlocsEnergy, STFT02BlocsEnergy, STFT03BlocsEnergy, STFT04BlocsEnergy, STFT05BlocsEnergy]), axis=0)

# Calcular média das energias do áudio "SIM" para o domínio de STFT
meanSTFTEnergy_YES = np.mean(np.array([STFT06BlocsEnergy, STFT07BlocsEnergy, STFT08BlocsEnergy, STFT09BlocsEnergy, STFT10BlocsEnergy]), axis=0)


# ________________________________Questão 09________________________________________

# Carregar os sinais de áudios para o teste
audiosMatDataTest = scipy.io.loadmat('./data/InputDataTest.mat')

# Pegar a matriz de dados dos sinais de áudio
audiosTestMatrix = audiosMatDataTest['InputDataTest']

# Separar os sinais de áudio 'NÃO'
audio01DataTest = audiosTestMatrix[:, 0]
audio02DataTest = audiosTestMatrix[:, 1]
audio03DataTest = audiosTestMatrix[:, 2]

# Separar os sinais de áudio 'SIM'
audio04DataTest = audiosTestMatrix[:, 3]
audio05DataTest = audiosTestMatrix[:, 4]
audio06DataTest = audiosTestMatrix[:, 5]
audio07DataTest = audiosTestMatrix[:, 6]



    #______________________ Cálculo de energias para o domínio do tempo ________________________

# Dividir os sinais de teste 'SIM' e 'NÃO' em 80 blocos de N/80 amostras
divisionNumber = 80
audio01TestDivided = np.array_split(audio01DataTest, divisionNumber)
audio02TestDivided = np.array_split(audio02DataTest, divisionNumber)
audio03TestDivided = np.array_split(audio03DataTest, divisionNumber)
audio04TestDivided = np.array_split(audio04DataTest, divisionNumber)
audio05TestDivided = np.array_split(audio05DataTest, divisionNumber)
audio06TestDivided = np.array_split(audio06DataTest, divisionNumber)
audio07TestDivided = np.array_split(audio07DataTest, divisionNumber)


# Instânciar vetores para armazenar as energias dos blocos de sinais
audio01TestEnergies = []
audio02TestEnergies = []
audio03TestEnergies = []
audio04TestEnergies = []
audio05TestEnergies = []
audio06TestEnergies = []
audio07TestEnergies = []

# Calcular a energia de cada bloco nos 10 sinais de áudio
for i in range(divisionNumber):
    audio01TestEnergies.append(np.sum(np.square(audio01TestDivided[i])))
    audio02TestEnergies.append(np.sum(np.square(audio02TestDivided[i])))
    audio03TestEnergies.append(np.sum(np.square(audio03TestDivided[i])))
    audio04TestEnergies.append(np.sum(np.square(audio04TestDivided[i])))
    audio05TestEnergies.append(np.sum(np.square(audio05TestDivided[i])))
    audio06TestEnergies.append(np.sum(np.square(audio06TestDivided[i])))
    audio07TestEnergies.append(np.sum(np.square(audio07TestDivided[i])))



    #______________________ Cálculo de energias para o domínio de TF ________________________
     
# Calcular o módulo ao quadrado da transformada de Fourier de cada sinal de teste
audio01fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01DataTest))))
audio02fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02DataTest))))
audio03fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03DataTest))))
audio04fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04DataTest))))
audio05fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05DataTest))))
audio06fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06DataTest))))
audio07fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07DataTest))))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, audiosTestMatrix.shape[0])

# Definir os indices das freqências no intervalo de 0 a pi/2 
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

#Definindo os intervalos de corte do sinal (0 a pi/2 )
x_freqCutStart = x_filtered[0]
x_freqCutEnd = x_filtered[len(x_filtered)-1] + 1

# Filtrando os sinais FT para as baixas frequências (0 a pi/2) 
audio01Testfft_filtered = audio01fft[x_freqCutStart:x_freqCutEnd]
audio02Testfft_filtered = audio02fft[x_freqCutStart:x_freqCutEnd]
audio03Testfft_filtered = audio03fft[x_freqCutStart:x_freqCutEnd]
audio04Testfft_filtered = audio04fft[x_freqCutStart:x_freqCutEnd]
audio05Testfft_filtered = audio05fft[x_freqCutStart:x_freqCutEnd]
audio06Testfft_filtered = audio06fft[x_freqCutStart:x_freqCutEnd]
audio07Testfft_filtered = audio07fft[x_freqCutStart:x_freqCutEnd]

# Dividir os sinais da TF dos áudios de teste 'SIM' e 'NÃO' em 80 blocos
divisionNumber = 80
audio01TestfftDivided = np.array_split(audio01Testfft_filtered, divisionNumber)
audio02TestfftDivided = np.array_split(audio02Testfft_filtered, divisionNumber)
audio03TestfftDivided = np.array_split(audio03Testfft_filtered, divisionNumber)
audio04TestfftDivided = np.array_split(audio04Testfft_filtered, divisionNumber)
audio05TestfftDivided = np.array_split(audio05Testfft_filtered, divisionNumber)
audio06TestfftDivided = np.array_split(audio06Testfft_filtered, divisionNumber)
audio07TestfftDivided = np.array_split(audio07Testfft_filtered, divisionNumber)

# Instânciar vetores para armazenar as energias dos blocos de sinais
audio01Testfft_filteredEnergies = []
audio02Testfft_filteredEnergies = []
audio03Testfft_filteredEnergies = []
audio04Testfft_filteredEnergies = []
audio05Testfft_filteredEnergies = []
audio06Testfft_filteredEnergies = []
audio07Testfft_filteredEnergies = []


# Calcular a energia de cada bloco nos 10 sinais de áudio
for i in range(divisionNumber):
    audio01Testfft_filteredEnergies.append(np.sum(audio01TestfftDivided[i]))
    audio02Testfft_filteredEnergies.append(np.sum(audio02TestfftDivided[i]))
    audio03Testfft_filteredEnergies.append(np.sum(audio03TestfftDivided[i]))
    audio04Testfft_filteredEnergies.append(np.sum(audio04TestfftDivided[i]))
    audio05Testfft_filteredEnergies.append(np.sum(audio05TestfftDivided[i]))
    audio06Testfft_filteredEnergies.append(np.sum(audio06TestfftDivided[i]))
    audio07Testfft_filteredEnergies.append(np.sum(audio07TestfftDivided[i]))
    
    

    #______________________ Cálculo de energias para o domínio de STFT ________________________
    
    # Dividir os sinais de teste 'SIM' e 'NÃO' em 10 blocos de N/10 amostras
divisionNumber = 10
audio01TestDivided = np.array_split(audio01DataTest, divisionNumber)
audio02TestDivided = np.array_split(audio02DataTest, divisionNumber)
audio03TestDivided = np.array_split(audio03DataTest, divisionNumber)
audio04TestDivided = np.array_split(audio04DataTest, divisionNumber)
audio05TestDivided = np.array_split(audio05DataTest, divisionNumber)
audio06TestDivided = np.array_split(audio06DataTest, divisionNumber)
audio07TestDivided = np.array_split(audio07DataTest, divisionNumber)


# Calcular o módulo ao quadrado da transformada de Fourier de cada bloco dos sinais de teste
# Transformada de Fourier de tempo curto (short-time Fourier transform – STFT)
audio01Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01TestDivided))))
audio02Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02TestDivided))))
audio03Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03TestDivided))))
audio04Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04TestDivided))))
audio05Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05TestDivided))))
audio06Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06TestDivided))))
audio07Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07TestDivided))))


# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, int(audiosTestMatrix.shape[0]/divisionNumber))

# Definir os indices das freqências no intervalo de 0 a pi/2 
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

#Definir os índices dos blocos da STFT
N_blocs = np.arange(audio01Test_STFT.shape[0])

# Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) 
audio01_STFT_filtered = audio01Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio02_STFT_filtered = audio02Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio03_STFT_filtered = audio03Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio04_STFT_filtered = audio04Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio05_STFT_filtered = audio05Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio06_STFT_filtered = audio06Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio07_STFT_filtered = audio07Test_STFT[N_blocs[:, np.newaxis], x_filtered]

# Dividir as STFT dos sinais de teste 'SIM' e 'NÃO' em 8 blocos de N/320 amostras
divisionNumber = 8

# Instânciar vetores para armazenar de cada bloco da STFT dividido por 8  
STFT01TestDividedBlocs = []
STFT02TestDividedBlocs = []
STFT03TestDividedBlocs = []
STFT04TestDividedBlocs = []
STFT05TestDividedBlocs = []
STFT06TestDividedBlocs = []
STFT07TestDividedBlocs = []

# Armazenar cada bloco da STFT dividido por 8 (10x8)
for i in range(10):
    STFT01TestDividedBlocs.append(np.array_split(audio01_STFT_filtered, divisionNumber))
    STFT02TestDividedBlocs.append(np.array_split(audio02_STFT_filtered, divisionNumber))
    STFT03TestDividedBlocs.append(np.array_split(audio03_STFT_filtered, divisionNumber))
    STFT04TestDividedBlocs.append(np.array_split(audio04_STFT_filtered, divisionNumber))
    STFT05TestDividedBlocs.append(np.array_split(audio05_STFT_filtered, divisionNumber))
    STFT06TestDividedBlocs.append(np.array_split(audio06_STFT_filtered, divisionNumber))
    STFT07TestDividedBlocs.append(np.array_split(audio07_STFT_filtered, divisionNumber))


# Instânciar vetores para armazenar as energias de cada bloco (N/320 amostras)
# Energias: 8 energias para cada uma das 10 STFTs
STFT01TestBlocsEnergy = []
STFT02TestBlocsEnergy = []
STFT03TestBlocsEnergy = []
STFT04TestBlocsEnergy = []
STFT05TestBlocsEnergy = []
STFT06TestBlocsEnergy = []
STFT07TestBlocsEnergy = []


# Calcular as 80 energias: 8 energias para cada uma das 10 partes dos STFT
for i in range(10):
    for j in range(8):
        STFT01TestBlocsEnergy.append(np.sum(STFT01TestDividedBlocs[i][j]))
        STFT02TestBlocsEnergy.append(np.sum(STFT02TestDividedBlocs[i][j]))
        STFT03TestBlocsEnergy.append(np.sum(STFT03TestDividedBlocs[i][j]))
        STFT04TestBlocsEnergy.append(np.sum(STFT04TestDividedBlocs[i][j]))
        STFT05TestBlocsEnergy.append(np.sum(STFT05TestDividedBlocs[i][j]))
        STFT06TestBlocsEnergy.append(np.sum(STFT06TestDividedBlocs[i][j]))
        STFT07TestBlocsEnergy.append(np.sum(STFT07TestDividedBlocs[i][j]))

# ________________________________Questão 10________________________________________


# Calculo das distâncias Euclidianas para o domínio de Tempo:

euclideanDist_audio01Test_Time_untilNO = distance.euclidean(audio01TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio02Test_Time_untilNO = distance.euclidean(audio02TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio03Test_Time_untilNO = distance.euclidean(audio03TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio04Test_Time_untilNO = distance.euclidean(audio04TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio05Test_Time_untilNO = distance.euclidean(audio05TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio06Test_Time_untilNO = distance.euclidean(audio06TestEnergies,meanTimeEnergy_NO)
euclideanDist_audio07Test_Time_untilNO = distance.euclidean(audio07TestEnergies,meanTimeEnergy_NO)
euclideanDist_Time_untilNO = [euclideanDist_audio01Test_Time_untilNO,
                              euclideanDist_audio02Test_Time_untilNO,
                              euclideanDist_audio03Test_Time_untilNO,
                              euclideanDist_audio04Test_Time_untilNO,
                              euclideanDist_audio05Test_Time_untilNO,
                              euclideanDist_audio06Test_Time_untilNO,
                              euclideanDist_audio07Test_Time_untilNO]


euclideanDist_audio01Test_Time_untilYes = distance.euclidean(audio01TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio02Test_Time_untilYes = distance.euclidean(audio02TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio03Test_Time_untilYes = distance.euclidean(audio03TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio04Test_Time_untilYes = distance.euclidean(audio04TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio05Test_Time_untilYes = distance.euclidean(audio05TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio06Test_Time_untilYes = distance.euclidean(audio06TestEnergies,meanTimeEnergy_YES)
euclideanDist_audio07Test_Time_untilYes = distance.euclidean(audio07TestEnergies,meanTimeEnergy_YES)
euclideanDist_Time_untilYes = [euclideanDist_audio01Test_Time_untilYes,
                               euclideanDist_audio02Test_Time_untilYes,
                               euclideanDist_audio03Test_Time_untilYes,
                               euclideanDist_audio04Test_Time_untilYes,
                               euclideanDist_audio05Test_Time_untilYes,
                               euclideanDist_audio06Test_Time_untilYes,
                               euclideanDist_audio07Test_Time_untilYes]

# Calculo das distâncias Euclidianas para o domínio da TF :

euclideanDist_audio01Test_FT_untilNO = distance.euclidean(audio01Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio02Test_FT_untilNO = distance.euclidean(audio02Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio03Test_FT_untilNO = distance.euclidean(audio03Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio04Test_FT_untilNO = distance.euclidean(audio04Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio05Test_FT_untilNO = distance.euclidean(audio05Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio06Test_FT_untilNO = distance.euclidean(audio06Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_audio07Test_FT_untilNO = distance.euclidean(audio07Testfft_filteredEnergies,meanTFEnergy_NO)
euclideanDist_FT_untilNO = [euclideanDist_audio01Test_FT_untilNO,
                            euclideanDist_audio02Test_FT_untilNO,
                            euclideanDist_audio03Test_FT_untilNO,
                            euclideanDist_audio04Test_FT_untilNO,
                            euclideanDist_audio05Test_FT_untilNO,
                            euclideanDist_audio06Test_FT_untilNO,
                            euclideanDist_audio07Test_FT_untilNO]


euclideanDist_audio01Test_FT_untilYes = distance.euclidean(audio01Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio02Test_FT_untilYes = distance.euclidean(audio02Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio03Test_FT_untilYes = distance.euclidean(audio03Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio04Test_FT_untilYes = distance.euclidean(audio04Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio05Test_FT_untilYes = distance.euclidean(audio05Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio06Test_FT_untilYes = distance.euclidean(audio06Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_audio07Test_FT_untilYes = distance.euclidean(audio07Testfft_filteredEnergies,meanTFEnergy_YES)
euclideanDist_FT_untilYes = [euclideanDist_audio01Test_FT_untilYes,
                             euclideanDist_audio02Test_FT_untilYes,
                             euclideanDist_audio03Test_FT_untilYes,
                             euclideanDist_audio04Test_FT_untilYes,
                             euclideanDist_audio05Test_FT_untilYes,
                             euclideanDist_audio06Test_FT_untilYes,
                             euclideanDist_audio07Test_FT_untilYes]

# Calculo das distâncias Euclidianas para o domínio da STFT:

euclideanDist_audio01Test_STFT_untilNO = distance.euclidean(STFT01TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio02Test_STFT_untilNO = distance.euclidean(STFT02TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio03Test_STFT_untilNO = distance.euclidean(STFT03TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio04Test_STFT_untilNO = distance.euclidean(STFT04TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio05Test_STFT_untilNO = distance.euclidean(STFT05TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio06Test_STFT_untilNO = distance.euclidean(STFT06TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_audio07Test_STFT_untilNO = distance.euclidean(STFT07TestBlocsEnergy,meanSTFTEnergy_NO)
euclideanDist_STFT_untilNO = [euclideanDist_audio01Test_STFT_untilNO,
                               euclideanDist_audio02Test_STFT_untilNO,
                               euclideanDist_audio03Test_STFT_untilNO,
                               euclideanDist_audio04Test_STFT_untilNO,
                               euclideanDist_audio05Test_STFT_untilNO,
                               euclideanDist_audio06Test_STFT_untilNO,
                               euclideanDist_audio07Test_STFT_untilNO]



euclideanDist_audio01Test_STFT_untilYes = distance.euclidean(STFT01TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio02Test_STFT_untilYes = distance.euclidean(STFT02TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio03Test_STFT_untilYes = distance.euclidean(STFT03TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio04Test_STFT_untilYes = distance.euclidean(STFT04TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio05Test_STFT_untilYes = distance.euclidean(STFT05TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio06Test_STFT_untilYes = distance.euclidean(STFT06TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_audio07Test_STFT_untilYes = distance.euclidean(STFT07TestBlocsEnergy,meanSTFTEnergy_YES)
euclideanDist_STFT_untilYes = [euclideanDist_audio01Test_STFT_untilYes,
                              euclideanDist_audio02Test_STFT_untilYes,
                              euclideanDist_audio03Test_STFT_untilYes,
                              euclideanDist_audio04Test_STFT_untilYes,
                              euclideanDist_audio05Test_STFT_untilYes,
                              euclideanDist_audio06Test_STFT_untilYes,
                              euclideanDist_audio07Test_STFT_untilYes]



TimeDomainHits = 0
FTDomainHits = 0
STFTDomainHits = 0

# Comparação de acertos com os áudios NÃO
for i in range(3):
    if(euclideanDist_Time_untilNO[i]<euclideanDist_Time_untilYes[i]):
        TimeDomainHits += 1      
    if(euclideanDist_FT_untilNO[i]<euclideanDist_FT_untilYes[i]):
        FTDomainHits += 1 
    if(euclideanDist_STFT_untilNO[i]<euclideanDist_STFT_untilYes[i]):
        STFTDomainHits += 1 

# Comparação de acertos com os áudios SIM
for i in range(3,7):
    if(euclideanDist_Time_untilNO[i]>euclideanDist_Time_untilYes[i]):
        TimeDomainHits += 1
    if(euclideanDist_FT_untilNO[i]>euclideanDist_FT_untilYes[i]):
        FTDomainHits += 1 
    if(euclideanDist_STFT_untilNO[i]>euclideanDist_STFT_untilYes[i]):
        STFTDomainHits += 1 


print(TimeDomainHits)
print(FTDomainHits)
print(STFTDomainHits)

'''
______________________________ Desenpendo dos Domínios ________________________________

    (Descrição dos resultados aqui)






'''


