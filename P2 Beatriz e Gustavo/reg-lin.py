import matplotlib.pyplot as plt
import math
import time

"""
Objetivo: 
Traçar a melhor reta linear que se ajuste aos dados usando o gradiente descendente
"""

def main():
    """
    Constantes
    """
    GRAU = 22
    TOLERANCIA = 10**-8
    LEARNING_RATE = 10**-3
    """
    Variaveis
    """
    datasetx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    datasety = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    """
    Normalizador (evita overflow na memoria)
    """
    max_x = max(datasetx)
    max_y = max(datasety)
    datasetx = [x / max_x for x in datasetx]
    datasety = [y / max_y for y in datasety]

    """
    Rodando o gradiente descendente
    """
    t0 = time.time()
    iteracoes, polinomios_final = gradDs_defs([1] * (GRAU + 1), TOLERANCIA, LEARNING_RATE, GRAU, datasetx, datasety)
    t1 = time.time()
    print(f"Número de iterações: {iteracoes}, valores finais dos coeficientes: {polinomios_final}")
    print(f"Tempo: {t1 - t0}")
    
    """
    Gerador de pontos no gráfico com ajuste de 0.1
    """
    x_grafico = [i * 0.1 for i in range(161)]
    y_grafico = []
    for x in x_grafico:
        y = 0
        for i in range(GRAU + 1):
            y += polinomios_final[i] * x**(GRAU - i)
        y_grafico.append(y)

    """
    Gráfico
    """
    plt.scatter(datasetx, datasety, color='blue', label='Pontos do dataset')
    plt.plot(x_grafico, y_grafico, label='Curva quadrática ajustada', color='red')
    plt.title('Equação quadrática usando gradiente descendente')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()


"""
Calculos de gradientes
"""
def grad_def(datasetx, datasety, coeficientes, GRAU):
    gradientes = [0] * len(coeficientes)
    
    for i in range(len(datasetx)):
        erro = 0
        for j in range(len(coeficientes)):
            erro += coeficientes[j] * datasetx[i]**(GRAU-j)
        erro -= datasety[i]
        
        for j in range(len(coeficientes)):
            gradientes[j] += erro * datasetx[i]**(GRAU-j)

    return gradientes

def gradDs_defs(polinomios, tolerancia, learning_rate, GRAU, datasetx, datasety):
    polinomios_anterior = polinomios.copy()
    polinomios_posterior = [0] * len(polinomios)
    i = 0
    while True:
        derivada = grad_def(datasetx, datasety, polinomios_anterior, GRAU)
        for j in range(len(polinomios_posterior)):
            polinomios_posterior[j] = polinomios_anterior[j] - learning_rate * derivada[j]  # Corrigido aqui
        
        dist = dist_def(polinomios_anterior, polinomios_posterior)
        i += 1

        if dist > tolerancia:
            polinomios_anterior = polinomios_posterior.copy()
        else:
            break
    return i, polinomios_anterior

"""
Distância
"""

def dist_def(polinomios_anterior, polinomios_posterior):
    d = 0
    for i in range(len(polinomios_anterior)):
        d += (polinomios_anterior[i] - polinomios_posterior[i])**2
    return math.sqrt(d)


if __name__ == "__main__":
    main()