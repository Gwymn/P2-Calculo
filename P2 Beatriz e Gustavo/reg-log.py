import matplotlib.pyplot as plt
import math
import time

"""
Objetivo: 
Ensinar um modelo de aprendizado para usar regressão logísitca para classificar se um
estudante passou ou não baseado na média de suas notas

https://www.kaggle.com/datasets/cchen002/pass-or-not-students-exam-score-data
"""

def main():
    """
    Constantes
    """
    A_INICIAL = 0.1
    B_INICIAL = 0.1
    TAXA_APRENDIZADO = 10**-3
    TOLERANCIA = 10**-4

    """
    Variaveis
    """
    atributos = []
    rotulos = []

    """
    Pegando dados do arquivo
    """
    with open("./exam_scores.csv", 'r') as arquivo_dados:
        for linha in arquivo_dados.readlines():
            elementos = linha.strip().split(',')
            if elementos[0] == "Exam Score1":
                continue
            exam1 = float(elementos[0])
            exam2 = float(elementos[1])
            media = (exam1 + exam2) / 2
            atributos.append(media)
            rotulos.append(int(elementos[2]))

    """
    Ordenando X crescente
    """
    combinados = list(zip(atributos, rotulos))
    ordenados = sorted(combinados, key=lambda item: item[0])
    x_ordenados, y_ordenados = zip(*ordenados)
    atributos, rotulos = list(x_ordenados), list(y_ordenados)

    """
    Normalizando dados
    """
    menor = min(atributos)
    maior = max(atributos)
    atributos = [(v - menor) / (maior - menor) for v in atributos]

    """
    Separando dados de treino e teste
    """
    indice_divisao = int(len(atributos) * (1 - 0.1))
    treino_x = atributos[:indice_divisao]
    treino_y = rotulos[:indice_divisao]
    teste_x = atributos[indice_divisao:]
    teste_y = rotulos[indice_divisao:]

    """
    Rodando o gradiente descendente e calculando a acuracia
    """
    t0 = time.time()
    iteracoes, a_otimizado, b_otimizado = gradDs_def(treino_x, treino_y, A_INICIAL, B_INICIAL, TOLERANCIA, TAXA_APRENDIZADO)
    acuracia = acc_def(teste_x, teste_y, a_otimizado, b_otimizado)
    f1 = f1_def(teste_x, teste_y, a_otimizado, b_otimizado)
    t1 = time.time()
    print(f"Iterações: {iteracoes} | A: {a_otimizado} | B: {b_otimizado} | Acuracia: {acuracia} | F1-Score: {f1}")
    print(f"Tempo: {t1 - t0}")

    """
    Gráficos
    """
    plt.scatter(treino_x, treino_y, color='red', label="Treinamento")
    plt.plot(treino_x, [sigmoid_def(a_otimizado * x + b_otimizado) for x in treino_x], color='blue', label="Sigmoide")
    plt.scatter(teste_x, teste_y, color='yellow', label="Teste")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sigmoide Gráfico")
    plt.legend()
    plt.grid(True)
    plt.show()


"""
Sigmoid
"""
def sigmoid_def(valor):
    return 1 / (1 + math.exp(-valor))

"""
Calcular distancia
"""
def dist_def(a1, b1, a2, b2):
    return math.sqrt((a2 - a1)**2 + (b2 - b1)**2)

"""
Calculos de gradientes
"""
def gdA_def(a, b, x, y_real):
    y_previsto = sigmoid_def(a * x + b)
    return (y_previsto - y_real) * x

def gdB_def(a, b, x, y_real):
    y_previsto = sigmoid_def(a * x + b)
    return (y_previsto - y_real)

def sumGd_def(dados_x, dados_y, a, b):
    gradiente_a_total = 0
    gradiente_b_total = 0

    for i in range(len(dados_x)):
        gradiente_a_total += gdA_def(a, b, dados_x[i], dados_y[i])
        gradiente_b_total += gdB_def(a, b, dados_x[i], dados_y[i])

    return gradiente_a_total, gradiente_b_total

def gradDs_def(x, y, a_inicial, b_inicial, tolerancia, taxa_aprendizado):
    a_atual, b_atual = a_inicial, b_inicial
    i = 0
    
    while True:
        grad_a, grad_b = sumGd_def(x, y, a_atual, b_atual)
        
        a_novo = a_atual - taxa_aprendizado * grad_a
        b_novo = b_atual - taxa_aprendizado * grad_b
        
        i += 1
        
        erro = dist_def(a_atual, b_atual, a_novo, b_novo)
        
        if erro > tolerancia:
            a_atual, b_atual = a_novo, b_novo
        else:
            break
    
    return i, a_atual, b_atual

"""
Calcular acurácia
"""
def acc_def(dados_x, dados_y, a, b):
    acertos = 0
    for i in range(len(dados_x)):
        y_real = dados_y[i]
        z = a * dados_x[i] + b
        y_previsto = 1 if sigmoid_def(z) >= 0.5 else 0
        if y_previsto == y_real:
            acertos += 1
    return acertos / len(dados_x)

"""
Calcular F1-Score
"""
def f1_def(dados_x, dados_y, a, b):
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    for i in range(len(dados_x)):
        y_real = dados_y[i]
        z = a * dados_x[i] + b
        y_previsto = 1 if sigmoid_def(z) >= 0.5 else 0
        
        if y_previsto == 1 and y_real == 1:
            verdadeiros_positivos += 1
        elif y_previsto == 1 and y_real == 0:
            falsos_positivos += 1
        elif y_previsto == 0 and y_real == 1:
            falsos_negativos += 1
    
    if verdadeiros_positivos + falsos_positivos == 0 or verdadeiros_positivos + falsos_negativos == 0:
        return 0
    
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    revocacao = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
    return 2 * (precisao * revocacao) / (precisao + revocacao)


if __name__ == "__main__":
    main()