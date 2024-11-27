import matplotlib.pyplot as plt
import math as math
import time

def main():
    """
    Constantes
    """
    A_INICIAL = 0.1
    B_INICIAL = 0.1
    C_INICIAL = 0.1
    D_INICIAL = 0.1
    K_INICIAL = 0.1
    L_INICIAL = 0.1
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
    iteracoes, a_otimizado, b_otimizado, c_otimizado, d_otimizado, k_otimizado, l_otimizado = gradDs_def(treino_x, treino_y, A_INICIAL, B_INICIAL, C_INICIAL, D_INICIAL, K_INICIAL, L_INICIAL, TOLERANCIA, TAXA_APRENDIZADO)
    acuracia = acc_def(teste_x, teste_y, a_otimizado, b_otimizado, c_otimizado, d_otimizado, k_otimizado, l_otimizado)
    f1 = f1_def(teste_x, teste_y, a_otimizado, b_otimizado, c_otimizado, d_otimizado, k_otimizado, l_otimizado)
    t1 = time.time()
    print(f"Iterações: {iteracoes} | A: {a_otimizado} | B: {b_otimizado} | C: {c_otimizado} | D: {d_otimizado} | K: {k_otimizado} | L: {l_otimizado} | Acuracia: {acuracia} | F1-Score: {f1}")
    print(f"Tempo: {t1 - t0}")

    """
    Gráficos
    """
    plt.scatter(treino_x, treino_y, color='red', label="Treinamento")
    plt.plot(treino_x, [sigmoid_def(a_otimizado * x ** 5 + b_otimizado * x ** 4 + c_otimizado * x ** 3 + d_otimizado * x ** 2 + k_otimizado * x + l_otimizado) for x in treino_x], color='blue', label="Sigmoide")
    plt.scatter(teste_x, teste_y, color='yellow', label="Teste")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sigmoide Gráfico")
    plt.legend()
    plt.grid(True)
    plt.show()


"""
Gradientes
"""
def gdA_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y) * x ** 5

def gdB_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y) * x ** 4

def gdC_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y) * x ** 3

def gdD_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y) * x ** 2

def gdK_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y) * x

def gdL_def(a, b, c, d, k, l, x, y):
    y_pred = sigmoid_def(a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l)
    return (y_pred - y)

def sigmoid_def(valor):
    return 1 / (1 + math.exp(-valor))

def gradDS(dados_x, dados_y, a, b, c, d, k, l):
    graiente_a_total = 0
    graiente_b_total = 0
    graiente_c_total = 0
    graiente_d_total = 0
    graiente_k_total = 0
    graiente_l_total = 0

    for i in range(len(dados_x)):
        graiente_a_total += gdA_def(a, b, c, d, k, l, dados_x[i], dados_y[i])
        graiente_b_total += gdB_def(a, b, c, d, k, l, dados_x[i], dados_y[i])
        graiente_c_total += gdC_def(a, b, c, d, k, l, dados_x[i], dados_y[i])
        graiente_d_total += gdD_def(a, b, c, d, k, l, dados_x[i], dados_y[i])
        graiente_k_total += gdK_def(a, b, c, d, k, l, dados_x[i], dados_y[i])
        graiente_l_total += gdL_def(a, b, c, d, k, l, dados_x[i], dados_y[i])

    return graiente_a_total, graiente_b_total, graiente_c_total, graiente_d_total, graiente_k_total, graiente_l_total

def dist2(a_inicial, b_inicial, c_inicial, d_inicial, k_inicial, l_inicial, a_n, b_n, c_n, d_n, k_n, l_n):
    return ((a_n - a_inicial)**2 + (b_n - b_inicial)**2 + (c_n - c_inicial)**2 + (d_n - d_inicial)**2 + (k_n - k_inicial)**2 + (l_n - l_inicial)**2)**0.5

def gradDs_def(X, Y, a_inicial, b_inicial, c_inicial, d_inicial, k_inicial, l_inicial, tol, lr):
    a_n, b_n, c_n, d_n, k_n, l_n = a_inicial, b_inicial, c_inicial, d_inicial, k_inicial, l_inicial
    a_n1, b_n1, c_n1, d_n1, k_n1, l_n1 = [99999999] * 6
    i = 0

    while True:
        grad_a, grad_b, grad_c, grad_d, grad_k, grad_l = gradDS(X, Y, a_n, b_n, c_n, d_n, k_n, l_n)

        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        c_n1 = c_n - lr * grad_c
        d_n1 = d_n - lr * grad_d
        k_n1 = k_n - lr * grad_k
        l_n1 = l_n - lr * grad_l

        i += 1

        err = dist2(a_n, b_n, c_n, d_n, k_n, l_n, a_n1, b_n1, c_n1, d_n1, k_n1, l_n1)

        if err > tol:
            a_n, b_n, c_n, d_n, k_n, l_n = a_n1, b_n1, c_n1, d_n1, k_n1, l_n1
        else:
            break

    return i, a_n, b_n, c_n, d_n, k_n, l_n

"""
Calcular acurácia
"""
def acc_def(dados_x, dados_y, a, b, c, d, k, l):
    acertos = 0
    for i in range(len(dados_x)):
        x = dados_x[i]
        y_real = dados_y[i]
        z = a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l
        y_pred = 1 if sigmoid_def(z) >= 0.5 else 0  # Previsão binária (limiar 0.5)
        if y_pred == y_real:
            acertos += 1
    return acertos / len(dados_x)

"""
Calcular F1-Score
"""
def f1_def(dados_x, dados_y, a, b, c, d, k, l):
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    for i in range(len(dados_x)):
        x = dados_x[i]
        y_real = dados_y[i]
        z = a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + k * x + l
        y_pred = 1 if sigmoid_def(z) >= 0.5 else 0  # Previsão binária (limiar 0.5)

        if y_pred == 1 and y_real == 1:
            verdadeiros_positivos += 1
        elif y_pred == 1 and y_real == 0:
            falsos_positivos += 1
        elif y_pred == 0 and y_real == 1:
            falsos_negativos += 1

    # Evitar divisão por zero
    if verdadeiros_positivos + falsos_positivos == 0 or verdadeiros_positivos + falsos_negativos == 0:
        return 0

    precision = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)  # Precisão
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)     # Recall
    return 2 * (precision * recall) / (precision + recall)  # F1 Score



if __name__ == "__main__":
    main()