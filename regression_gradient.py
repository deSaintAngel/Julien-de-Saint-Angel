# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import numpy as np


# Gradient du paramètre m
def grad_a(a, b, X, Y):
    return sum(-2 * X * (Y- (a * X + b)) / float(len(X)))

# Gradient du paramètre b
def grad_b(a, b, X, Y):
    return sum(-2 * (Y - (a * X + b)) / float(len(X)))

def gradient_descent(X, Y, epochs, lr):
    a = 0
    b = 0
    for e in range(epochs):
        a = a - lr * grad_a(a, b, X, Y)
        b = b - lr * grad_b(a, b, X, Y)
    return a, b

if __name__ == '__main__':
    # Génération du jeu de données
    np.random.seed(13)
    X = np.linspace(-1, 1, 100) 
    Y = X + np.random.normal(0, 0.25, 100)

    # Exécution de l'algorithme
    a, b = gradient_descent(X, Y, epochs=1000, lr=0.01)

    # Visualisation de la droite avec les valeurs de m et b trouvées par descente de gradient
    y_predict = np.array(a*X + b)
    plt.plot(X, y_predict, 'b',label='aproximation')
    plt.plot(X, Y, 'ro',label='Original data')
    plt.legend()
    plt.show()
    
    #val = np.array(a*X + b)


    print("err :", np.mean(Y-y_predict))