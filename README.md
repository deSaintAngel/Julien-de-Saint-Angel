
## Authors : 
 
* [Julien de SAINT ANGEL](mailto:juliencine17@gmail.com)

# Projet du perceptron au GAN : Le Deep_Learning 
## Deep Learning : le principe 
Le maching learning est un processus informatique inspiré par la biologie qui consiste à creer un réseau de neurones sous forme de graphe qui à partir d'une entré, va extraires des caracteristiques et donner une prédiction. Un neurone se présente de la façon suivante :



les <img src="https://render.githubusercontent.com/render/math?math=x_i"> sont les valeurs d'entrés du neurones. Les <img src="https://render.githubusercontent.com/render/math?math=w_i"> sont les poids qui seront au cours de l'apprentissage ajusté pour donner une valeur de prédiction. f est une fonction dite d'activation ( identité, relu, softmax ...). celle-ci depend du problème à résoudre. Enfin <img src="https://render.githubusercontent.com/render/math?math=y"> est la valeur d'activation du neurone artificiel. 

l'obtention de la valeur de sortie à partir des neurones d'entrés correspond à l'inférence. 

Le principe est le suivant, on calcule l'inférence. On donne des valeurs d'entées (les pixels d'une image par exemple), on calcule les valeurs d'activations de chaques couche de neurons, puis on regarde la différence entre les valeurs de prédictions obtenue et les valeurs attendus ( appellé targets). 
L'etape suivante ( l'apprentissage qui correspond à la backpropagation) consiste à minimiser l'ecart qui est donnée par la fonction de couts ( cost function). Enfin on ajuste avec un taux d'apprentissage les poids pour obtenir une valeurs d'activation proche de la valeurs espéré. On réitère les étapes jusqu'a ce que la fonction de cout atteigne un certain seuil. 


<img src="https://render.githubusercontent.com/render/math?math=e^{i +\pi} =x+1">
