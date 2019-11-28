Modèle du reseau de neurone : le preceptron 

Dans cette partie nous allons utiliser la même base de donnée que celle de partie précédente. Cette fois ci on prends 70% des données ppour entrainer le modèle et le reste est utiliser pour le test du modèle.

Reseau de neurones à plusieurs couches auncun sens de mettre plusieure neurones sur la meme couche
On réalise X couches avec un nombre de neurones regressif par couche

Influence du nombre des couches par rapport à la précision : 
On affecte dans un premier temps 50 neurones par couches et on diminue le nombre de couches au fur et à mesure : 
par défaut Adam est utilisé
(on ne peut pas trouver le modèle parfait pour un système il faut procéder de manière empirique)
40 : 0.4342
20 : 0.9705
10 : 0.9727
2 :  0.9596

Ca dépends du nombre de paramètres et du nombre d'échantillons. En effet si on a beaucoup de paramètres ici 784, on peut ne pas converger au minimum d'erreur globale mais vers un minimum locale vu qu'on peut avoir moins de données. Ici c'est notre cas avec 40 couches composées de 50 neurones. Pour compenser cette défaillance on peut auglenter le nombre d'échantillons (ici uniquement 49000)  

#faire le temps d'execution
import time

# Debut du decompte du temps
start_time = time.time()

# Mettez votre code ici…

# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))


Test sur les données d'entraînement : 
40 : 0.7690     #c'est la memem chose

Diminuer le nombre de neurones par couches de manières regressif :
60 40 20 10 : 0.9613
10 20 40 60 : 0.9257

Comparaison : 
      Adam   lbfgs     sgd
20 : 0.9705  0.9133    0.9584
10 : 0.9727  0.9626    0.9601
2 :  0.9596  0.9254    0.9245
Meilleur adam , 

Comparaison en fonction des fonctions d'activations : 
      relu    tanh     logistic  identity  
10 : 0.9727   0.9291    0.1118    0.9152

Comparaison avec la valeur aplha, relu ,adam: 
       0.0001   0.001    0.01 
10 :   0.9727   


