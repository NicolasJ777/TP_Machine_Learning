import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
import sklearn.neighbors as neighbors
import time

#apprentissage supervisé car étiquette associé
#phase de production : test de nlle donnée
#proba donne le pourcentage de voisins associé à la bonne classe



mnist = fetch_openml('mnist_784')

#plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")

images = mnist.data.reshape((-1, 28, 28))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(images[i],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title(mnist.target[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
sample=np.random.randint(70000, size=5000)
rand_data = mnist.data[sample]
rand_target = mnist.target[sample]


xtrain, xtest, ytrain, ytest = train_test_split(rand_data, rand_target, train_size=0.7)
'''
for i in range(10,15):
    n_neighbors = i+1
    clf = neighbors.KNeighborsClassifier (n_neighbors)

    clf.fit(xtrain,ytrain)
    
    target = clf.predict(xtest) # tableau de prediction de tous les élements de test
    print(ytest[3])
    print(target[3])
    print("Score pour k="+str(i)+" : ")
    print(clf.score(xtrain,ytrain))
'''
'''
for i in range(1,10):
    n_neighbors = 10
    
    percent = (i/10)
    xtrain, xtest, ytrain, ytest = train_test_split(rand_data, rand_target, train_size=percent)

    clf = neighbors.KNeighborsClassifier (n_neighbors)
    clf.fit(xtrain,ytrain)
    target = clf.predict(xtest) # tableau de prediction de tous les élements de test
    print(ytest[3])
    print(target[3])
    print("Score pour percent"+str(percent)+" : ")
    print(clf.score(xtrain,ytrain))
'''
'''
n_neighbors = 5
clf = neighbors.KNeighborsClassifier (n_neighbors,p=1)

clf.fit(xtrain,ytrain)
    
print("Score pour : ")
print(clf.score(xtrain,ytrain))
'''
'''
n_neighbors = 5
clf = neighbors.KNeighborsClassifier (n_neighbors, n_jobs=-1)
start_time = time.time()
clf.fit(xtrain,ytrain)
# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))
    
print("Score pour : ")
print(clf.score(xtrain,ytrain))
'''
'''

Score pour k=0 : 
0.922
Score pour k=1 : 
0.908
Score pour k=2 : 
0.924
Score pour k=3 : 
0.9186666666666666
Score pour k=4 : 
0.92
Score pour k=5 : 
0.9126666666666666
Score pour k=6 : 
0.9146666666666666
Score pour k=7 : 
0.9126666666666666
Score pour k=8 : 
0.91
Score pour k=9 : 
0.9073333333333333
Score pour k=10 : 
0.9053333333333333
Score pour k=11 : 
0.9006666666666666
Score pour k=12 : 
0.898
Score pour k=13 : 
0.8973333333333333
Score pour k=14 : 
0.894

'''
'''
for i in range(2,15):
    n_neighbors = i+1
    clf = neighbors.KNeighborsClassifier (n_neighbors)

    clf.fit(xtrain,ytrain)
     
    print("Score pour k="+str(i+1)+" : ")
    print(KFold(len(xtest),n_folds=10,shuffle=True) )'''
 
