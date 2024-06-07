# Deep Learning

## Objectif
L'objectif de ce projet est de démontrer la capacité des couches convolutionnelles à améliorer la précision d'un modèle de classification d'image.  
Requirements : ( tensorflow, numpy )
  
## Réalisation
- Dans le fichier `Deep_learning.py` un modèle de base :
Le modèle est une séquence de couches :
Une couche d'aplatissement (Flatten) pour convertir les images 2D en vecteurs 1D.
Une couche dense de 128 neurones avec une fonction d'activation ReLU.
Une couche de régularisation Dropout pour éviter le surapprentissage.
Une couche dense de sortie avec 10 neurones correspondant aux 10 classes de vêtements.
Une couche d'activation Softmax pour obtenir des probabilités de classe.
Il utilise une fonction de perte de Cross-Entropy Catégorielle Éparse (SparseCategoricalCrossentropy) pour mesurer l'écart entre les prédictions et les étiquettes réelles. Il est optimisé avec l'optimiseur Adam, utilisant la précision comme métrique d'évaluation.

- Dans le fichier `Deep_Learning_Conv.py` un modèle "amélioré" avec une couche convolutionelle ( profondeur 32 ).  

- Dans le fichier `Deep_learning_Pred.py` une fonction permettant de réaliser quelques prédictions ( 20 ).  

- Dans le fichier `Matrice_De_Confusion.py` une matrice de confusion sur 200 prédictions du modèle sans couche de convolution.  

## Conclusion
Nous avons réussi à améliorer la précision de ~5 points ( % ).  

*Danies Gabriel*, *Goulesque Enzo*  
*Mohammed Attik*
