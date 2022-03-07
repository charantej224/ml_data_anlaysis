import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

PALETTE = sns.color_palette('deep', n_colors=3)
CMAP = ListedColormap(PALETTE.as_hex())

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

x = np.array(df[['sepal length', 'sepal width', 'petal length', 'petal width']])
y = np.array(df[['target']])
x_embedded = TSNE(n_components=2, perplexity=30, n_iter=4000).fit_transform(x)
principalDf = pd.DataFrame(data=x_embedded
                           , columns=['Eigen 1', 'Eigen 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis=1)

print(finalDf.head(10))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Eigen 1', fontsize=15)
ax.set_ylabel('Eigen 2', fontsize=15)
ax.set_title('2 Eigen TSNE', fontsize=20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Eigen 1']
               , finalDf.loc[indicesToKeep, 'Eigen 2']
               , c=color
               , s=50)

ax.legend(targets)
ax.grid()
plt.show()
