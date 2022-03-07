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
# sns.scatterplot(x_embedded[:, 0], x_embedded[:, 1], hue=y)
sns.set_style("darkgrid")
plt.scatter(x_embedded[:,0], x_embedded[:, 1], cmap=CMAP, s=70)
plt.title("TSNE", fontsize=20, y=1.03)
plt.show()

plt.xlabel("1_Eigen", fontsize=16)
plt.ylabel("2_Eigen", fontsize=16)
