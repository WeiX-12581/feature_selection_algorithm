import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
pea = []
spea = []
kendal = []
euclid = []
data = pd.read_csv('../data/iris.csv')
data['class'], _ = pd.factorize(data['class'])
print(list(data.columns))

fea = ['f1', 'f2', 'f3', 'f4']
# features = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']

for feature in list(data.columns)[0:-1]:
    pearson_corr, _ = pearsonr(data[feature], data['class'].astype(float))
    spearman_corr, _ = spearmanr(data[feature], data['class'].astype(float))
    kendall_corr, _ = kendalltau(data[feature], data['class'].astype(float))
    euclidean_dist = euclidean(data[feature], data['class'].astype(float))
    print(
        f'({feature}, class): ({abs(pearson_corr):.4f}, {abs(spearman_corr):.4f}, {abs(euclidean_dist):.4f}, {abs(kendall_corr):.4f}')
    pea.append(pearson_corr)
    spea.append(spearman_corr)
    kendal.append(kendall_corr)
    euclid.append(euclidean_dist)

plt.subplot(2, 2, 1)
plt.plot(fea, pea, linestyle='--', color='blue')
plt.title("pea")

plt.subplot(2, 2, 2)
plt.plot(fea, spea, linestyle='-.', color='red')
plt.title("spea")

plt.subplot(2, 2, 3)
plt.plot(fea, kendal, linestyle='dashed', color='green')
plt.title("kendal")

plt.subplot(2, 2, 4)
plt.plot(fea, euclid, linestyle=':', color='black')
plt.title("euclid")
plt.tight_layout()  # adjust the pic
plt.show()
