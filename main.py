import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

binomial = np.random.binomial(100.0, 0.5, 124750)

num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(binomial, num_bins, normed=1, facecolor='blue', alpha=0.5)

# add a 'best fit' line
mu = np.mean(binomial)
sigma = np.std(binomial)
y = stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu={}$, $\sigma={}$'.format(mu, sigma))

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
# ---------------

f = open('Files/NumSeeds0010.txt')
lines = f.readlines()
refLines = lines.copy()
dictHDs = []
numLines = len(lines)
totalCount = 0
HDs = []

for i in tqdm(range(numLines)):
    iStr = lines[i].strip()
    iLine = list(map(int, iStr))
    numChars = len(iLine)
    interHDs = []

    for j in range(numLines):
        if j <= i:
            continue

        jLine = list(map(int, refLines[j].strip()))
        count = np.sum(np.logical_xor(iLine, jLine))

        hd = 100.0 * count / numChars
        interHDs.append(hd)
        HDs.append(hd)
    dictHDs.append([iStr, interHDs])
    totalCount += len(interHDs)

num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(HDs, num_bins, normed=1, facecolor='blue', alpha=0.5)

# add a 'best fit' line
mu = np.mean(HDs)
sigma = np.std(HDs)
y = stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu={}$, $\sigma={}$'.format(mu, sigma))

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
