import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as pltl
import os
import imageio

def beta_pd_funct(a, b, mu):
    pd_funct = (gamma(a + b) / (gamma(a) * gamma(b))) * (mu ** (a - 1)) * ((1 - mu) ** (b - 1))
    return pd_funct


def data_mean_size(mean, size):
    # If N = 1 then binomial becomes bernoulli distribution
    return np.random.binomial(1, mean, size)


def beta_distribution(a, b):
    return np.unique(np.random.beta(a, b, 10000))


def beta_distribution_plot(a, b, index, posterior):
    points = beta_distribution(a, b)
    X = points.tolist()
    Y = []
    for x in X:
        Y.append(beta_pd_funct(a, b, x))
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    if posterior == True:
        pltl.clf()
        pltl.title("Posterior Distribution of μ with a {} and b {}".format(a, b))
        pltl.xlabel("μ")
        pltl.ylabel("P(μ|D)")
        pltl.plot(X, Y)
        #pltl.show()
        image_name = "assets/" + str(index) + ".JPG"
        pltl.savefig(image_name)
    else:
        pltl.clf()
        pltl.title("Prior Distribution of μ with a {} and b {}".format(a, b))
        pltl.xlabel("μ")
        pltl.ylabel("P(μ)")
        pltl.plot(X, Y)
        #pltl.show()
        image_name = "assets/" + str(index) + ".JPG"
        pltl.savefig(image_name)
    pltl.clf()


"""
    Prior (a, b) beta distribution parameters
    Posterior (a + m, b + l) beta distribution parameters where m is number of 1s and b is number of 0s
"""
def posterior_sequential(likelihood_mean, likelihood_size, prior_mean):
    # Generate dataset of 1s and 0s for bernoulli distribution
    data_points = data_mean_size(likelihood_mean, likelihood_size)
    # Prior Distribution parameters
    a = prior_mean * 10
    b = (1 - prior_mean) * 10
    # Plot Prior Distribution
    beta_distribution_plot(a, b, "prior_sequential", False)
    for index, x in enumerate(data_points):
        # a -> a + m where m is number of 1s and b -> b + l where l is number of 0s
        if x == 1:
            a += 1
        else :
            b += 1
        # Plot Posterior Distribution
        beta_distribution_plot(a, b, index, True)


posterior_sequential(0.3, 160, 0.4)


# Generating GIF
images = []
filenames = []
for i in range(0, 160, 1):
    filenames.append("assets/{}.JPG".format(i))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('assets/posterior_sequential.gif', images, duration = 0.1)


# Deleting 0-159 images

for file in filenames:
    os.remove(file)


def posterior_entire(likelihood_mean, likelihood_size, prior_mean):
    # Generate dataset of 1s and 0s for bernoulli distribution
    data_points = data_mean_size(likelihood_mean, likelihood_size)
    # Prior Distribution parameters
    a = prior_mean * 10
    b = (1 - prior_mean) * 10
    # Plot Prior Distribution
    beta_distribution_plot(a, b, "prior_entire", False)
    m = 0
    for x in data_points:
        # a -> a + m where m is number of 1s and b -> b + l where l is number of 0s
        if x == 1:
            m += 1
    l = likelihood_size - m
    # Plot Posterior Distribution
    beta_distribution_plot(a + m, b + l, "posterior_entire", True)


posterior_entire(0.3, 160, 0.4)


