import numpy as np

# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.
def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))
    
# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=1)