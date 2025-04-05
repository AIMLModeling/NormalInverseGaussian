import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from functools import partial
from scipy.optimize import minimize
from scipy.integrate import quad
import scipy.special as scps

np.random.seed(seed=42)
paths = 40000  # number of paths
steps = 10000  # number of time steps

t = 2
delta = 3 * t  # time dependent barrier
gamma = 2  # drift
T_max = 20
T_vec, dt = np.linspace(0, T_max, steps, retstep=True)
X0 = np.zeros((paths, 1))  # each path starts at zero
increments = ss.norm.rvs(loc=gamma * dt, scale=np.sqrt(dt), size=(paths, steps - 1))

Z = np.concatenate((X0, increments), axis=1).cumsum(1)
T = np.argmax(Z > delta, axis=1) * dt  # first passage time
# Computes parameters for the Inverse Gaussian (IG) distribution, which models first-passage time.
x = np.linspace(0, 10, 10000)
mm = delta / gamma
lam = delta**2
mm1 = mm / lam  # scaled mean

# Plots the histogram of simulated hitting times and compares it with the IG PDF.
plt.plot(x, ss.invgauss.pdf(x, mu=mm1, scale=lam), color="red", label="IG density")
plt.hist(T, density=True, bins=100, facecolor="LightBlue", label="frequencies of T")
plt.title("First passage time distribution")
plt.legend()
plt.show()

print("Theoretical mean: ", mm)
print("Theoretical variance: ", delta / gamma**3)
print("Estimated mean: ", T.mean())
print("Estimated variance: ", T.var())
T = 2  # terminal time
N = 1000000  # number of generated random variables

theta = -0.1  # drift of the Brownian motion
sigma = 0.2  # volatility of the Brownian motion
kappa = 0.5  # variance of the Gamma process

lam = T**2 / kappa  # scale
mus = T / lam  # scaled mu
np.random.seed(seed=42)
IG = ss.invgauss.rvs(mu=mus, scale=lam, size=N)  # The IG RV
Norm = ss.norm.rvs(0, 1, N)  # The normal RV
X = theta * IG + sigma * np.sqrt(IG) * Norm
def cf_NIG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Normal Inverse Gaussian random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Inverse Gaussian process variance
    """
    return np.exp(
        t * (1j * mu * u + 1 / kappa - np.sqrt(1 - 2j * theta * kappa * u + kappa * sigma**2 * u**2) / kappa)
    )
# Gil-Pelaez PDF Inversion
# Numerically inverts the CF to compute the PDF using Gil-Pelaez theorem.
def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    1.	Computes the PDF by inverting the characteristic function using the Gil-Pelaez theorem.
    2.	quad: Performs numerical integration from 00 to right_lim.
    """
    def integrand(u):
        return np.real(np.exp(-u * x * 1j) * cf(u))
    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]
def NIG_density(x, T, c, theta, sigma, kappa):
    A = theta / (sigma**2)
    B = np.sqrt(theta**2 + sigma**2 / kappa) / sigma**2
    C = T / np.pi * np.exp(T / kappa) * np.sqrt(theta**2 / (kappa * sigma**2) + 1 / kappa**2)
    return (
        C
        * np.exp(A * (x - c * T))
        * scps.kv(1, B * np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa))
        / np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa)
    )
# Evaluates NIG PDF via modified Bessel function.
# Evaluate the NIG density using parameters α, β, δ, and μ,
# derived from the internal parametrization via theta, sigma, kappa, c.
def NIG_abdmu(x, T, c, theta, sigma, kappa):
    beta = theta / (sigma**2)
    alpha = np.sqrt(beta**2 + 1 / (kappa * sigma**2))
    delta = T * sigma / np.sqrt(kappa)
    mu = c * T
    g = lambda y: np.sqrt(delta**2 + (x - mu) ** 2)
    cost = delta * alpha / np.pi
    return (
        cost
        * np.exp(delta * np.sqrt(alpha**2 - beta**2) + beta * (x - mu))
        * scps.kv(1, alpha * g(x - mu))
        / g(x - mu)
    )
cf_NIG_b = partial(cf_NIG, t=T, mu=0, theta=theta, sigma=sigma, kappa=kappa)
x = np.linspace(X.min(), X.max(), 500)
y = np.linspace(-2, 1, 30)

plt.figure(figsize=(16, 5))
plt.plot(x, NIG_density(x, T, 0, theta, sigma, kappa), color="r", label="NIG density")
plt.plot(y, [Gil_Pelaez_pdf(i, cf_NIG_b, np.inf) for i in y], "p", label="Fourier inversion")
plt.hist(X, density=True, bins=200, facecolor="LightBlue", label="frequencies of X")
plt.legend()
plt.title("NIG Histogram")
plt.show()
qqplot(X, line="s")
plt.show()
sigma_mm1 = np.std(X) / np.sqrt(T)
kappa_mm1 = T * ss.kurtosis(X) / 3
theta_mm1 = np.sqrt(T) * ss.skew(X) * sigma_mm1 / (3 * kappa_mm1)
c_mm1 = np.mean(X) / T - theta_mm1
print(
    "Estimated parameters: \n\n c={} \n theta={} \n sigma={} \n \
kappa={}\n".format(
        c_mm1, theta_mm1, sigma_mm1, kappa_mm1
    )
)
print("Estimated c + theta = ", c_mm1 + theta_mm1)
def log_likely_NIG(x, data, T):
    return (-1) * np.sum(np.log(NIG_density(data, T, x[0], x[1], x[2], x[3])))
result_NIG = minimize(
    log_likely_NIG,
    x0=[c_mm1, theta_mm1, sigma_mm1, kappa_mm1],
    method="L-BFGS-B",
    args=(X, T),
    tol=1e-8,
    bounds=[[-1, 1], [-1, -1e-15], [1e-15, 2], [1e-15, None]],
)
print("Number of iterations performed by the optimizer: ", result_NIG.nit)
print("MLE parameters: ", result_NIG.x)
