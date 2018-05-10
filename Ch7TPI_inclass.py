import numpy as np
import scipy.optimize as opt
import pickle

#read pickle file (allows to save to disk whilst preserving python types)
demog_obs = pickle.load(open('demographic_objects.pkl', 'rb'))
omega_path_S = demog_obs['omega_path']
omega_S_preTP = demog_obs['omega_preTP']
g_n_path = demog_obs['g_n_path']
imm_rate_path = demog_obs['imm_rates']

rho = demog_obs['rho']
omega = demog_obs['omega_SS']
g_n = demog_obs['g_n_SS']
imm_rate = demog_obs['imm_rates'][-1, :]
g_y = 0.04


# define parameters
econ_life = 80  # number of years from begin work until death
frac_work = 4 / 5  # fraction of time from begin work until death
# that model agent is working
S = 80  # number of model periods
alpha = 0.3
A = 1
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** np.round(econ_life / S))
beta_annual = 0.96
beta = beta_annual ** np.round(econ_life / S)
sigma = 2.0
nvec = np.empty(S)
retire = int(np.round(frac_work * S)) + 1
nvec[:retire] = 1.0
nvec[retire:] = 0.2


def get_K(bvec, omega, g_n, imm_rate):
    '''
    Compute the aggregate capital stock
    '''
    K = (( 1 / (1 + g_n)) * (omega[:-1] * bvec + imm_rate [1:] * omega[1:] * bvec)).sum()
    return K


def get_L(nvec, omega):
    '''
    Compute aggregate labor
    '''
    L = (omega * nvec).sum()

    return L

def get_Y(A, K, L, alpha):
    Y = A * (K ** alpha) * (L ** (1-alpha))
    return Y

def get_BQ(bvec, omega, rho, g_n, r):
    BQ = ((1 + r) / (1 + g_n)) * ((rho[:-1] * omega[:-1] * bvec)).sum()
    return BQ

def get_r(nvec, bvec, omega, g_n, A, imm_rate, alpha, delta):
    K = get_K(bvec, omega, g_n, imm_rate)
    L = get_L(nvec, omega)
    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r

def get_w(nvec, bvec, omega, g_n, A, imm_rate, alpha, delta):
    K = get_K(bvec, omega, g_n, imm_rate)
    L = get_L(nvec, omega)
    w = (1 - alpha) * A * (K / L) ** alpha
    return w

def get_c(nvec, bvec, g_y, r, w, omega, rho, g_n):
    b_s = np.append(0.0, bvec)
    b_splus1 = np.append(bvec, 0.0)
    BQ = get_BQ(bvec, omega, rho, g_n, r)
    cvec = (1 + r) * b_s + w * nvec + BQ - b_splus1 * np.exp(g_y)
    return cvec

def u_prime(cvec, sigma):
    MUc = cvec ** (- sigma)
    return MUc

def euler_system(bvec, *args):
    (nvec, omega, imm_rate, rho, g_y, g_n, beta, sigma, A, alpha, delta) = args
    r = get_r(nvec, bvec, omega, g_n, A, imm_rate, alpha, delta)
    w = get_w(nvec, bvec, omega, g_n, A, imm_rate, alpha, delta)
    cvec = get_c(nvec, bvec, g_y, r, w, omega, rho, g_n)
    euler_errors = u_prime(cvec[:-1], sigma) - (beta * np.exp(g_y * sigma) *
                    (1 - rho[:-1]) * (1+r) * u_prime(cvec[1:], sigma))
    return euler_errors

def get_r_path(L, K, A, alpha, delta):
    rpath = alpha * A * (L / K) ** (1 - alpha) - delta
    return rpath

def get_w_path(L, K, A, alpha):
    wpath = (1 - alpha) * A * (K / L) ** alpha
    return wpath

def get_Lpath (nvec, omega_path):
    L_path = (omega_path * nvec).sum(axis=1)  # create Lmat and Lpath is sum at axis=1
    return L_path

def euler_sys_tpi(guesses, *args):
    bvec = guesses
    (rpath, wpath, nvec, b_init, BQpath, beta, sigma, g_y, omega, rho) = args
    b_s_vec = np.append(b_init, bvec)
    b_sp1_vec = np.append(bvec, 0.0)
    cvec = (1 + rpath) * b_s_vec + wpath * nvec + BQpath - b_sp1_vec * np.exp(g_y)
    eulererrors = (u_prime(cvec[:-1], sigma) -
                 beta * np.exp(g_y * sigma) * (1 - rho[:-1]) *
                 (1 + rpath[1:]) * u_prime(cvec[1:], sigma))
    return eulererrors

# initial value for savings distribution

bvec_init = np.ones((S - 1)) * 0.01
eul_args = (nvec, omega, imm_rate, rho, g_y, g_n, beta, sigma,
            A, alpha, delta)
results = opt.root(euler_system, bvec_init, args=(eul_args), tol=1e-14)
b_ss = results.x
zero_val = results.fun
print('The SS savings are: ', b_ss, ' the errors are: ', zero_val)
r_ss = get_r(nvec, b_ss, omega, g_n, A, imm_rate, alpha, delta)
w_ss = get_w(nvec, b_ss, omega, g_n, A, imm_rate, alpha, delta)
K_ss = get_K(b_ss, omega, g_n, imm_rate)
BQ_ss = get_BQ(b_ss, omega, rho, g_n, r_ss)
L_ss = get_L(nvec, omega)
Y_ss = get_Y(A, K_ss, L_ss, alpha)
print('The SS interest rate is: ', r_ss, ' Annual rate of: ',
      (1 + r_ss) ** (1 / (econ_life / S)) - 1)
print('The SS wage rate is: ', w_ss)
print('The SS capital stock is: ', K_ss)
print('The SS labor supply is: ', L_ss)
print('The SS aggregate output is: ', Y_ss)
print('The SS bequest is: ', BQ_ss)

'''
Starting the time path solution
'''
bvec1 = 0.9 * b_ss
T = 400 - S + 2
xi = 0.2

# create a initial vector for aggregate capital stock

K1 = get_K(bvec1, omega, g_n, imm_rate)
Kpath = np.linspace(K1, K_ss, num=T)
Kpath = np.append(Kpath, K_ss * np.ones(S-2))

# create a initial vector for Bequest

BQ1 = 0.9 * BQ_ss
BQpath = np.linspace(BQ1, BQ_ss, num=T)
BQpath = np.append(BQpath, BQ_ss * np.ones(S-2))

Lpath = get_Lpath(nvec, omega_path_S)

dist_tot = 10
tpi_iter = 0
tpi_max_iter = 100
tpi_tol = 1e-6
while (dist_tot > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1
    Lpath = get_Lpath(nvec, omega_path_S)
    rpath = get_r_path(Lpath, Kpath, A, alpha, delta)
    wpath = get_w_path(Lpath, Kpath, A, alpha)
    bmat = np.zeros((T + S - 2, S - 1))
    bmat[0, :] = bvec1
    for p in range(1, S - 1):
        bguess = np.diag(bmat[:p, -p:])
        b_args = (rpath[:p + 1], wpath[:p + 1], nvec[-p - 1:], bvec1[-p - 1],
                  BQpath[:p + 1], beta, sigma, g_y, omega, rho[-p - 1:])
        results = opt.root(euler_sys_tpi, bguess, args=(b_args))
        bvec = results.x
        diagmask = np.eye(p, dtype=bool)
        bmat[1:p + 1, -p:] = diagmask * bvec + bmat[1:p + 1, -p:]

    for t in range(1, T):
        bguess = np.diag(bmat[t - 1: t + S - 1, :])
        b_args = (rpath[t - 1: t + S - 1], wpath[t - 1: t + S - 1],
                  nvec, 0.0, BQpath[t - 1: t + S - 1], beta, sigma, g_y, omega, rho)
        results = opt.root(euler_sys_tpi, bguess, args=(b_args))
        bvec = results.x
        diagmask = np.eye(S - 1, dtype=bool)
        bmat[t:t + S - 1, :] = diagmask * bvec + bmat[t:t + S - 1, :]

    omega_pathmin1 = np.vstack((omega_S_preTP, omega_path_S))

    Kprime = (( 1 / (1 + g_n)) * (omega_pathmin1[:-1, :-1] * bmat +
              imm_rate [1:] * omega_pathmin1[:-1, 1:] * bmat)).sum(axis = 1)
    BQprime = ((1 + rpath) / (1 + g_n)) * ((rho[:-1]
              * omega_pathmin1[:-1, :-1] * bmat)).sum(axis = 1)
    dist_tot = ((Kpath[:T] - Kprime[:T]) ** 2 + (BQpath[:T] - BQprime[:T]) ** 2).sum()

    print('Distance at iteration ', tpi_iter, ' is ', dist_tot)

    Kpath[:T] = xi * Kprime[:T] + (1 - xi) * Kpath[:T]
    BQpath[:T] = xi * BQprime[:T] + (1 - xi) * BQpath[:T]

print('Kpath = ', Kpath)
print('BQpath = ', BQpath)
import matplotlib.pyplot as plt
plt.plot(rpath)
plt.show()
