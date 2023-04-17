#%%
import numpy as np
from numba import njit


@njit
def logadd(a, b):
    if a < b:
        a, b = b, a
    return a + np.log(1 + np.exp(b - a))


@njit
def logsub(a, b):
    assert a > b
    return a + np.log(1 - np.exp(b - a))


@njit
def log_integral_opt(n, C, lw, upper):
    """Integral of f(x)

    Parameters
    ----------
    n : int
        Number of players
    C : float
    lw : float
        log(w)
    upper : float
        Upper bound of integration
    """
    w = np.exp(lw)

    def f0():
        a = C + w / n
        b = C + w
        return np.log(a) - np.log(b) + lw / (n - 1)

    def f(x):
        a = w + n * C * (1 - x) + C * x
        b = np.log(n) + np.log(1 - x) + np.log(C + w)
        c = C * x + w
        d = 1 - x
        return np.log(a) - b + (np.log(c) - np.log(d)) / (n - 1)

    a = f0()
    b = f(upper)
    return logsub(b, a)


@njit
def log_integral_x_opt(n, C, lw, upper):
    """Integral of x f(x)

    Parameters
    ----------
    n : int
        Number of players
    C : float
    lw : float
        log(w)
    upper : float
        Upper bound of integration
    """
    w = np.exp(lw)

    def f0():
        return np.log1p(-1 / n) + lw - np.log(C + w) + lw / (n - 1)

    def f(x):
        sa = 1
        a = w * (1 - n * (1 - x)) + C * x
        if a < 0:
            a = -a
            sa = -1
        b = n * (1 - x) * (C + w)
        c = C * x + w
        d = 1 - x
        return np.log(a) - np.log(b) + (np.log(c) - np.log(d)) / (n - 1), sa

    a = f0()
    b, sb = f(upper)
    if sb == 1:
        return logadd(a, b)
    else:
        return logsub(a, b)


@njit
def cutoff_of_w(n, C, lw, tol=1e-8):
    """Find the cutoff value for the n-player game"""
    a = 0
    b = 1
    while b - a > tol:
        c = (a + b) / 2
        if log_integral_opt(n, C, lw, c) > 0:
            b = c
        else:
            a = c
    return c


@njit
def new_lw(n, C, lw, tol=1e-8):
    """Find the new value of log(w)"""
    cutoff = cutoff_of_w(n, C, lw, tol)
    log_r_bar = log_integral_x_opt(n, C, lw, cutoff)
    return log_r_bar * (n - 1)


@njit
def find_w_bin(n, C, tol=1e-8):
    """Find the value of log(w) by binary search"""
    a = -1
    b = 0
    while new_lw(n, C, a, tol) - a <= 0:
        a *= 10
    while b - a > tol:
        m = (a + b) / 2
        if new_lw(n, C, m, tol) - m > 0:
            a = m
        else:
            b = m
    return m


@njit
def find_w_iter(n, C, tol=1e-8):
    """Find the value of log(w) by iteration"""
    lw = 0
    assert new_lw(n, C, lw, tol) < 0
    s = set()
    for _ in range(100):
        nlw = new_lw(n, C, lw, tol)
        if nlw in s or abs(nlw - lw) < tol:
            break
        lw = nlw
        s.add(nlw)
    else:
        print("Timeout")
    return lw


find_w = find_w_bin


@njit
def find_cutoff(n, C, tol=1e-8):
    """Find the cutoff value for the n-player game"""
    lw = find_w(n, C, tol)
    return cutoff_of_w(n, C, lw, tol)


@njit
def solution(n, C, x, tol=1e-8):
    """Compute the solution probabilities for the n-player game"""
    lw = find_w(n, C, tol)
    cutoff = cutoff_of_w(n, C, lw, tol)
    w = np.exp(lw)
    num = C + w
    exp = 1 / (n - 1)
    den = (n - 1) * (1 - x) ** (2 + exp) * (C * x + w) ** (1 - exp)
    return np.where(x < cutoff, num / den, 0)


@njit
def log_efficiency(n, C, tol=1e-8):
    """Compute the efficiency of the n-player game"""
    lw = find_w(n, C, tol)
    return lw + np.log(n)


if __name__ == "__main__":

    #%%
    import matplotlib.pyplot as plt

    PRINT = False
    n = 10000
    all_n = np.arange(100, n, 10)
    exp = 0.9
    all_C = np.array([1 / n**exp for n in all_n])
    all_eff = np.array([log_efficiency(n, C, tol=1e-10) for n, C in zip(all_n, all_C)])
    all_w = np.array([find_w(n, C, tol=1e-10) for n, C in zip(all_n, all_C)])
    all_cutoff = np.array(
        [cutoff_of_w(n, C, lw, tol=1e-10) for n, C, lw in zip(all_n, all_C, all_w)]
    )
    plt.plot(np.log(all_n), all_eff)
    plt.show()
    plt.plot(np.log(all_n), all_w - np.log(all_C))
    plt.show()
    plt.plot(np.log(all_n), all_w - np.log(all_n * all_C))
    plt.show()
    plt.plot(np.log(all_n), all_cutoff)
    plt.show()
    plt.plot(np.log(all_n), np.exp(all_w / all_n))
    plt.show()
    plt.plot(np.log(all_n), all_cutoff / (1 - all_cutoff) / all_n)
    plt.show()
    plt.plot(np.log(all_n), np.exp(all_w) / (1 - all_cutoff) / all_n ** (1 - exp))

    # %%

    PRINT = True
    n = 10000 / 3
    exp = 0.1
    C = 1 / n**exp
    log_efficiency(n, C, tol=1e-10)

    # %%
    n = 10000
    all_n = np.arange(2, n)
    all_C = [1 / n**1.5 for n in all_n]
    PRINT = False
    all_eff = np.array([log_efficiency(n, C, tol=1e-12) for n, C in zip(all_n, all_C)])
    plt.plot(all_n, all_eff)
    # %%
    n = 1900
    log_efficiency(n, 1 / n**1.3)
    # %%
    n = 1901
    log_efficiency(n, 1 / n**1.3)

    # %%
