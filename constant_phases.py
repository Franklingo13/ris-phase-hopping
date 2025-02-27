import itertools as it

import numpy as np
from scipy import stats
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt
import hankel

from phases import gains_constant_phase, rvs_channel_phases, rvs_ris_phases
from utils import export_results

# Función de aproximación de probabilidad de interrupción para N enlaces
def outage_n_links_approx(rate, num_elements, los_amp: float = 0.):
    if num_elements == 0:
        cdf_appr = np.where(rate < np.log2(1 + los_amp ** 2), 0, 1)
        return cdf_appr
    if los_amp == 0.:
        cdf_appr = 1. - np.exp(-(2 ** rate - 1) / num_elements)  # N --> oo, for sum
    else:
        cdf_appr = stats.ncx2(df=2, nc=2 * los_amp ** 2 / num_elements).cdf(2 * (2 ** rate - 1) / num_elements)
    return cdf_appr

# Función exacta de probabilidad de interrupción para N enlaces
def outage_n_links_exact(rate, num_elements, los_amp: float = 0.):
    if los_amp != 0:
        raise NotImplementedError("Solo se admite NLOS")
    if num_elements == 0:
        return np.where(rate < np.log2(1 + los_amp ** 2), 0, 1)

    nu = 1
    _int_func = lambda x: special.j0(x) ** num_elements / x
    _h, _, _N = hankel.get_h(_int_func, nu=nu)
    ht = hankel.HankelTransform(nu=nu, N=_N, h=_h)
    s = np.sqrt(2 ** rate - 1)
    Fs, err_hank = ht.transform(_int_func, k=s, ret_err=True)
    return np.clip(s * Fs, 0, 1)

# Función de aproximación de probabilidad de interrupción para fases estáticas
def outage_static_phases_approx(rate, num_elements, connect_prob: float = 1., los_amp: float = 0.):
    _out_approx = [outage_n_links_approx(rate, _links, los_amp=los_amp) for _links in range(num_elements + 1)]
    _out_weights = stats.binom(n=num_elements, p=connect_prob).pmf(range(num_elements + 1))
    cdf_appr = np.average(_out_approx, weights=_out_weights, axis=0)
    return cdf_appr

# Función exacta de probabilidad de interrupción para fases estáticas
def outage_static_phases_exact(rate, num_elements, connect_prob: float = 1., los_amp: float = 0.):
    if los_amp == 0:
        _out_weights = stats.binom(n=num_elements, p=connect_prob).pmf(range(num_elements + 1))
        _out_exact = [outage_n_links_exact(rate, _links, los_amp=los_amp) for _links in range(num_elements + 1)]
        cdf_exact = np.average(_out_exact, weights=_out_weights, axis=0)
    else:
        cdf_exact = np.zeros_like(rate)
    return cdf_exact

# Función para calcular las probabilidades de interrupción con fases RIS constantes
def constant_ris_phases(num_elements, connect_prob=[1.], los_amp=0., num_samples=50000, plot=False, export=False):
    if plot:
        fig, axs = plt.subplots()
    los_phases = 2 * np.pi * np.random.rand(num_samples) if los_amp > 0 else None
    for _num_elements, _conn_prob in it.product(num_elements, connect_prob):
        print("Trabajando en N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
        results = {}
        channel_realizations = rvs_channel_phases(_num_elements, num_samples)
        channel_absolute = stats.bernoulli.rvs(p=_conn_prob, size=(num_samples, _num_elements))
        const_phase = gains_constant_phase(channel_realizations, los_phase=los_phases, los_amp=los_amp,
                                           path_amp=channel_absolute)
        capac_const_phase = np.log2(1 + const_phase)
        _hist = np.histogram(capac_const_phase, bins=1000)
        _r_ax = np.linspace(0.01, max(capac_const_phase) * 1.1, 1000)
        cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
        cdf_appr = outage_static_phases_approx(_r_ax, _num_elements, _conn_prob, los_amp=los_amp)
        cdf_exact = outage_static_phases_exact(_r_ax, _num_elements, _conn_prob, los_amp=los_amp)

        if plot:
            axs.plot(_r_ax, cdf_hist, label="ECDF N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
            axs.plot(_r_ax, cdf_appr, '--', label="Aprox. N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
            axs.plot(_r_ax, cdf_exact, '-.', label="Exacta N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
        if export:
            results["rate"] = _r_ax
            results["ecdf"] = cdf_hist
            results["aproximada"] = cdf_appr
            results["exacta"] = cdf_exact
            _fn_prefix = "prob-interrupcion-fases-constantes"
            _fn_mid = "los-a{:.2f}".format(los_amp) if los_amp > 0 else "nlos"
            _fn_end = "N{:d}-p{:.3f}".format(_num_elements, _conn_prob)
            _fn = "{}-{}-{}".format(_fn_prefix, _fn_mid, _fn_end)
            export_results(results, _fn)
    if plot:
        axs.set_xlabel("Tasa $R$")
        axs.set_ylabel("Probabilidad de Interrupción")
        axs.legend()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-N", "--num_elements", nargs="+", type=int, required=True)
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])
    parser.add_argument("-s", "--num_samples", type=int, default=50000)
    parser.add_argument("-a", "--los_amp", type=float, default=0.)
    args = vars(parser.parse_args())
    constant_ris_phases(**args)
    plt.show()
