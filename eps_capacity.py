import itertools as it

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from random_phases import ergodic_capac_approximation, ergodic_capac_exact
from utils import export_results

def main(num_elements, connect_prob, plot=False, export=False, **kwargs):
    # Definir un conjunto de valores de tolerancia a interrupciones (epsilon) en escala logarítmica
    eps = np.logspace(-9, -1, 200)
    
    # Iterar sobre las combinaciones de número de elementos y probabilidad de conexión
    for _num_el, _conn_prob in it.product(num_elements, connect_prob):
        results = {"eps": eps}
        print("Trabajando en N={:d}, p={:.3f}".format(_num_el, _conn_prob))
        
        # Crear una distribución binomial con el número de elementos y probabilidad de conexión dados
        binom = stats.binom(n=_num_el, p=_conn_prob)
        
        # Calcular la probabilidad mínima de interrupción
        print("Probabilidad mínima de interrupción: {:.3E}".format(binom.cdf(0)))
        
        # Calcular el número de enlaces activos para cada valor de tolerancia a interrupciones
        _active_links = binom.ppf(eps)
        
        # Calcular la capacidad de tolerancia a interrupciones utilizando aproximación y exactitud
        eps_cap_appr = [ergodic_capac_approximation(_n, los_amp=0.) for _n in _active_links]
        eps_cap_exact = [ergodic_capac_exact(_n, los_amp=0.) for _n in _active_links]
        
        # Calcular la capacidad óptima de tolerancia a interrupciones
        eps_cap_optim = np.log2(1 + _active_links**2)
        
        # Almacenar los resultados en el diccionario
        results["exact"] = eps_cap_exact
        results["approx"] = eps_cap_appr
        results["optimal"] = eps_cap_optim
        
        # Exportar resultados si se solicita
        if export:
            export_results(results, "eps-capac-N{:d}-p{:.3f}.dat".format(_num_el, _conn_prob))
        
        # Graficar si se solicita
        if plot:
            plt.semilogx(eps, eps_cap_exact, label="N={:d}, p={:.3f} (Exacto)".format(_num_el, _conn_prob))
            plt.semilogx(eps, eps_cap_appr, '--', label="N={:d}, p={:.3f} (Aproximado)".format(_num_el, _conn_prob))
            plt.semilogx(eps, eps_cap_optim, '.-', label="N={:d}, p={:.3f} (Óptimo)".format(_num_el, _conn_prob))
    
    # Mostrar leyenda y etiquetas de ejes si se solicitó la visualización
    if plot:
        plt.legend()
        plt.xlabel("Probabilidad Tolerada de Interrupción $\\varepsilon$")
        plt.ylabel("$\\varepsilon$-Capacidad de Interrupción $R^{\\varepsilon}$")

# Sección principal del script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")  # Habilitar la visualización de gráficos
    parser.add_argument("--export", action="store_true")  # Habilitar la exportación de resultados
    parser.add_argument("-N", "--num_elements", type=int, nargs="+", required=True)  # Número de elementos
    parser.add_argument("-a", "--los_amp", type=float, default=0.)  # Amplitud de la línea de visión
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])  # Probabilidad de conexión
    args = vars(parser.parse_args())
    main(**args)
    plt.show()  # Mostrar los gráficos si se solicitó la visualización
