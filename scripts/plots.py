from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
matplotlib.rcParams.update({'font.size': 22})


def plot_degree_errors_plt(degrees, lambdas, results, title, filename):
    """
    Create and save a plot to visualize degrees as lines which relate lambdas with accuracy results.

    :param degrees: array of degrees analyzed.
    :param lambdas: array of lambdas analyzed.
    :param results: matrix of classification results.
    """
    plt.figure(figsize=(8, 6), dpi=80)
    for i, deg in enumerate(degrees):
        accuracy = results[i]
        accuracy[accuracy == 0] = 'nan'
        plt.plot(lambdas, accuracy, label=f"{deg}")
    plt.xscale('log')
    plt.title(title, pad=20)
    plt.xlabel('$\lambda$')
    plt.ylim([0.7, 0.9])
    plt.ylabel('Accuracy')
    leg = plt.legend(loc=8, title="Degrees", ncol=3)
    leg.get_frame().set_alpha(0.3)
    
    plt.savefig(filename + '.pgf')
    plt.show()

