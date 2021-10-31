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


def plot_degree_errors_plt(degrees, lambdas, results):
    plt.figure(figsize=(8, 6), dpi=80)
    for i, deg in enumerate(degrees):
        accuracy = results[i]
        accuracy[accuracy == 0] = 'nan'
        plt.plot(lambdas, accuracy, label=f"{deg}")
    plt.xscale('log')
    plt.title("least squares on jet0")
    plt.xlabel('$\lambda$')
    plt.ylim([0.7, 0.9])
    plt.ylabel('Accuracy')
    leg = plt.legend(loc='best', title="Degrees")
    leg.get_frame().set_alpha(0.3)
    plt.savefig('plot-.pgf')

