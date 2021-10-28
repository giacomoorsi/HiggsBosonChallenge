from matplotlib import pyplot as plt


def plot_degree_errors_plt(degrees, lambdas, results):
    plt.figure(figsize=(8, 6), dpi=80)
    for i, deg in enumerate(degrees):
        accuracy = results[i]
        accuracy[accuracy == 0] = 'nan'
        plt.plot(lambdas, accuracy, label=f"{deg}")
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylim([0.7, 0.9])
    plt.ylabel('Accuracy')
    leg = plt.legend(loc='best', title="Degrees")
    leg.get_frame().set_alpha(0.3)
