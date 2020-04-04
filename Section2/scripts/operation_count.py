import re
import os.path
import sys
import matplotlib.pyplot as plt

def main(inputs):
    operations = ("ADD", "SUB", "MUL", "PDIV", "ABS",
        "CUBE", "CBRT", "EXP", "FACT", "INV", "SQUARE",
        "GRT", "INV", "LT", "LOG", "LOG10", "MOD", "MAX2",
        "MAX3", "MIN2", "MIN3", "POW")

    if os.path.isdir(inputs[1]):
        import numpy as np
        os.chdir(inputs[1])
        counts_sum = np.zeros((len(operations),))
        filenames = os.listdir('.')

        for f in filenames:
            with open(f, 'r') as fin:
                text = fin.readline()
                count = operation_counts(operations, text, False, False)
                counts_sum += count
        
        counts_sum =  counts_sum / counts_sum.sum() * 100.0
        fig, ax = plt.subplots()
        print(len(operations), len(counts_sum))
        ax.bar(operations, counts_sum)
        plt.ylabel("Apperance (%)")
        l = ax.get_xticklabels()
        plt.setp(l, rotation=45, horizontalalignment='right')
        plt.show()
        
    elif type(inputs[1]) is str:
            text = inputs[1]
            operation_counts(operations, text)
    else:
        pass


def operation_counts(ops, text, graph=True, print=False):
    operations = ops
    counts = []
    sum = 0
    for op in operations:
        c = len(re.findall(op, text))
        sum += c
        counts.append(c)

    percentages = []
    for o, c in zip(operations, counts):
        p = c / sum * 100
        percentages.append(p)
        if print:
            print("{}: {:.3} %".format(o, p))

    if graph:
        plt.bar(operations, percentages)
    
    return counts

main(sys.argv)