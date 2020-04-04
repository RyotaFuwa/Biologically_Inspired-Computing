#! /usr/bin/env python
import os
import os.path as pth
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
Created only for making graph from csv file
Not readable and reusable at all
"""

class GraphGen(object):
    def __init__(self, file_arr):
        if pth.isdir(file_arr[0]):
            files = os.listdir(file_arr[0])
            file_loc = []
            for f in files:
                floc = pth.join(file_arr[0], f)
                file_loc.append(floc)
            file_loc.sort()
            self.file_arr = GraphGen.get_name(file_loc)
            self.data_arr = GraphGen.data_import(file_loc)
        else:
            file_arr.sort()
            self.file_arr = GraphGen.get_name(file_arr)
            self.data_arr = GraphGen.data_import(file_arr)


    @staticmethod
    def get_name(file_arr):
        out_arr = []
        for file in file_arr:
            filename = file.split("/")[-1]
            out_arr.append(filename[:-4]) # not supposed to be readable, but it extract filename
        out_arr.sort()
        return out_arr

    @staticmethod
    def data_import(file_arr):
        out_arr = []
        for file in file_arr:
            if "csv" in file:
                try:
                    out_arr.append(pd.read_csv(file))
                except:
                    out_arr.append(None)
            else:
                out_arr.append(None)
        return out_arr

    def plot_data(self, y, x, legend=False):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for i in range(len(self.data_arr)):
            di = self.data_arr[i]
            if di is not None:
                self.ax.plot(di[x],  di[y], label=self.file_arr[i])
                if di[x].min() < min_x:
                    min_x = di[x].min()
                if di[x].max() > max_x:
                    max_x = di[x].max()
                if di[y].min() < min_y:
                    min_y = di[y].min()
                if di[y].max() > max_y:
                    max_y = di[y].max()
        self.ax.set_xlim( min_x * 0.95, max_x * 1.05)
        self.ax.set_ylim( min_y * 1.05, max_y * 1.05)
        #self.ax.set_xlim( 500, 1000)
        #self.ax.set_ylim( 0, 0.1)
        if(legend):
            self.ax.legend()
        plt.show()
        
    def save_plot(self, outfilename):
        self.fig.savefig(outfilename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_files', metavar='N', type=str, nargs='*', help='csv file names')
    parser.add_argument('--output', dest='out', type=str, nargs=1, help='graph store file name')
    parser.add_argument('-x', dest='x', type=str, nargs=1, help='graph x axis parameter')
    parser.add_argument('-y', dest='y', type=str, nargs=1, help='graph y axis parameter')
    
    arg_list = parser.parse_args()
    gg = GraphGen(arg_list.data_files)

    if arg_list.x:
        try:
            x = arg_list.x[0]
        except:
            print("-x [parameter for xaxis]")
    if arg_list.y:
        try:
            y = arg_list.y[0]
        except:
            print("-y [parameter for yaxis]")
    gg.plot_data(x=x, y=y, legend=True)

    if arg_list.out:
        gg.save_plot(arg_list.out[0])

if __name__ == '__main__':
    main()
