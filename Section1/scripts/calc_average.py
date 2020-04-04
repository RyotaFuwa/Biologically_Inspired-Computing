#! /usr/bin/env python

import pandas as pd
import os
import sys

tmpdir = sys.argv[1]
tmpfiles = os.listdir(tmpdir)
file_locs = []
for tmpfile in tmpfiles:
    file_locs.append(os.path.join(tmpdir, tmpfile))

data_pack = pd.read_csv(file_locs[0])
for filename in file_locs[1:]:
    data_pack += (pd.read_csv(filename))
data_pack /= len(file_locs)
data_pack.to_csv(sys.argv[2])

    
    

