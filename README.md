# EMC
Expand-Maxize-Compress algorithm for merging x-ray single particle imaging snapshots based on OpenCL

## Installation
To install run:
```
$ git clone https://github.com/andyofmelbourne/EMC.git && cd EMC && pip install -e .
```

To install dependencies through conda, run:
```
$ conda env create -f environment.yml -n EMC
$ conda activate EMC
```

branch: merge_in_I

Try summing into I-space   : sum_d P_dr K_di / C_i
and sum into to the overlap: sum_d P_dr sum_i K_di / (sum_i Wold_ri)

compare the log-likelihood with this update, compared to old one.
might be a bad idea early in the reconstruction, since this update will be less smooth.
also might need a low beta factor when using this
