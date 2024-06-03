# Graph Association Motion-aware Tracker for tiny object in satellite videos
Open sourcing of source code and results is ongoing.



## Data Preparation
Download SV248S dataset from the official website [[SV248S]](https://github.com/xdai-dlgvv/SV248S)
The dataset should look like this:
   ```
   ${data_root}
├── 01_000000
├── 01_000001
├── 01_000002
...
├── 06_000045
├── 06_000046
├── 06_000047
└── list.txt
   ```
modify the `root` to the `${data_root}` in demo_fast.py
The `list.txt` file contains sequence names of the datasets.
Simplely create it with
`ls > list.txt`

We provide scripts to link original SV248S dataset to this data structure.

## Evaluation

```
python demo_fast.py
```

## Visualization or Debug 
modify the demo_fast.py and set `use_time` to `False` 
