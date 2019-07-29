## Image Generation 
- `test_folder.py` generates samples and calculates the FID score. (You need to download the [TTUR](https://github.com/layumi/TTUR) for the FID evaluation.)
```bash
python visual_tools/test_folder.py --name E0.5new_reid0.5_w30000 --which_epoch 100000
```

- `show_rainbow.py` generates Figure 1 of the paper.
```bash
python visual_tools/show_rainbow.py
```

- `show_swap.py` swaps the codes on Market-1501. (Figure 6)

- `show_smooth.py` generates Figure 5 of the paper.

- `show_smooth_structure.py` generates Figure 9 of the paper.

- `show1by1.py` swaps the codes of all images in one folder (many to many) and save the generated image one by one. 
