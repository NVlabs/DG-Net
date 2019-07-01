## Evaluation

- For Market-1501
```bash
python test_2label.py --name E0.5new_reid0.5_w30000  --which_epoch 100000 --multi
```
The result is `Rank@1:0.9477 Rank@5:0.9798 Rank@10:0.9878 mAP:0.8609`.
`--name` model name 

`--which_epoch` selects the i-th model

`--multi` extracts and evaluates the multiply query. The result is `multi Rank@1:0.9608 Rank@5:0.9860 Rank@10:0.9923 mAP:0.9044`.
