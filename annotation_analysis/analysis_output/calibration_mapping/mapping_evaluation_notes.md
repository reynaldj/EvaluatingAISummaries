# Mapping evaluation metric definitions

For each dimension and split (`calibration` or `test`), MAE is calculated
between the human rating and the mapped integer prediction:

\[
\operatorname{MAE} = \frac{1}{N}\sum_{i=1}^{N}
\left|y_i - \hat{y}_i\right|,
\]

where \(y_i\) is the human rating, \(\hat{y}_i\) is the mapped prediction,
and \(N\) is the number of rows in that dimension and split. Ratings are on
the 1–5 scale, so MAE is expressed in rating points; 0 means exact agreement
on every row, and lower values indicate smaller average absolute differences.

The `mae` values in `mapping_evaluation.csv` are computed as
`mean(abs(actual - predicted))` in `evaluate()` in
`annotation_analysis/calibrate_ordinal_mappings.py`.
