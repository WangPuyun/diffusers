# Flux2 Klein loss-weight autoresearch

This harness searches the relative weights of FM-MSE, velocity cosine, and
high-frequency residual losses. `candidate.json` is the only file that the
autoresearch loop may edit.

## Start

Gradient calibration is not required. When the two calibration weights in
`study.json` are `null`, `candidate.json` values are used directly as the raw
cosine and HF loss weights. Copy the complete contents of
`AUTORESEARCH_PROMPT.txt` into Codex to start. Autoresearch runs the pure-MSE
configuration as Iteration 0 and then performs 25 candidate iterations.

Successful
trials are immutable and cached under `outputs/autoresearch_loss_search/`.
Training and evaluation logs never appear on the Verify command's stdout; its
only successful stdout line is the objective value.

The same ten image pairs are used for search and reporting. Results therefore
describe this fixed development set and are not evidence of held-out
generalization. FID and KID are recorded as exploratory diagnostics only.
When either image has no detected face, the registered ArcFace aggregation
policy assigns that pair `-1`; face-pair counts remain available in every JSON
result so this behavior is explicit.
