.PHONY: test toy stage1 stage2 stage3 stage4 analyze
test: ; pytest -q
toy:  ; python scripts/06_toy.py --style pythia --arm control --seed 0 && python scripts/06_toy.py --style pythia --arm component:2 --seed 0 && python scripts/06_toy.py --style pythia --arm bias:1.0 --seed 0 && python scripts/05_analyze_interventions.py --model toy --revision pythia --indir results/06_toy
stage1: ; python scripts/01_checkpoint_sweep.py --models EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b --checkpoints coarse
stage2: ; python scripts/02_activations.py --model EleutherAI/pythia-70m --revisions step8000 step32000 step143000 --device cuda
stage3: ; python scripts/03_ablation.py --model EleutherAI/pythia-70m --device cuda
stage4: ; for s in 0 1 2 3 4; do for arm in control component:-1 component:2 bias:0.5:random bias:1:random bias:2:random; do python scripts/04_intervene.py --model EleutherAI/pythia-70m --revision step8000 --arm $$arm --seed $$s --bf16; done; done
analyze: ; python scripts/05_analyze_interventions.py --model EleutherAI/pythia-70m --revision step8000
