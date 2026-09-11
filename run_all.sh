#!/usr/bin/env bash
# run_all.sh — pull the repo and run the experiment pipeline on a single GPU workbench.
#
#   bash run_all.sh quick        # ~1 h: tests, toy, Stage 1 (70m only), Stage 2/3 on 70m, ONE seed of Stage 4, analysis
#   bash run_all.sh full         # ~1 day: all sizes, 5 seeds x 6 arms of Stage 4, analysis
#   bash run_all.sh stage4       # any single stage: tests|toy|stage1|stage2|stage3|stage4|analyze
#
# Resumable: every stage skips work whose output file already exists. Logs go to logs/<stage>.log.
# Run inside tmux/nohup for the long stages:  nohup bash run_all.sh full > run_all.log 2>&1 &
#
# Overridable via environment:
#   REPO_URL   git URL to clone (default: the Minds-R-Lab repo)      REPO_DIR  checkout dir (default: rwloop)
#   MODEL      main model for stages 2-4 (default EleutherAI/pythia-70m)
#   SEEDS      seeds for stage 4 (default "0 1 2 3 4"; quick mode uses "0")
#   ARMS       arms for stage 4 (default "control component:-1 component:2 bias:0.5:random bias:1:random bias:2:random")
#   STEPS      continued-pretraining steps per arm (default 8000; quick 2000)
#   DATASET    HF dataset with a `text` column (default monology/pile-uncopyrighted)
set -euo pipefail

MODE="${1:-quick}"
REPO_URL="${REPO_URL:-https://github.com/Minds-R-Lab/alignment_checkpoint_dynamics.git}"
REPO_DIR="${REPO_DIR:-rwloop}"
MODEL="${MODEL:-EleutherAI/pythia-70m}"
DATASET="${DATASET:-monology/pile-uncopyrighted}"
ARMS="${ARMS:-control component:-1 component:2 bias:0.5:random:frozen bias:1:random:frozen bias:2:random:frozen bias:1:random}"
if [ "$MODE" = "quick" ]; then SEEDS="${SEEDS:-0}"; STEPS="${STEPS:-2000}"; SIZES="EleutherAI/pythia-70m";
else SEEDS="${SEEDS:-0 1 2 3 4}"; STEPS="${STEPS:-8000}"; SIZES="EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b"; fi
MTAG="$(echo "$MODEL" | tr '/' '_')"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- 0. checkout + environment
if [ -d "$REPO_DIR" ]; then
  if [ -d "$REPO_DIR/.git" ]; then log "updating $REPO_DIR"; (cd "$REPO_DIR" && git pull --ff-only || true)
  else log "using existing $REPO_DIR (unzipped, not a git checkout)"; fi
elif [ -f rwloop.zip ]; then log "unzipping rwloop.zip"; unzip -q rwloop.zip
else log "cloning $REPO_URL"; git clone "$REPO_URL" "$REPO_DIR"; fi
cd "$REPO_DIR"; mkdir -p logs results data_cache
if [ ! -d .venv ]; then
  log "creating venv"; python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import rwloop, zstandard" 2>/dev/null || { log "installing"; pip install -q --upgrade pip; pip install -q -e . pytest zstandard; }
# --- make sure torch can actually see the GPU (a wheel built for a newer CUDA than the driver falls back to CPU)
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DRV=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | grep -o '[0-9]*\.[0-9]*' | head -1)
    MAJ=${DRV%%.*}; MIN=${DRV##*.}
    if   [ "$MAJ" -ge 13 ]; then TAG=cu130
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then TAG=cu128
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then TAG=cu126
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ]; then TAG=cu124
    else TAG=cu121; fi
    log "torch cannot see the GPU (driver CUDA $DRV); reinstalling torch from the $TAG wheel index"
    pip uninstall -y -q torch; pip install -q torch --index-url "https://download.pytorch.org/whl/$TAG"
  fi
fi
python - << 'EOF'
import torch; print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
EOF
DEVICE=$(python -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")
if [ "$DEVICE" = "cpu" ] && [ "$MODE" != "tests" ] && [ "$MODE" != "toy" ] && [ "$MODE" != "stage1" ]; then
  log "WARNING: running without a GPU; stages 2-4 will be very slow"
fi

run_stage() {   # run_stage <name> <command...>
  local name="$1"; shift
  log "=== $name ==="; "$@" 2>&1 | tee -a "logs/$name.log"
}

# ---------------------------------------------------------------- stages
do_tests()  { run_stage tests python -m pytest -q; }

do_toy()    { for arm in control component:2 bias:1.0; do
                f="results/06_toy/toy_pythia_${arm//:/_}_seed0.pkl"; [ -f "$f" ] && continue
                run_stage toy python scripts/06_toy.py --style pythia --arm "$arm" --seed 0 --threads 8; done
              run_stage toy python scripts/05_analyze_interventions.py --model toy --revision pythia --indir results/06_toy; }

do_stage1() { [ -f results/01_checkpoint_sweep/summary.csv ] && log "skip pythia sweep (summary.csv exists)" || \
                  run_stage stage1 python scripts/01_checkpoint_sweep.py --models $SIZES --checkpoints coarse
              # gated family with public checkpoints (sign question); skip silently if revisions are unavailable
              run_stage stage1 python scripts/01_checkpoint_sweep.py --models allenai/OLMo-2-0425-1B \
                  --revisions stage1-step10000-tokens21B stage1-step50000-tokens105B stage1-step150000-tokens315B main || true; }

do_stage2() { run_stage stage2 python scripts/02_activations.py --model "$MODEL" --revisions step1000 step2000 step4000 step8000 step143000 \
                  --device "$DEVICE" --dataset "$DATASET"
              [ "$MODE" = "full" ] && run_stage stage2 python scripts/02_activations.py --model EleutherAI/pythia-410m \
                  --revisions step8000 step32000 step143000 --device "$DEVICE" --dataset "$DATASET" || true; }

do_stage3() { [ -f "results/03_ablation/${MTAG}_main.json" ] || run_stage stage3 python scripts/03_ablation.py --model "$MODEL" --device "$DEVICE" --dataset "$DATASET"
              [ "$MODE" = "full" ] && { [ -f results/03_ablation/EleutherAI_pythia-410m_main.json ] || \
                  run_stage stage3 python scripts/03_ablation.py --model EleutherAI/pythia-410m --device "$DEVICE" --dataset "$DATASET"; } || true; }

do_stage4() { for s in $SEEDS; do for arm in $ARMS; do
                f="results/04_intervene/${MTAG}_step8000_${arm//:/_}_seed${s}.pkl"
                if [ -f "$f" ]; then log "skip $arm seed $s (exists)"; continue; fi
                run_stage stage4 python scripts/04_intervene.py --model "$MODEL" --revision step8000 --arm "$arm" --seed "$s" \
                    --steps "$STEPS" --bf16 --device "$DEVICE" --dataset "$DATASET"
              done; done; }

do_analyze() { run_stage analyze python scripts/05_analyze_interventions.py --model "$MODEL" --revision step8000
               log "report: results/05_analysis/report.md"; }

# ---- Stage 7: single-layer forward arm (c -> f), removing the cross-layer confound in H4.
# One layer intervened at a time; matched controls are the Stage-4 control_seed*.pkl (same seed).
# Override LAYERS7 (default "1 2 3 4") and ARMS7 (default "component:-1 component:2").
do_stage7() { LAYERS7="${LAYERS7:-1 2 3 4}"; ARMS7="${ARMS7:-component:-1 component:2}"
              for s in $SEEDS; do
                f="results/04_intervene/${MTAG}_step8000_control_seed${s}.pkl"
                if [ ! -f "$f" ]; then
                  run_stage stage7 python scripts/04_intervene.py --model "$MODEL" --revision step8000 --arm control \
                      --seed "$s" --steps "$STEPS" --bf16 --device "$DEVICE" --dataset "$DATASET"
                fi
                for L in $LAYERS7; do for arm in $ARMS7; do
                  g="results/07_single_layer/${MTAG}_step8000_${arm//:/_}_L${L}_seed${s}.pkl"
                  if [ -f "$g" ]; then log "skip $arm L$L seed $s (exists)"; continue; fi
                  run_stage stage7 python scripts/04_intervene.py --model "$MODEL" --revision step8000 --arm "$arm" \
                      --layers "$L" --outdir 07_single_layer --seed "$s" --steps "$STEPS" --bf16 \
                      --device "$DEVICE" --dataset "$DATASET"
                done; done
              done
              run_stage stage7 python scripts/07_single_layer_forward.py --model "$MODEL" --revision step8000
              log "report: results/07_single_layer/report.md"; }

case "$MODE" in
  quick|full) do_tests; do_toy; do_stage1; do_stage2; do_stage3; do_stage4; do_analyze ;;
  tests) do_tests ;; toy) do_toy ;; stage1) do_stage1 ;; stage2) do_stage2 ;; stage3) do_stage3 ;; stage4) do_stage4 ;; analyze) do_analyze ;;
  stage7|singlelayer) do_stage7 ;;
  *) echo "usage: bash run_all.sh [quick|full|tests|toy|stage1|stage2|stage3|stage4|analyze|stage7]"; exit 1 ;;
esac
log "done ($MODE). Results under $(pwd)/results, logs under $(pwd)/logs"
