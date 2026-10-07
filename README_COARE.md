# PestNet baseline from the uploaded GitHub archive, prepared for COARE

This package runs the baseline defined by `cfg/hsdpa.yaml` and the custom Ultralytics source in the supplied `HSDPA-main(2).zip`. All 252 retained archived files are preserved byte for byte under `source/HSDPA-main/`. The two unused original weight-pointer files are omitted, as recorded in `SOURCE_MANIFEST.json`. SHA-256 verification checks the source and the executable runtime before each job. Existing model behavior, bugs, pooling, model scale, layer repeats, padding, parser, losses, and checkpoint implementation are preserved. No CR-HSDPA, channel reduction, paper-derived correction, or architecture substitution is applied.

The full training defaults in `configs/baseline_train.yaml` are 200 epochs, batch size 32, and no pretraining. Batch size 32 follows the user’s instruction; the repository README itself lists batch size 16. Epochs and no pretraining follow the README. The YAML's single `ex` scale stays `[0.33, 1.00, 1024]`. This is a GitHub implementation reproduction. The archive does not contain a complete training command, dataset split files, original environment lock, or author weights, so this package cannot establish every setting used in the published experiments. No paper was used to reconstruct missing choices.

## What is supplied separately

The archive declares Ultralytics 8.3.18 but omits its `ultralytics/cfg/` package, including `default.yaml`. The supplied official 8.3.18 wheel adds only files absent from the archive into `runtime/`; existing source files always take precedence. `runtime/SUPPORT_FILES.json` records every addition and its checksum. The unmodified custom parser, `ultralytics/nn/ext/Blocks.py`, training code, and baseline YAML are used at runtime. The installed standard wheel is needed for distribution metadata; the launcher explicitly imports the verified local runtime and refuses another import path. Do not run a generic installed `yolo` command for this baseline.

Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and the pinned training dependencies are proposed COARE compatibility choices, not a recovered author environment. The CPU validation used Python 3.12.14 and PyTorch 2.4.1+cpu. CUDA 11.8 is the example PyTorch wheel choice; confirm compatibility with the assigned GPU driver. `opencv-python-headless` supplies OpenCV on servers without a desktop. The wheel is installed with `--no-deps` so it does not introduce a second OpenCV package or replace the chosen PyTorch build.

The existing original pooling branches continue to read `x`, and the fourth attention call continues to use `self.sdp(y1)`. The original CBS padding behavior remains intact. Pooling kernel sizes remain 3, 7, and 13. These details describe the preserved implementation and are not changed by the launcher.

The archived `weights/ip102.pt` and `weights/r200.pt` files were 134-byte Git LFS pointers, not usable PyTorch checkpoints. They are omitted from both source and runtime because they cause Git LFS uploads to request absent author-weight objects. Their original checksums are retained as omission metadata. The launcher never needed them: README baseline training starts without pretraining. Actual author weights were not retrieved. The root `.gitignore` excludes new checkpoints and generated outputs. The root `.gitattributes` preserves archived bytes through Git checkout, so checksum verification survives Windows/Git line-ending conversion.

## Step 1: Upload and connect

Download the ZIP to your laptop and use the hostname and username supplied by COARE. In Windows PowerShell, first connect with your COARE key. Replace `USERNAME` and `HOST`; neither is known from the account-application screenshot.

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_coare" USERNAME@HOST
```

On COARE, create a folder in your assigned project storage. Exit SSH and upload the ZIP from the laptop, using the actual destination path. The commands below show a home-folder example only; follow COARE's instructions about where your group should keep data and outputs.

```bash
mkdir -p ~/thesis
exit
```

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519_coare" ".\PestNet_COARE_GitHub_Baseline.zip" "USERNAME@HOST:~/thesis/"
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_coare" USERNAME@HOST
```

Unzip into the intended shared filesystem location. Code, the Conda environment, dataset, and outputs must be visible from the allocated compute node. The package writes outputs and logs beside its source, so extract it where experiment outputs are permitted. No dataset, pretrained target-model weights, or synthetic test weights are included.

```bash
cd ~/thesis
unzip PestNet_COARE_GitHub_Baseline.zip
cd PestNet_COARE_GitHub_Baseline
```

## Step 2: Obtain the account-specific resource values

Use the login node for file management and Slurm submission. Check the actual account association and available modules/partitions. Do environment installation and ML execution through the supplied jobs.

```bash
sacctmgr show assoc user=$USER -P
sinfo
module avail
cp coare/coare.env.example coare/coare.env
nano coare/coare.env
```

Fill `CONDA_MODULE`, `ENV_PREFIX`, `DATA_YAML`, both partition/QOS pairs, and an account if required. Use the exact granted values. The example requests one GPU, eight CPU cores, and 32 GB of host RAM; these are adjustable resource requests, not confirmed account limits. Set `TRAIN_TIME` within the granted QOS limit. Put the Conda prefix under the scratch-3 path COARE assigns. The scripts reject unfilled placeholders. Keep important results backed up because scratch storage can be purged.

## Step 3: Submit environment setup

Run this from the package folder. Setup creates a dedicated Conda environment, installs dependencies, installs the support wheel, and verifies the runtime. It does not run model training. Package downloads require permitted access from the setup node; if unavailable, arrange an offline dependency cache with COARE. An existing environment prefix is rejected so a rebuild cannot overwrite it accidentally.

```bash
bash coare/submit.sh setup
squeue -u $USER
```

Inspect the returned setup job's files in `logs/`. Do not submit training until setup finishes successfully. `outputs/environment_JOBID.txt` captures the installed package versions. Installation and Python work execute on the allocated node.

## Step 4: Retarget the existing IP102 dataset YAML

Upload your existing YOLO-formatted images, labels, and dataset YAML. Preserve your own split membership and class ordering. The native dataloader accepts `images/train` and `images/val` with matching `labels/train` and `labels/val`. This package does not split, relabel, fix, or download IP102. It does not rename an existing test set to validation automatically.

Create a separate COARE YAML with an absolute Linux dataset root. Run the following within an allocated setup/interactive compute job with the environment activated, or edit a separate copy of your YAML manually. Replace the paths with your actual dataset paths. The helper requires the real 102-class names mapping, retains existing train/val/test entries, checks their existence, and refuses Windows paths. Image-list text files must also contain valid Linux paths; retargeting the YAML does not rewrite their contents.

```bash
python coare/prepare_data.py \
  --input /YOUR/DATASET/IP102/data.yaml \
  --root /YOUR/DATASET/IP102 \
  --output /YOUR/DATASET/IP102/data_coare.yaml
```

Set `DATA_YAML` in `coare/coare.env` to that output. If you edit manually, keep the existing `names`, `train`, `val`, and optional `test` values, and change only the environment-specific paths. No species names or class-order assumptions are invented. This package targets the YAML's 102-class IP102 model.

## Step 5: Check the allocated GPU and model

Submit the short construction/forward check. It verifies CUDA availability, imports the custom implementation, verifies source checksums, constructs the baseline, reports unfused training and fused inference parameter counts, and performs 640 × 640 forward checks. Inference fusion is checked only on a separate model copy and is never applied to the training model. It does not need the dataset.

```bash
bash coare/submit.sh check
squeue -u $USER
```

The unchanged model has 79,660,747 parameters for training before fusion. Its inference model has 79,638,859 parameters after the original framework’s native Conv/BatchNorm fusion: exactly 79.638859 M. This is 79.63 M if truncated to two decimals, or 79.64 M when rounded normally. The requested approximate 79.63 M inference count therefore does not require removing model parameters. Batch size changes do not change parameter count. The warning about assuming scale `ex` comes from the unchanged parser and is expected for the archived model filename.

## Step 6: Run the small real-data smoke job

The smoke job selects at most 32 existing training images and eight existing validation images into separate text lists under its own output folder. It retains the same 102-class mapping. It runs one epoch with batch size 32 and zero dataloader workers to check the actual dataset, GPU execution, native loss, validation, and checkpoint output. The one-epoch duration and zero-worker setting apply only to the smoke job. Batch size is 32 for both smoke and full training. A dataset with fewer than 32 training images produces a smaller actual batch, so use at least 32 valid training images when checking GPU memory. The job does not rearrange your dataset.

```bash
bash coare/submit.sh smoke
squeue -u $USER
```

Check both log files for dataset corruption, label issues, CUDA errors, memory use, and successful completion. Native framework code may download plot fonts and the small `yolo11n.pt` reference used by its AMP diagnostic. That reference does not initialize the PestNet model. If the node cannot access those assets, stage them as COARE permits or inspect the native diagnostic messages. No AMP patch is included.

## Step 7: Submit the baseline training

After the GPU smoke job succeeds, submit the full baseline. It builds the archived YAML from scratch with 200 epochs, batch size 32, and `pretrained=False`. Unspecified training settings come from the matching 8.3.18 framework defaults, including 640 image size, optimizer `auto`, seed 0, deterministic mode, native augmentations, AMP, and patience 100. The patience setting can stop training before epoch 200; it has not been changed. The author's complete run configuration is unavailable in this archive.

```bash
bash coare/submit.sh train
squeue -u $USER
```

Slurm returns a job ID. Training proceeds on COARE after SSH disconnects. Results go to `outputs/runs/baseline_JOBID/`, weights to its `weights/`, and launch provenance to `outputs/train_JOBID/launch_record.json`. Native `args.yaml` and `results.csv` record the trainer's actual settings and metrics. The package separates different new runs by job ID.

## Step 8: Monitor or cancel a job

Use the actual job ID and log filename. Slurm's allocated wall time ends the job even if training is unfinished; the code is not changed to stop before a timeout. Back up completed checkpoints and results outside purgeable scratch according to your group's storage instructions.

```bash
squeue -u $USER
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f logs/train_JOBID.out
scancel JOBID
```

## Step 9: Resume an interrupted baseline

The unchanged trainer saves native `last.pt` each epoch. Resume uses its stored epoch, optimizer, EMA, and training arguments. The wrapper checks that the checkpoint's architecture matches the archived YAML and rejects checkpoints whose optimizer or epoch state has been stripped. It does not promise restoration of RNG, AMP-scaler state, or dataloader position that this trainer does not explicitly save. Progress from an unfinished epoch is lost.

```bash
bash coare/submit.sh resume /ACTUAL/RUN/PATH/weights/last.pt
```

Native resume restores the previous run's output location and checkpoint settings, with the native batch override explicitly set to the requested 32. Resuming a checkpoint originally made with batch size 16 therefore changes its batch size to 32; use a checkpoint already trained at 32 for consistent experiment settings. Keep the original paths valid on COARE. The supplied `DATA_YAML` can replace a missing old YAML through the native resume handling. Finished runs normally have their final `last.pt` and `best.pt` stripped by the original trainer and cannot be used for full-state resume. Use an unstripped checkpoint from an interrupted run.

## Optional native periodic checkpoints

The default keeps the original framework's `save_period=-1`; no custom 50-epoch callback is added. If you later want periodic snapshots, the launcher exposes `--save-period 50`. Add that argument to the training invocation in `coare/job.sh` or run the launcher inside a permitted GPU allocation. The original code uses zero-based epoch indexing, so this produces `epoch0.pt`, `epoch50.pt`, `epoch100.pt`, etc. after completed epochs 1, 51, 101, etc. That indexing has deliberately not been fixed. Periodic checkpoint files are separate from the stripped final last/best files.

## Validation and limits

See `VALIDATION.md` for the checks performed here. The COARE account is not provisioned in this workspace, and there is no assigned GPU or IP102 dataset here. No job was submitted to COARE. The package is prepared and locally verified; its CUDA build, resource requests, actual dataset, and Slurm integration still require the first allocated check/smoke jobs.

COARE references used for infrastructure preparation are its [FAQ](https://coarewiki.asti.dost.gov.ph/en/getting-help/faq) and [Acceptable Use Policy](https://coarewiki.asti.dost.gov.ph/en/policies-manuals/aup.pdf). The proposed Torch/torchvision pairing is listed on [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/). Model and training choices above come from the uploaded repository and matching framework, not from the baseline paper.

## Publishing the code to GitHub after a rejected LFS push

Use a fresh clone of your GitHub repository and copy this updated package into it. Keep the previous local repository as a backup. A new deletion commit on the failed local branch may still retain the missing LFS pointers in its earlier commits. A fresh clone uses only remote history and avoids carrying those failed local commits. Do not copy the old `.git` directory. Preserve any personal COARE settings separately; the package does not contain your configured `coare/coare.env`.

Run `git lfs ls-files` before committing/pushing: this code-only package has no `.pt` assets to upload. Push normally, without a force push or `lfs.allowincompletepush`. Keep your newly trained weights on the assigned COARE storage and back them up separately.
