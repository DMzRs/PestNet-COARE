"""COARE launcher. Uses unchanged repository model/parser/trainer source."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from build_runtime import ROOT, RUNTIME, verify_source

sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/ultralytics_settings'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'false')


def bootstrap():
    verify_source(runtime=True)
    sys.path.insert(0, str(RUNTIME))
    import ultralytics
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.ext.Blocks as blocks
    for module in (ultralytics, tasks, blocks):
        if not Path(module.__file__).resolve().is_relative_to(RUNTIME):
            raise RuntimeError(f'Wrong module loaded: {module.__file__}')
    if ultralytics.__version__ != '8.3.18':
        raise RuntimeError('Wrong Ultralytics version.')
    print('Verified archived source: model YAML, parser, custom Blocks, and trainer unchanged.', flush=True)
    print(f'Import path: {ultralytics.__file__}', flush=True)
    return ultralytics.YOLO


def smoke_data(data, folder):
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import yaml_save
    # Native enumeration preserves repository interpretation of directory/list splits.
    def enumerate_images(split):
        dummy = object.__new__(YOLODataset)
        dummy.prefix = ''
        dummy.fraction = 1.0
        return dummy.get_img_files(data[split])
    smoke = {'path': str(folder), 'names': data['names'], 'nc': data['nc']}
    for split in ('train', 'val'):
        paths = enumerate_images(split)[:32 if split == 'train' else 8]
        if not paths:
            raise RuntimeError(f'No images for smoke {split}')
        target = folder / f'{split}.txt'
        target.write_text('\n'.join(str(Path(p).resolve()) for p in paths)+'\n')
        smoke[split] = str(target)
    yaml_save(folder / 'data.yaml', smoke)
    return folder / 'data.yaml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'smoke', 'train', 'resume'])
    parser.add_argument('--data')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', choices=['0', 'cpu'], default='0')
    parser.add_argument('--save-period', type=int, default=None,
                        help='Optional native save_period override; does not patch its indexing.')
    args = parser.parse_args()
    YOLO = bootstrap()
    import torch
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG_DICT, yaml_load
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Run in a GPU allocation and check the installed torch build.')
    stamp = os.environ.get('SLURM_JOB_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record_dir = ROOT / 'outputs' / f'{args.action}_{stamp}'
    record_dir.mkdir(parents=True, exist_ok=False)
    model_yaml = RUNTIME / 'cfg/hsdpa.yaml'
    base = yaml_load(ROOT / 'configs/baseline_train.yaml')
    if args.action == 'resume':
        if not args.checkpoint or not Path(args.checkpoint).is_file():
            raise RuntimeError('Provide an existing trusted repository checkpoint with --checkpoint.')
        checkpoint = Path(args.checkpoint).resolve()
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ckpt.get('optimizer') is None or ckpt.get('epoch', -1) < 0:
            raise RuntimeError('Checkpoint has no resumable optimizer/epoch state (possibly finalized/stripped).')
        recorded_yaml = (ckpt.get('ema') or ckpt.get('model')).yaml
        for key in ('backbone', 'head', 'scales', 'nc'):
            if recorded_yaml.get(key) != yaml_load(model_yaml).get(key):
                raise RuntimeError(f'Checkpoint model differs from archived baseline: {key}')
        model = YOLO(str(checkpoint), task='detect')
    else:
        model = YOLO(str(model_yaml), task='detect')
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f'Baseline training parameters (unfused): {parameters:,}', flush=True)
    settings = {**DEFAULT_CFG_DICT, **base, 'model': str(model_yaml), 'device': args.device}
    record = {'action': args.action, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'parameters': parameters, 'source_manifest': json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()),
              'launch_settings': settings, 'checkpoint': args.checkpoint}
    if args.action == 'check':
        model.model.to('cuda:0' if args.device == '0' else 'cpu').eval()
        tensor = torch.zeros(1, 3, 640, 640, device=next(model.model.parameters()).device)
        with torch.no_grad():
            result = model.model(tensor)
        print('PASS: baseline model construction and 640x640 forward pass.', flush=True)
        record['forward_output_shape'] = list(result[0].shape)
        # Diagnose inference count on a separate copy; never fuse the training model.
        fused = deepcopy(model.model).fuse(verbose=False).eval()
        fused_parameters = sum(p.numel() for p in fused.parameters())
        with torch.no_grad():
            fused_result = fused(tensor)
        torch.testing.assert_close(fused_result[0], result[0], rtol=1e-4, atol=1e-4)
        record['parameters_fused'] = fused_parameters
        print(f'Baseline inference parameters (fused): {fused_parameters:,} '
              f'({fused_parameters / 1e6:.6f} M)', flush=True)
        (record_dir/'launch_record.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
        return
    if not args.data or not Path(args.data).is_file():
        raise RuntimeError('Supply your existing 102-class YOLO dataset YAML with --data.')
    original = yaml_load(Path(args.data).resolve())
    if 'path' in original and not Path(original['path']).is_absolute():
        raise RuntimeError('Use prepare_data.py to set an absolute COARE dataset root first.')
    data = check_det_dataset(str(Path(args.data).resolve()), autodownload=False)
    if data['nc'] != 102:
        raise RuntimeError('The archived baseline targets 102 classes. No automatic class-count modification allowed.')
    data_yaml = Path(args.data).resolve()
    kwargs = {**base, 'data': str(data_yaml), 'device': args.device,
              'project': str(ROOT/'outputs/runs'), 'name': f'baseline_{stamp}'}
    if args.save_period is not None:
        kwargs['save_period'] = args.save_period
    if args.action == 'smoke':
        data_yaml = smoke_data(data, record_dir)
        kwargs.update(data=str(data_yaml), epochs=1, workers=0, name=f'smoke_{stamp}')
    if args.action == 'resume':
        # Native trainer restores checkpoint training arguments; no restart/fine-tune fallback.
        kwargs = {'resume': True, 'data': str(data_yaml), 'device': args.device,
                  'batch': base['batch']}
    record['launch_kwargs'] = kwargs
    record['resume_restores_checkpoint_arguments'] = args.action == 'resume'
    (record_dir/'launch_record.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    model.train(**kwargs)
    print(f'Completed {args.action}; results: {model.trainer.save_dir}', flush=True)

if __name__ == '__main__':
    main()
