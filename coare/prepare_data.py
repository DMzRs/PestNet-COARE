"""Retarget an existing YOLO dataset YAML without relabeling or splitting images."""
import argparse
import re
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Existing dataset data.yaml with real class names')
parser.add_argument('--root', required=True, help='Actual absolute dataset root on COARE')
parser.add_argument('--output', required=True)
args = parser.parse_args()
data = yaml.safe_load(Path(args.input).read_text())
if not isinstance(data, dict) or not all(k in data for k in ('train', 'val', 'names')):
    raise SystemExit('Input YAML must contain train, val, and the original names mapping/list.')
if len(data['names']) != 102 or data.get('nc', 102) != 102:
    raise SystemExit('This package targets the 102-class IP102 baseline; provide its actual 102-class YAML.')
root = Path(args.root)
if not root.is_absolute() or '\\' in args.root or re.match(r'^[A-Za-z]:', args.root):
    raise SystemExit('--root must be an absolute Linux path on COARE.')
for key in ('train', 'val', 'test'):
    values = data.get(key)
    if values is None:
        continue
    for value in values if isinstance(values, list) else [values]:
        if not isinstance(value, str) or '\\' in value or re.match(r'^[A-Za-z]:', value):
            raise SystemExit(f'{key} contains a Windows/non-path entry; update it to the actual Linux image path first.')
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not path.exists():
            raise SystemExit(f'{key} path does not exist: {path}')
data['path'] = str(root.resolve())
data.pop('download', None)  # User supplies dataset; the adapter never downloads/re-splits it.
output = Path(args.output).resolve()
if output == Path(args.input).resolve():
    raise SystemExit('Use a separate output YAML to preserve the original.')
if output.exists():
    raise SystemExit('Output already exists; choose a new filename.')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(f'Created {output}; original split entries and class mapping retained.')
