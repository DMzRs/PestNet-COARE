"""Supply omitted framework files without changing any archived source file."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/HSDPA-main'
RUNTIME = ROOT / 'runtime'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_source(runtime=False):
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    expected = manifest['files']
    actual = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise RuntimeError('Archived source file list changed.')
    for name, sha in expected.items():
        if digest(SOURCE / name) != sha:
            raise RuntimeError(f'Archived source changed: {name}')
        if runtime and '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):
            if not (RUNTIME / name).is_file() or digest(RUNTIME / name) != sha:
                raise RuntimeError(f'Runtime source differs from archive: {name}')
    if runtime:
        provenance = json.loads((RUNTIME / 'SUPPORT_FILES.json').read_text())
        for name, sha in provenance['added_files'].items():
            if digest(RUNTIME / name) != sha:
                raise RuntimeError(f'Added support file changed: {name}')
        wanted = {n for n in expected if '__pycache__' not in Path(n).parts and not n.endswith('.pyc')}
        wanted |= set(provenance['added_files']) | {'SUPPORT_FILES.json'}
        present = {p.relative_to(RUNTIME).as_posix() for p in RUNTIME.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if present != wanted:
            raise RuntimeError('Unexpected/missing runtime files. Rebuild the runtime.')
    return manifest

def build():
    manifest = verify_source()
    wheel = ROOT / 'vendor/ultralytics-8.3.18-py3-none-any.whl'
    if digest(wheel) != manifest['support_wheel_sha256']:
        raise RuntimeError('Support wheel checksum differs.')
    if RUNTIME.exists():
        raise RuntimeError('runtime already exists. Use --verify, or remove only runtime/ before rebuilding.')
    shutil.copytree(SOURCE, RUNTIME, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    added = {}
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if not name.startswith('ultralytics/') or name.endswith('/'):
                continue
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe support-wheel entry.')
            target = RUNTIME / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
            added[name] = digest(target)
    (RUNTIME / 'SUPPORT_FILES.json').write_text(json.dumps({
        'source': 'Official ultralytics 8.3.18 wheel; only absent files supplied',
        'wheel_sha256': manifest['support_wheel_sha256'], 'added_files': added}, indent=2)+'\n')
    verify_source(runtime=True)
    print(f'Preserved {len(manifest["files"])} archived files; supplied {len(added)} missing framework files.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify_source(runtime=True)
        print('PASS: original and runtime source checksums match the archive.')
    else:
        build()
