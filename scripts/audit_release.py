"""Audit Git-visible project files only; never print suspected secret values."""
from pathlib import Path
import re
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z','--','.'],cwd=root).decode().split('\0')
problems=[]
for name in filter(None,files):
    p=root/name
    if not p.is_file(): continue
    if p.suffix.lower() in {'.pth','.pt','.onnx','.npy','.npz','.zip','.gz','.tar','.ckpt','.safetensors'}:
        problems.append((name,'model/data/archive file'))
    if p.stat().st_size>5*1024*1024: problems.append((name,'file exceeds 5 MiB'))
    if p.suffix.lower() in {'.jpg','.jpeg','.png','.gif','.webp'} and not name.startswith('docs/screenshots/'):
        problems.append((name,'image outside reviewed screenshots'))
    try: text=p.read_text()
    except (UnicodeError,OSError): continue
    patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',r'AKIA[A-Z0-9]{16}',r'sk-[A-Za-z0-9_-]{32,}']
    if any(re.search(pattern,text) for pattern in patterns): problems.append((name,'possible credential'))
    if re.search(('/' + r'Users/[^/\s]+/|' + '/' + r'home/[^/\s]+/'),text): problems.append((name,'machine-specific home path'))
for name,reason in problems: print(f'{name}: {reason}')
print(f'Audited {len([f for f in files if f])} Git-visible paths; {len(problems)} findings. Heuristic scan, not proof of absence.')
sys.exit(bool(problems))
