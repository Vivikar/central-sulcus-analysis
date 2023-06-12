from pathlib import Path
import subprocess
from tqdm import tqdm

segm_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/correctes_res/predicted')
imgs = [x for x in segm_path.glob('**/*sulc*.nii.gz')]
print(len(imgs))
for i in tqdm(imgs):
    o = str(i).replace('.nii.gz', '')
    o = o.replace('segmentations', 'meshes')
    AimsMeshBrainL = ['/home/vladyslavz/portable_apps/brainvisa4_5/bin/AimsMeshBrain', '-i', str(i), '-o', f'{o}.ply']
    subprocess.run(AimsMeshBrainL)
