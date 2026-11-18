"""Audit unchanged episode receipts; reconcile only the explicitly pinned NFS artifact.

The published evaluator inventories every file, including an unreferenced .nfs
artifact that NFS later removed. It consequently assigned two directory hashes
while all persistent params/assets stayed byte-identical. Never rewrite receipts,
ignore arbitrary missing files, or reinterpret a genuine weight difference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import fcntl

TASKS=['bimanual_assembly','bimanual_hanoi','pinch_tongs','fold_glasses','bimanual_microwave_cook',
       'hammer_nail','bimanual_unlock_ipad','pick_bucket','bimanual_photograph','click_mouse','water_plant']
IGNORED = 'params/ocdbt.process_0/d/.nfs0000000018588aeb00000a12'
IGNORED_SHA = 'c72bd5770dc7b4aa0bfc0743b9e18546abe1da91b4522605b90b165c603017c5'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def inventory_sha(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def write(p,obj):
    temp=p.with_suffix('.pending');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(p)


def canonical_inventory(files, expected):
    result=dict(files)
    if IGNORED in result:
        if result.pop(IGNORED)!=IGNORED_SHA:raise ValueError('NFS artifact differs from the pinned original')
    if result!=expected:raise ValueError('Persistent checkpoint params/assets differ')
    return inventory_sha(result)


def verify_model(model):
    manifest=json.loads((model/'MIGRATION_VERIFIED.json').read_text())
    if not manifest['verified'] or manifest['optimizer_updates']!=60000:raise ValueError('Wrong final model')
    if '59999/'+IGNORED in manifest['files']:
        raise ValueError('Artifact unexpectedly belongs to the approved source model')
    for name,entry in manifest['files'].items():
        if (model/name).stat().st_size!=entry['bytes'] or sha(model/name)!=entry['sha256']:
            raise ValueError('Published source model bytes differ: '+name)
    expected={name.removeprefix('59999/'):entry['sha256'] for name,entry in manifest['files'].items()
              if name.startswith('59999/') and name!='59999/'+IGNORED}
    return manifest,expected


def verify_task(root,task,manifest,expected,full):
    shard=root/'shards'/task
    records=[]
    for ep in range(50):
        folder=shard/task/f'episode_{ep:02d}'
        if not (folder/'verified.json').exists():break
        verification=json.loads((folder/'verified.json').read_text())
        if not verification['verified'] or sha(folder/'receipt.json')!=verification['receipt_sha256']:
            raise ValueError('Episode receipt checksum differs')
        row=json.loads((folder/'receipt.json').read_text())
        if (row['task']!=task or row['episode']!=ep or row['seed']!=0 or row['smoke']
            or row['optimizer_updates']!=60000 or row['evaluation_seed']!=0 or row['training_seed']!=42
            or row['denoising_steps']!=10 or row['replan_ratio']!=.8 or row['level']!='rand_obj'
            or row['episodes_per_task']!=50 or row['mode']!='formal' or row['run']!='late_touch_real'
            or row['official_commit']!='8d23b0fab23b17a58c4b55f3942e17013aaf8267'):
            raise ValueError('Formal evaluation protocol differs')
        if row['checkpoint_sha256']!=inventory_sha(row['checkpoint_files']):raise ValueError('Raw model inventory hash differs')
        canonical_inventory(row['checkpoint_files'],expected)
        if row['E_sha256']!=manifest['files']['encoder-60000.pt']['sha256']:
            raise ValueError('Paired encoder identity differs')
        if sha(folder/'progress.pkl')!=row['progress_sha256']:raise ValueError('Episode progress checksum differs')
        if len(row['videos'])!=(3 if task.startswith('bimanual_') else 2):raise ValueError('Missing camera video')
        if full:
            for video in row['videos']:
                if sha(folder/video['file'])!=video['sha256'] or video['frames']<=1:raise ValueError('Video checksum/frame evidence differs')
            inferences=json.loads((folder/'inferences.json').read_text())
            if not inferences or not all(x['finite'] and not x['warmup'] for x in inferences):
                raise ValueError('Missing/invalid actual inference trace')
        records.append(row)
    if full:
        completion=json.loads((shard/'complete.json').read_text())
        if not completion['complete'] or completion['episodes']!=50 or len(records)!=50:
            raise ValueError('Incomplete formal task shard')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');a=p.parse_args()
    base=Path(os.environ['AFCE_EVAL_ROOT']);root=base/'results/real/seed0'
    lock=(root/'final_identity_audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=base/'models/single_stage_late_touch_touch_seed42'
    manifest,expected=verify_model(model)
    while True:
        all_rows=[row for task in TASKS for row in verify_task(root,task,manifest,expected,False)]
        portables={json.dumps(x['portable_eval_code'],sort_keys=True) for x in all_rows}
        if len(portables)>1:raise ValueError('Mixed portable inference/evaluation protocol code')
        aliases={x['checkpoint_sha256']:canonical_inventory(x['checkpoint_files'],expected) for x in all_rows}
        proof={'verified':True,'canonical_checkpoint_sha256':inventory_sha(expected),
            'raw_inventory_aliases':aliases,'ignored_only':{IGNORED:IGNORED_SHA},
            'reason':'Receipt-pinned destination NFS artifact is not in the approved source manifest; every approved params/asset file remains byte-identical',
            'persistent_model_files_verified':len(manifest['files']),
            'migration_receipt_sha256':sha(model/'MIGRATION_VERIFIED.json'),
            'episode_receipts_unchanged':True,'verified_episode_count':len(all_rows),'unix':time.time()}
        write(root/'MODEL_IDENTITY_EQUIVALENCE_VERIFIED.json',proof)
        if all((root/'shards'/task/'complete.json').exists() for task in TASKS):
            final=[row for task in TASKS for row in verify_task(root,task,manifest,expected,True)]
            if len(final)!=550 or len({(x['task'],x['episode']) for x in final})!=550:raise ValueError('Need 550 unique episodes')
            if len({json.dumps(x['portable_eval_code'],sort_keys=True) for x in final})!=1:raise ValueError('Mixed protocol')
            tasks=[{'task':task,'episodes':50,'successes':sum(x['success'] for x in final if x['task']==task)} for task in TASKS]
            result={'verified':True,'complete':True,'episodes':550,'successes':sum(x['success'] for x in final),
                'condition':'real','evaluation_seed':0,'optimizer_updates':60000,'tasks':tasks,
                'checkpoint_sha256':inventory_sha(expected),'identity_equivalence_receipt_sha256':sha(root/'MODEL_IDENTITY_EQUIVALENCE_VERIFIED.json'),
                'receipt_sha256':{f'{x["task"]}/{x["episode"]}':sha(root/'shards'/x['task']/x['task']/f'episode_{x["episode"]:02d}/receipt.json') for x in final},
                'completed_unix':time.time()}
            (root/'REPORT.md').write_text('# Real tactile final evaluation\n\n'+
                f'Verified: 550 episodes; successes: {result["successes"]}/550.\n\n'+
                '\n'.join(f'- {x["task"]}: {x["successes"]}/50' for x in tasks)+
                '\n\nRaw directory-inventory hashes differ only by one pinned NFS artifact. Persistent model bytes, protocol, and unmodified episode receipts verified; see MODEL_IDENTITY_EQUIVALENCE_VERIFIED.json.\n')
            write(root/'complete.json',result);print('ALL_550_EPISODES_VERIFIED',flush=True);return 0
        if a.once:print(json.dumps(proof));return 0
        time.sleep(60)


if __name__=='__main__':raise SystemExit(main())
