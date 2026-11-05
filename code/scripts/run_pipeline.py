"""Version-pinned DocScope pipelines. Camera-ready is the default."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

CODE = Path(__file__).resolve().parents[1]
LEGACY_CODE = 'ac0de26b050d4f51de0407c45b5cf3fdf245e1bd'
RELEASES = {
    'legacy': ('4326db57d779829a44e46849fed0ebbe782f2e16', 'ad07c6dfc43e5275ae92b894bf69f151c3dd9467ed79f5b12df904438c5ecb8e'),
    'camera-ready': ('505d47f5d767e613affbf6103a511b29d5d9b553', 'bb22d45884278aef474b0e45e51e21d346ee48dca30018d195fecfc9d6e4793e'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_id')
    p.add_argument('--version',choices=RELEASES,default='camera-ready')
    p.add_argument('--config',type=Path,default=CODE/'models.yaml')
    p.add_argument('--dataset-dir',type=Path,help='Optional local snapshot (benchmark.json and pdfs/); hash must match selected release')
    p.add_argument('--output-root',type=Path,default=CODE/'results')
    p.add_argument('--split',choices=['test','dev'],default='test')
    p.add_argument('--limit',type=int,default=0,help='Select first N questions for a smoke run; 0 means all')
    p.add_argument('--prepare-only',action='store_true',help='Prepare pinned inputs without API calls')
    a=p.parse_args()
    if a.limit<0: p.error('--limit must be nonnegative')
    import yaml
    cfg=yaml.safe_load(a.config.read_text()); defaults=cfg.get('defaults') or {}
    model=next((m for m in cfg['models'] if m['id']==a.model_id),None)
    if model is None: p.error(f'Unknown model: {a.model_id}')
    if Path(a.model_id).name!=a.model_id or a.model_id in ('.','..'): p.error('model_id must be a directory-safe name')
    revision, digest=RELEASES[a.version]
    if a.dataset_dir:
        data=a.dataset_dir.resolve()
    else:
        from huggingface_hub import snapshot_download
        data=Path(snapshot_download('MiliLab/DocScope',repo_type='dataset',revision=revision,
                                  allow_patterns=['benchmark.json','pdfs/*.pdf']))
    raw=(data/'benchmark.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Dataset hash does not match selected version; use its pinned snapshot')
    questions=[q for q in json.loads(raw) if q['split']==a.split]
    if a.limit: questions=questions[:a.limit]
    if not questions: raise ValueError('No selected questions')
    for q in questions:
        pdf=data/'pdfs'/f"{q['pdf']['doc_id_str']}.pdf"
        if not pdf.is_file() or not pdf.open('rb').read(5)==b'%PDF-':
            raise ValueError(f'Missing PDF or unresolved LFS pointer: {pdf}')
    out=a.output_root.resolve()/a.version/a.model_id/(a.split+(f'-limit{a.limit}' if a.limit else ''))
    out.mkdir(parents=True,exist_ok=True)
    judges={k:v for k,v in os.environ.items() if k in ('JUDGE_MODEL','JUDGE_BASE_URL','BBOX_JUDGE_MODEL','BBOX_JUDGE_BASE_URL','ATTRIBUTION_JUDGE_MODEL','ATTRIBUTION_JUDGE_BASE_URL')}
    if a.version=='camera-ready':
        for k,v in [('JUDGE_MODEL','qwen3.6-plus'),('BBOX_JUDGE_MODEL','gpt-5.5'),('ATTRIBUTION_JUDGE_MODEL','qwen3.8-max')]:
            judges.setdefault(k,v);os.environ.setdefault(k,v)
    source_hash=hashlib.sha256()
    for folder in ('infer','eval'):
        for f in sorted((CODE/folder).rglob('*')):
            if f.suffix in ('.py','.txt'):
                source_hash.update(str(f.relative_to(CODE)).encode()+f.read_bytes())
    manifest=dict(version=a.version,dataset_revision=revision,benchmark_sha256=digest,
        code_revision=LEGACY_CODE if a.version=='legacy' else source_hash.hexdigest(),
        model=model,judges=judges,split=a.split,limit=a.limit,question_ids=[q['id'] for q in questions])
    mf=out/'run.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:
        raise ValueError('Run configuration changed; choose a new --output-root')
    mf.write_text(json.dumps(manifest,indent=2)+'\n')
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    if a.version=='legacy':
        archive=subprocess.check_output(['git','-C',str(CODE.parent),'archive',LEGACY_CODE,'code'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # Members come from a pinned commit in this repository.
            for member in tar.getmembers():
                if member.isfile():
                    target=runtime/Path(member.name).relative_to('code')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
    else:
        for folder in ('infer','eval','prompts'):
            shutil.copytree(CODE/folder,runtime/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    link=runtime/'pdfs'
    if link.is_symlink() and link.resolve()!=(data/'pdfs').resolve(): link.unlink()
    if not link.exists(): link.symlink_to(data/'pdfs',target_is_directory=True)
    benchmark=out/'benchmark.json';benchmark.write_text(json.dumps(questions,ensure_ascii=False))
    if a.prepare_only:
        print(f'Prepared {len(questions)} questions: {out}');return
    (out/'summary.json').unlink(missing_ok=True)
    def run(script,*args):
        subprocess.run([sys.executable,str(runtime/script),*map(str,args)],cwd=runtime,check=True)
    infer=out/'infer.jsonl'
    # Existing inference code resumes by ID. Retain only successful rows so failures retry.
    if infer.exists():
        rows=[json.loads(line) for line in infer.read_text().splitlines() if line.strip()]
        infer.write_text(''.join(json.dumps(r)+'\n' for r in rows if r.get('status')=='ok'))
    run(Path('infer/run_infer.py'),'--backend',model['backend'],'--model',model['model'],
        '--api_key_env',model['api_key_env'],'--base_url',model['base_url'],
        '--benchmark',benchmark,'--output',infer,'--concurrency',defaults.get('infer_concurrency',4),
        *model.get('extra_args',[]))
    records=[json.loads(line) for line in infer.read_text().splitlines() if line.strip()]
    if len(records)!=len(questions) or any(r.get('status')!='ok' for r in records):
        raise ValueError('Inference incomplete; rerun to retry failed questions')
    if a.version=='camera-ready':
        run(Path('eval/camera_ready/score.py'),'--benchmark',benchmark,'--infer',infer,
            '--output',out/'judgments','--pdf-dir',data/'pdfs','--concurrency',defaults.get('eval_concurrency',4))
        shutil.copy2(out/'judgments/summary.json',out/'summary.json')
    else:
        for name in ('pages','bbox','facts','answer'):
            output=out/f'{name}.jsonl'
            if output.exists():
                rows=[json.loads(l) for l in output.read_text().splitlines() if l.strip()]
                output.write_text(''.join(json.dumps(r)+'\n' for r in rows if r.get('status')=='ok'))
            run(Path(f'eval/score_{name}.py'),'--benchmark',benchmark,'--infer',infer,'--output',output)
            if output.exists() and any(json.loads(l).get('status')!='ok' for l in output.read_text().splitlines() if l.strip()):
                raise ValueError(f'{name} judging failed; rerun to retry')
        summary=subprocess.check_output([sys.executable,str(runtime/'scripts/summarize_eval.py'),'--eval-dir',str(out)],cwd=runtime)
        (out/'summary.json').write_bytes(summary)
    print(f'Completed: {out / "summary.json"}')


if __name__=='__main__': main()
