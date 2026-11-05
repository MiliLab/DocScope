"""Run camera-ready judges and aggregate nine metrics for a new inference run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval_common import make_judge_client, make_bbox_judge_client, stream_chat, parse_judge_json, png_to_data_uri
import render_eval_page
from metrics import validbox, count_question, aggregate


def fill(template, **values):
    return re.sub(r'\{(' + '|'.join(values) + r')\}', lambda m: str(values[m[1]]), template)


def judge(client, model, prompt, image=None):
    content = prompt if image is None else [
        {'type': 'image_url', 'image_url': {'url': png_to_data_uri(image)}},
        {'type': 'text', 'text': prompt}]
    raw, usage = stream_chat(client, model, [{'role': 'user', 'content': content}],
                             max_tokens=16384, temperature=None)
    data = parse_judge_json(raw)
    if not isinstance(data, dict):
        raise ValueError('Judge did not return a JSON object')
    return data, {'model': model, 'raw': raw, 'usage': usage}


def score_question(q, inf, output, clients, models):
    records = []
    def call(kind, prompt, image=None):
        data, trace = judge(clients[kind], models[kind], prompt, image)
        records.append({'kind': kind, **trace})
        return data
    templates = {name: (HERE/'prompts'/f'{name}.txt').read_text() for name in
                 ('region', 'region_extension', 'facts', 'attribution', 'answer')}
    parsed = inf['parsed']
    answer = parsed.get('answer') or inf.get('model_raw', '')
    a = call('text', templates['answer'].format(question=q['question'],
        gold_answer=q['answer']['answer_text'] if q['answer']['is_answerable'] else 'Unanswerable', model_answer=answer))
    if type(a.get('consistent')) is not bool:
        raise ValueError('Invalid answer judgment')
    result = dict(answer_correct=a['consistent'], regions=[], facts=[])
    if q['answer']['is_answerable']:
        cs = parsed['citations']
        es = {e['local_id']: e for e in q['evidences']}
        hit = set(parsed['predicted_pages']) & {e['page'] for e in es.values()}
        for pg in sorted(hit):
            gold = [e for e in es.values() if e['page'] == pg]
            valid = [(i,c) for i,c in enumerate(cs) if c['page'] == pg and validbox(c)]
            if valid:
                image = output/'imgs'/f"{q['id'].replace('::','__')}_{pg}.png"
                if not render_eval_page.render_pair(q['pdf']['doc_id_str'], pg,
                        [e['bbox'] for e in gold], [c['bbox'] for _,c in valid], image):
                    raise ValueError(f'Cannot render {q["id"]} page {pg}')
                block = '\n'.join(f"id=g{i} :: index_on_page={i}, element_type={e.get('element_type','unknown')}, gold_bbox_px={e['bbox']}" for i,e in enumerate(gold,1))
                prompt = templates['region'].format(question=q['question'],gold_answer=q['answer']['answer_text'],
                    gt_page=pg,gold_total_on_page=len(gold),gold_facts_block=block,n_pred=len(valid),
                    pred_bboxes_norm=[c['bbox'] for _,c in valid])+templates['region_extension']
                r = call('region', prompt, image)
                preds = r.get('predictions', [])
                expected = {f'p{i}' for i in range(1,len(valid)+1)}
                if len(preds)!=len(expected) or {p['id'] for p in preds}!=expected:
                    raise ValueError('Missing/duplicate individual predicted-box judgments')
                for pred in preds:
                    ids = []
                    seen = set()
                    for match in pred['matches']:
                        gid = match['gold_id']
                        if gid not in {f'g{i}' for i in range(1,len(gold)+1)} or gid in seen:
                            raise ValueError('Invalid region match ID')
                        seen.add(gid)
                        if match['label'] not in ('covered','imprecise'):
                            raise ValueError('Invalid region match label')
                        if match['label']=='covered': ids.append(gold[int(gid[1:])-1]['local_id'])
                    result['regions'].append(dict(citation_index=valid[int(pred['id'][1:])-1][0],covered_evidence_ids=ids))
            facts = [f for f in q['facts'] if any(es[e]['page']==pg for e in f.get('evidence_local_ids') or [f['evidence_local_id']])]
            if not facts: continue
            block = '\n'.join(f"id={f['local_id']} :: {f['text_description']}" for f in facts)
            siblings = '; '.join(f['text_description'] for f in q['facts'] if f not in facts) or '(none)'
            fj = call('text', templates['facts'].format(question=q['question'],gt_page=pg,
                 gold_facts_block=block,sibling_facts=siblings,model_raw=inf['model_raw']))
            items = fj.get('items',[])
            labels = {x['id']:x['label'] for x in items}
            if len(items)!=len(facts) or set(labels)!={f['local_id'] for f in facts}:
                raise ValueError('Incomplete fact judgments')
            catalog = [dict(citation_id=f'c{i+1}',source_citation_index=i,
                       **{k:v for k,v in c.items() if k not in ('sentence','span_start','span_end')}) for i,c in enumerate(cs)]
            for f in facts:
                ar = call('attribution', fill(templates['attribution'], question=q['question'],
                    fact_id=f['local_id'],fact_text=f['text_description'],page=pg,
                    model_raw=inf['model_raw'],catalog=json.dumps(catalog,ensure_ascii=False)))
                candidates = [f'c{i+1}' for i,c in enumerate(cs) if c['page']==pg]
                checks = ar.get('candidate_checks',[])
                assocs = ar.get('associations',[])
                ids = [x['citation_id'] for x in assocs]
                if ([x['citation_id'] for x in checks]!=candidates or
                    any(type(x.get('associated')) is not bool for x in checks) or
                    ids!=[x['citation_id'] for x in checks if x['associated']]):
                    raise ValueError('Invalid attribution candidate ledger')
                for assoc in assocs:
                    idx = int(assoc['citation_id'][1:])-1
                    prefix = inf['model_raw'][:cs[idx]['span_start']]
                    if not assoc.get('evidence_quote') or assoc['evidence_quote'] not in prefix:
                        raise ValueError('Attribution quote is absent before its citation')
                    if assoc.get('context_quote') and assoc['context_quote'] not in prefix:
                        raise ValueError('Attribution context is absent before its citation')
                result['facts'].append(dict(fact_id=f['local_id'],page=pg,label=labels[f['local_id']],
                                            citation_indices=[int(cid[1:])-1 for cid in ids]))
    counts = count_question(q, inf, result)
    return dict(question_id=q['id'],judgments=result,counts=counts,judge_traces=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--infer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdf-dir',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    # A previous summary must never survive a failed rerun.
    (a.output/'summary.json').unlink(missing_ok=True)
    bench=json.loads(a.benchmark.read_text()); inf={}
    for line in a.infer.read_text().splitlines():
        r=json.loads(line); inf[r['question_id']]=r
    if set(inf)!={q['id'] for q in bench} or any(r.get('status')!='ok' for r in inf.values()):
        raise ValueError('Inference must contain a successful record for every selected question')
    render_eval_page.PDF_DIR=a.pdf_dir.resolve()
    models=dict(text=os.environ.get('JUDGE_MODEL','qwen3.6-plus'),
                region=os.environ.get('BBOX_JUDGE_MODEL','gpt-5.5'),
                attribution=os.environ.get('ATTRIBUTION_JUDGE_MODEL','qwen3.8-max'))
    from openai import OpenAI
    clients=dict(text=make_judge_client(),region=make_bbox_judge_client(),attribution=OpenAI(
        api_key=os.environ.get('ATTRIBUTION_JUDGE_API_KEY') or os.environ['JUDGE_API_KEY'],
        base_url=os.environ.get('ATTRIBUTION_JUDGE_BASE_URL') or os.environ['JUDGE_BASE_URL']))
    def work(q):
        fingerprint=hashlib.sha256(json.dumps([q,inf[q['id']],models,
            {k:os.environ.get(k) for k in ('JUDGE_BASE_URL','BBOX_JUDGE_BASE_URL','ATTRIBUTION_JUDGE_BASE_URL')},
            {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [*HERE.glob('*.py'),*(HERE/'prompts').glob('*.txt')]}],sort_keys=True).encode()).hexdigest()
        dest=a.output/f'{fingerprint}.json'
        if dest.exists(): return json.loads(dest.read_text())
        r=score_question(q,inf[q['id']],a.output,clients,models)
        r['fingerprint']=fingerprint
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));tmp.replace(dest)
        print(f"Scored {q['id']}",flush=True)
        return r
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(work,bench))
    summary=aggregate(r['counts'] for r in results)
    summary.update(models=models,benchmark_sha256=hashlib.sha256(a.benchmark.read_bytes()).hexdigest(),units='percent')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
