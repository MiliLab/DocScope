"""Reproduce the agreed micro metrics; do not fabricate per-citation correctness.
No API calls. All input files remain unchanged. Run from any directory.
"""
from pathlib import Path
import json, hashlib, collections, math
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True, help='Directory containing the benchmark and frozen judge artifacts; see README.md')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT=args.root.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)

def artifact_path(value):
 # Frozen indices may contain paths from the original author workstation.
 p = Path(value)
 if p.exists():
  return p
 parts = p.parts
 for marker in ('output', 'Inference_benchmark_noimg_20260915'):
  if marker in parts:
   return ROOT.joinpath(*parts[parts.index(marker):])
 raise FileNotFoundError(value)

BENCHMARK_SHA256=hashlib.sha256((ROOT/'HF-DocScope-v2-20260916/benchmark.json').read_bytes()).hexdigest()
SRC=ROOT/'Inference_benchmark_noimg_20260915/results/api_eval'
REG=ROOT/'output/region_judge_full_32104_20260916'
CIT=ROOT/'output/citation_attribution_full_qwen38_r10_20260917'
ANSWER_REJUDGE=ROOT/'output/answer_rejudge_qwen36_20260918/results'
FACT_REJUDGE=ROOT/'output/fact_judge_alternative_qwen36_20260918/results'
RESOLUTION=ROOT/'output/complete_resolution_20260918'
RUN=REG/'_runs/codex_gpt-5.5_none_retry_remaining_20260917_085902_attempt3'
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with open(p) as f:
  for l in f:
   if l.strip():yield json.loads(l)
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def rate(a,b):return 100*a/b if b else None
def f1(p,r):return 2*p*r/(p+r) if p is not None and r is not None and p+r else (0 if p==0 and r==0 else None)
def boxkey(c):return (c['page'],tuple(c.get('bbox') or []))
def validbox(c):
 b=c.get('bbox') or []
 return bool(c.get('is_normalized')) and len(b)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1
B={q['id']:q for q in read(ROOT/'HF-DocScope-v2-20260916/benchmark.json') if q['split']=='test'}
A={qid:q for qid,q in B.items() if q['answer']['is_answerable']}
assert len(B)==729 and len(A)==678
nf=sum(len(q['facts']) for q in A.values());assert nf==3847
frozen={q['id']:q for q in read(REG/'source/benchmark.json')}
for qid,q in A.items():
 assert q['evidences']==frozen[qid]['evidences'],qid+' changed evidence since region judging'
 assert q['facts']==frozen[qid]['facts'],qid+' changed facts since region judging'
S={}
for qid,q in A.items():
 es={e['local_id']:e for e in q['evidences']}
 fs={f['local_id']:f for f in q['facts']}
 fe={fid:set(f.get('evidence_local_ids') or [f['evidence_local_id']]) for fid,f in fs.items()}
 assert all(e in es for ee in fe.values() for e in ee)
 fp={fid:{es[e]['page'] for e in ee} for fid,ee in fe.items()}
 S[qid]=(es,fs,fe,fp,{e['page'] for e in es.values()})
manifest=collections.defaultdict(list)
for t in rows(REG/'manifest.jsonl'):
 if t['question_id'] in A:manifest[t['model']].append(t)
unlinked=[{'question_id':qid,'evidence_id':eid,'page':e['page']} for qid,(es,fs,fe,fp,pg) in S.items() for eid,e in es.items() if not any(eid in ids for ids in fe.values())]
dump('reference_regions_without_fact_links.json',unlinked)
result_paths={r['task_id']:artifact_path(r['result_path']) for r in read(RUN/'reused_results.json')}
if (RESOLUTION/'region_result_index.json').exists():
 for rr in read(RESOLUTION/'region_result_index.json'):
  assert rr['task_id'] not in result_paths
  result_paths[rr['task_id']]=artifact_path(rr['result_path'])
hashes=read(REG/'source/source_hashes.json')
models=['gemini-3.1-pro-preview','gemini-3.1-flash-lite-preview','claude-opus-4-7','claude-sonnet-4-6-20260217','gpt-5.4-2026-03-05','qwen3.6-plus','intern-s1-pro','qwen3.5-397b-a17b','qwen3.5-122b-a10b','qwen3.5-27b','qwen3-vl-235b-a22b-instruct','qwen3-vl-32b-instruct','qwen3-vl-30b-a3b-instruct','qwen3-vl-8b-instruct','gemma-4-31b-it','gemma-4-26b-a4b-it','ministral-14b-2512','ministral-8b-2512']
short=['Gemini 3.1 Pro','Gemini 3.1 Flash Lite','Claude Opus 4.7','Claude Sonnet 4.6','GPT-5.4','Qwen3.6 Plus','Intern-S1-Pro','Qwen3.5-397B-A17B','Qwen3.5-122B-A10B','Qwen3.5-27B','Qwen3-VL-235B-A22B','Qwen3-VL-32B','Qwen3-VL-30B-A3B','Qwen3-VL-8B','Gemma-4-31B','Gemma-4-26B-A4B','Ministral-14B','Ministral-8B']
allstats=[];allq=[];gaps=[];changed=[];alignment_errors=[];missingregions=[];changedanswers=[]
for model,name in zip(models,short):
 print('Calculating',model,flush=True)
 assert hashlib.sha256((SRC/model/'infer.jsonl').read_bytes()).hexdigest()==hashes[model],model+' source changed'
 inf={r['question_id']:r for r in rows(SRC/model/'infer.jsonl') if r['question_id'] in B}
 ans={r['question_id']:r for r in rows(SRC/model/'answer.jsonl') if r['question_id'] in B}
 overlay_path=ANSWER_REJUDGE/f'{model}.jsonl'
 if overlay_path.exists():
  for updated in rows(overlay_path):
   if updated.get('status')=='ok':
    qid=updated['question_id'];assert qid in B
    assert updated['gold_answer']==B[qid]['answer']['answer_text']
    ans[qid]=updated
 assert inf.keys()==B.keys() and ans.keys()==B.keys()
 assert all(r.get('status')=='ok' for r in inf.values()),model+' inference failure'
 # Failed answer judgments remain unknown and retain their denominator.
 stale_answers={qid for qid,r in ans.items() if r.get('status')!='ok'}
 for qid in stale_answers:changedanswers.append({'model':model,'question_id':qid,'reason':'answer_judge_failure','status':ans[qid].get('status'),'model_answer':ans[qid]['model_answer']})
 for qid,q in B.items():
  assert ans[qid]['model_answer']==(inf[qid]['parsed']['answer'] or inf[qid].get('model_raw') or ''),(model,qid,'answer alignment')
  if q['answer']['is_answerable']:
   if ans[qid]['gold_answer']!=q['answer']['answer_text']:
    stale_answers.add(qid)
    changedanswers.append({'model':model,'question_id':qid,'question':q['question'],'old_gold_answer':ans[qid]['gold_answer'],'current_gold_answer':q['answer']['answer_text'],'model_answer':ans[qid]['model_answer'],'old_correct':ans[qid]['consistent']})
  else:assert str(ans[qid]['gold_answer']).strip().lower() in ('','unanswerable'),(model,qid,'unanswerable gold')
 citation={}
 for r in rows(CIT/'results'/f'{model}.jsonl'):
  if r['question_id'] not in A:continue
  key=(r['question_id'],r['page'],r['fact_id']);assert key not in citation
  citation[key]=r
  for a in r['associations']:
   c=a['citation'];i=c['source_citation_index'];src=inf[r['question_id']]['parsed']['citations'][i]
   assert all(c.get(k)==src.get(k) for k in ['page','bbox','span_start','span_end','sentence']),(model,key,'citation alignment')
 factcoverage=dict(citation)
 if FACT_REJUDGE.exists():
  assert read(FACT_REJUDGE.parent/'summary.json')['successful_target_fact_records']==2247
  for r in rows(FACT_REJUDGE/f'{model}.jsonl'):
   assert r['status']=='ok'
   assert r['benchmark_sha256']==BENCHMARK_SHA256
   key=(r['question_id'],r['page'],r['fact_id']);assert key not in factcoverage,'Supplement must not replace old judgments'
   factcoverage[key]={**r,'original_coverage_label':r['label'],'associations':[], 'coverage_source':'alternative_page_rejudge'}
 if RESOLUTION.exists():
  assert read(RESOLUTION/'summary.json')['successful_tasks']==75
  assert read(RESOLUTION/'configuration.json')['benchmark_sha256']==BENCHMARK_SHA256
  for rr in (rows(RESOLUTION/'fact_results'/f'{model}.jsonl') if (RESOLUTION/'fact_results'/f'{model}.jsonl').exists() else []):
   key=(rr['question_id'],rr['page'],rr['fact_id']);assert rr['status']=='ok'
   factcoverage[key]={**rr,'original_coverage_label':rr['label'],'associations':[],'coverage_source':'remaining_fact_rejudge'}
  for rr in (rows(RESOLUTION/'attribution_results'/f'{model}.jsonl') if (RESOLUTION/'attribution_results'/f'{model}.jsonl').exists() else []):
   key=(rr['question_id'],rr['page'],rr['fact_id']);assert rr['status']=='ok'
   citation[key]=rr
 # qid: prediction box -> reference evidence ids. Duplicated boxes use OR of individual matches.
 matched=collections.defaultdict(lambda:collections.defaultdict(set));unknown=collections.defaultdict(dict);seenpred=collections.defaultdict(set)
 for t in manifest[model]:
  qid=t['question_id'];meta=read(REG/t['folder']/'metadata.json');assert meta['source_infer_sha256']==hashes[model]
  ps={p['prediction_id']:p for p in meta['predicted_regions']};gs={g['gold_id']:g['local_id'] for g in meta['gold_regions']}
  for p in ps.values():
   src=inf[qid]['parsed']['citations'][p['source_citation_index']]
   assert boxkey(p)==boxkey(src)
   seenpred[qid].add(boxkey(p))
  path=result_paths.get(t['task_id'])
  if path is None:
   missingregions.append({**t,'reason':'No validated result in final reused_results index'})
   for p in ps.values():unknown[qid][boxkey(p)]=set(gs.values())
   continue
  result=read(path);pr={p['id']:p for p in result['predictions']};assert pr.keys()==ps.keys()
  for pid,p in pr.items():
   for match in p['matches']:
    assert match['gold_id'] in gs
    if match['label']=='covered':matched[qid][boxkey(ps[pid])].add(gs[match['gold_id']])
 totals=collections.Counter();modelgaps=[]
 for qid,q in A.items():
  es,fs,fe,fp,acceptable=S[qid];parsed=inf[qid]['parsed'];pages=set(parsed['predicted_pages']);hit=pages & acceptable
  cs=parsed['citations'];pred={boxkey(c):c for c in cs if c.get('page') in hit}
  expected={k for k,c in pred.items() if validbox(c)}
  assert expected==seenpred[qid],(model,qid,'region task coverage mismatch',expected-seenpred[qid],seenpred[qid]-expected)
  eligible={fid for fid,pg in fp.items() if pg & hit}
  covered_e=set().union(*matched[qid].values()) if matched[qid] else set()
  upper_e=covered_e | (set().union(*unknown[qid].values()) if unknown[qid] else set())
  regionfacts={fid for fid in eligible if fe[fid]&covered_e};regionupper={fid for fid in eligible if fe[fid]&upper_e}
  goodboxes={k for k,e in matched[qid].items() if e};upperboxes=goodboxes|set(unknown[qid])
  assert goodboxes<=pred.keys() and upperboxes<=pred.keys()
  # Coverage reuses canonical judgments and ORs in new page judgments; attribution remains separate.
  oldpositive=set();noattrpositive=set();proxychain=set();unknownfacts=set();newpairs=0;changedpairs=0
  for fid in eligible:
   f=fs[fid]
   for pg in sorted(fp[fid]&hit):
    r=factcoverage.get((qid,pg,fid))
    if r is None:
     newpairs+=1;unknownfacts.add(fid)
     modelgaps.append({'model':model,'question_id':qid,'fact_id':fid,'page':pg,'reason':'missing_page_fact_coverage','fact_text':f['text_description']})
     continue
    if r['fact_text']!=f['text_description']:
     changedpairs+=1;unknownfacts.add(fid);changed.append({'model':model,'question_id':qid,'fact_id':fid,'page':pg,'old_fact':r['fact_text'],'current_fact':f['text_description'],'old_label':r['original_coverage_label']})
     continue
    if r['original_coverage_label']=='consistent':
     oldpositive.add(fid)
     if not r['associations']:noattrpositive.add(fid)
     # Proxy only: old page-level coverage AND association AND own matching box.
     # Text of this particular associated citation has NOT been checked for correctness.
     for a in r['associations']:
      c=a['citation']
      if fe[fid]&matched[qid].get(boxkey(c),set()):proxychain.add(fid)
  # User-confirmed operational metric: fact coverage AND attributable citation
  # with its own valid page and individually covered box. No extra text rejudge.
  linked=set();linked_possible=set();witnesses={};link_gaps=[]
  for fid in fs:
   for pg in sorted(fp[fid]&hit):
    ar=citation.get((qid,pg,fid))
    if ar is not None and ar['fact_text']==fs[fid]['text_description']:
     for association in ar['associations']:
      c=association['citation'];k=boxkey(c)
      assert c['page']==pg
      if fe[fid]&matched[qid].get(k,set()):
       linked.add(fid);linked_possible.add(fid)
       witnesses.setdefault(fid,[]).append({'citation_id':association['citation_id'],'page':pg,'bbox':c['bbox'],'matching_evidence_ids':sorted(fe[fid]&matched[qid][k])})
      elif fe[fid]&unknown[qid].get(k,set()):linked_possible.add(fid)
    else:
     # Missing attribution is unknown only when an eligible own box could match.
     candidates=[c for c in cs if c.get('page')==pg and (fe[fid]&matched[qid].get(boxkey(c),set()) or fe[fid]&unknown[qid].get(boxkey(c),set()))]
     if candidates:
      linked_possible.add(fid);link_gaps.append({'fact_id':fid,'page':pg,'reason':'attribution_missing_or_reference_changed','candidate_citation_indices':[i for i,c in enumerate(cs) if c in candidates]})
  complete_facts=oldpositive&linked
  complete_possible_facts=(oldpositive|unknownfacts)&linked_possible
  complete=complete_facts==fs.keys();complete_possible=complete_possible_facts==fs.keys()
  assert not complete or complete_possible
  rec=dict(model=model,question_id=qid,reference_facts=len(fs),predicted_pages=len(pages),valid_pages=len(hit),eligible_facts=len(eligible),predicted_regions=len(pred),invalid_format_regions=sum(not validbox(c) for c in pred.values()),valid_regions=len(goodboxes),valid_regions_upper=len(upperboxes),covered_region_facts=len(regionfacts),covered_region_facts_upper=len(regionupper),correct_answer=bool(ans[qid].get('consistent')),complete=complete,complete_possible=complete_possible,complete_fact_ids=sorted(complete_facts),complete_possible_fact_ids=sorted(complete_possible_facts),complete_witnesses=witnesses,complete_link_gaps=link_gaps,confirmed_covered_facts=len(oldpositive),unresolved_fact_coverage=len(unknownfacts-oldpositive),covered_facts_without_attribution=len(noattrpositive),proxy_chain_facts=len(proxychain),proxy_complete=bool(proxychain==fs.keys()),missing_fact_coverage_page_fact_pairs=newpairs,changed_reference_pairs=changedpairs)
  allq.append(rec)
  for k,v in rec.items():
   if isinstance(v,(int,bool)):totals[k]+=v
 correct=sum(bool(r.get('consistent')) for qid,r in ans.items() if qid not in stale_answers)
 p=rate(totals['valid_pages'],totals['predicted_pages']);r=rate(totals['eligible_facts'],nf)
 rp=rate(totals['valid_regions'],totals['predicted_regions']);rr=rate(totals['covered_region_facts'],totals['eligible_facts'])
 rpu=rate(totals['valid_regions_upper'],totals['predicted_regions']);rru=rate(totals['covered_region_facts_upper'],totals['eligible_facts'])
 complete_count=sum(x['complete'] and x['correct_answer'] for x in allq if x['model']==model)
 complete_upper_count=sum(x['complete_possible'] and x['correct_answer'] for x in allq if x['model']==model)
 mc=sum(x['proxy_complete'] and x['correct_answer'] for x in allq if x['model']==model)
 allstats.append(dict(model=model,name=name,Page_P=p,Page_R=r,Page_F1=f1(p,r),Region_P=rp,Region_R=rr,Region_F1=f1(rp,rr),Region_P_upper=rpu,Region_R_upper=rru,Region_F1_upper=f1(rpu,rru),Fact_Coverage=rate(totals['confirmed_covered_facts'],totals['eligible_facts']),Fact_Coverage_upper=rate(totals['confirmed_covered_facts']+totals['unresolved_fact_coverage'],totals['eligible_facts']),Acc=rate(correct,len(B)),Acc_upper=rate(correct+len(stale_answers),len(B)),Acc_legacy=rate(sum(bool(r.get('consistent')) for r in ans.values()),len(B)),changed_answer_count=len(stale_answers),Complete_at_Correct=rate(complete_count,totals['correct_answer']),Complete_at_Correct_upper=rate(complete_upper_count,totals['correct_answer']),complete_correct_count=complete_count,complete_correct_upper_count=complete_upper_count,answer_correct=correct,answer_total=len(B),counts=dict(totals),region_missing_tasks=sum(t['model']==model for t in missingregions),diagnostics={'confirmed_fact_positive_over_expanded_eligible':rate(totals['confirmed_covered_facts'],totals['eligible_facts']),'proxy_complete_at_correct_NOT_VALIDATED':rate(mc,totals['correct_answer']),'proxy_complete_correct_count':mc,'citation_rows':len(citation),'attribution_associations':sum(len(v['associations']) for v in citation.values())}))
 gaps.extend(modelgaps)
 dump('metrics.json',allstats)
dump('pending_answer_records.json',changedanswers);dump('missing_region_tasks.json',missingregions);dump('missing_fact_coverage_page_fact_pairs.json',gaps);dump('changed_reference_facts.json',changed)
(OUT/'question_counts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allq))
dump('provenance.json',{'benchmark_path':str(ROOT/'HF-DocScope-v2-20260916/benchmark.json'),'benchmark_sha256':BENCHMARK_SHA256,'final_resolution_overlay':str(RESOLUTION),'alternative_fact_judge_overlay':str(FACT_REJUDGE),'answer_rejudge_overlay':str(ANSWER_REJUDGE),'answer_unknown_records':len(changedanswers),'test_questions':len(B),'answerable':len(A),'reference_facts':nf,'citation_results':str(CIT/'results'),'region_result_index':str(RUN/'reused_results.json'),'inference_sha256':hashes,'api_calls':0,'aggregation':'micro counts; answer/question complete are question proportions','region_matching':'individual prediction match.label==covered; exact page+bbox dedup; invalid normalized-coordinate predictions remain in denominator and cannot hit','missing_region_policy':'lower and upper bounds; retain full denominator','alternative_evidence':'accepted and partial both treated as acceptable supports','fact_and_complete_status':'Fact Coverage reuses unchanged old consistency judgments with OR over acceptable pages, and reports lower/upper bounds for missing or changed references. Complete@Correct uses user-confirmed operational conjunction: fact-level coverage AND an associated citation with its own valid page and covered box. It does not claim separately verified correctness of that citation text. Missing attribution/matching judgments yield lower/upper bounds.'})
lines=['# 主表计算结果（2026-09-18）','','测试集 729 题；可回答 678 题；参考事实 3,847 条。accepted 与 partial 扩充证据均纳入。Page/Region 采用约定的 micro 口径；Region 仅在命中有效证据页上计算，严格采用逐预测框的 covered 匹配。','', '**影响主表数值的缺口已补齐，九项指标均为确定数值。Fact Coverage 已合并旧判定、新增支持页判定及剩余 22 条补评；Acc 已完成重判；Complete@Correct 按用户确认的事实覆盖 + citation 关联 + 该 citation 的有效页面与 covered 框计算。** 原始参考及判定范围未变的 consistent/not_consistent 可以复用；完整性按每个参考事实分别检查：该事实被覆盖，且至少有一条与它关联的 citation，其自身页码有效、框 covered 对应事实的参考区域。不额外要求逐 citation 的文本正确性判定。','','| 模型 | Page P | Page R | Page F1 | Region P | Region R | Region F1 | Fact Coverage | Acc | Complete@Correct |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
def fmt(v):return '—' if v is None else f'{v:.2f}'
for s in allstats:
 vals=[fmt(s[k]) for k in ['Page_P','Page_R','Page_F1']]
 for k in ['Region_P','Region_R','Region_F1']:
  lo,hi=s[k],s[k+'_upper'];vals.append(fmt(lo) if lo==hi else fmt(lo)+'–'+fmt(hi))
 vals.extend([(fmt(s['Fact_Coverage']) if s['Fact_Coverage']==s['Fact_Coverage_upper'] else fmt(s['Fact_Coverage'])+'–'+fmt(s['Fact_Coverage_upper'])),fmt(s['Acc']) if s['Acc']==s['Acc_upper'] else fmt(s['Acc'])+'–'+fmt(s['Acc_upper']),(fmt(s['Complete_at_Correct']) if s['Complete_at_Correct']==s['Complete_at_Correct_upper'] else fmt(s['Complete_at_Correct'])+'–'+fmt(s['Complete_at_Correct_upper']))]);lines.append('| '+s['name']+' | '+' | '.join(vals)+' |')
lines+=['','所有数值均为百分比。Complete@Correct 的分母为测试集中回答正确的可回答题目。已确认此前 37 个待确定组合的最终状态；其余尚未补全的扩充页关联不会改变任何一道答案正确题目的完整性，因而不会影响 Complete@Correct；对答案错误题目的完整性不作这一保证。事实覆盖沿用原提示词的证据单元与语义判断规则；不额外引入逐 citation 文本正确性裁判。','','## 数据审计','','- 全部 18 个模型的 729 道测试题均有原始推理和旧答案判定；已核对答案判定对应当前推理。每模型有 7 题标准答案变更（126 条），另有 3 条旧答案裁判解析失败，共 129 条已按原提示词、原参数使用 Qwen3.6 Plus 完成重判；两条仅修复返回 JSON 的反斜杠转义，未修改裁判判断。原始缺口见 changed_answer_records.json，重判结果见 ../answer_rejudge_qwen36_20260918/。','- 区域推理源文件的 SHA-256 均与任务准备时冻结的源文件相同；当前参考区域和事实与区域任务快照相同。','- 有 17 个参考区域没有关联到任何参考事实。页面/区域 Precision 仍承认这些人工参考区域；事实单位的 Recall 只使用显式事实关联，不自行补造关联。清单见 reference_regions_without_fact_links.json。',
'- 非归一化或无效坐标框计入命中页上的预测框分母，但不能记为有效框；完全相同的页面+坐标框只计一次。',f'- 未通过校验的区域任务：{len(missingregions)} 个，详见 missing_region_tasks.json。',f'- 尚缺失的事实—页面覆盖判定：{len(gaps):,} 条，详见 missing_fact_coverage_page_fact_pairs.json。新增支持页的 2,247 条覆盖判定，以及剩余 22 条原始页/参考变更判定已完成。影响完整性数值的 57 条 citation 关联已补齐。',f'- 旧事实判定的参考文本与当前参考不一致：{len(changed)} 条模型记录，涉及 {len({(x["question_id"],x["fact_id"]) for x in changed})} 个独立参考事实，详见 changed_reference_facts.json。','','## 辅助诊断（Complete 代理值不是正式指标）','','第一列等于当前已确认的 Fact Coverage 下界。第二列为「旧事实正确性 + 引用关联 + 对应框命中」的 Complete 代理值，分母已更新为重判后的正确答案数，但没有逐引用正确性保证，不能作为正式 Complete@Correct 或其严格上下界。','','| 模型 | 合并原始页与扩充页已确认覆盖率 (%) | 旧事实命中+关联+同框命中的 Complete 代理值 (%) | 缺失页面—事实判定数 |','|---|---:|---:|---:|']
for s in allstats:lines.append('| '+s['name']+' | '+fmt(s['diagnostics']['confirmed_fact_positive_over_expanded_eligible'])+' | '+fmt(s['diagnostics']['proxy_complete_at_correct_NOT_VALIDATED'])+' | '+str(s['counts']['missing_fact_coverage_page_fact_pairs'])+' |')
lines+=['','## 补齐情况','','1. 2,247 条新增页判定及剩余 22 条事实判定均已完成，Fact Coverage 无未决记录。','2. 57 条会影响完整性结果的 citation 关联已补齐，Complete@Correct 的上下界已完全相等。','3. 两条区域返回的逐框 matches 保持原样，仅按原校验规则重新生成其汇总标签，通过原验证器；答案的 129 条缺口此前已补齐。','','复算：`python3 output/main_metrics_20260918/calculate.py`。`metrics.json` 保留模型汇总和原始计数；`question_counts.jsonl` 保留逐题计数；`provenance.json` 记录口径及来源。']
start=lines.index('## 辅助诊断（Complete 代理值不是正式指标）');end=lines.index('## 补齐情况');lines=lines[:start]+lines[end:]
assert all(s[k]==s[k+'_upper'] for s in allstats for k in ['Region_P','Region_R','Region_F1','Fact_Coverage','Acc','Complete_at_Correct']),'Some metric remains unresolved'
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print('DONE',len(allstats),'models; missing region',len(missingregions),'missing fact-page',len(gaps),'changed references',len(changed),flush=True)
