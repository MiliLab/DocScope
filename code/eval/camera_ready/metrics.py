"""Camera-ready micro metrics. Inputs contain validated, individual judgments."""
import math
from collections import Counter


def boxkey(c):
    return c['page'], tuple(c.get('bbox') or [])


def validbox(c):
    b = c.get('bbox') or []
    return (c.get('is_normalized') and len(b) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in b)
            and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1)


def count_question(q, inference, judgments):
    correct = judgments['answer_correct']
    if type(correct) is not bool:
        raise ValueError('Answer judgment must be boolean')
    counts = dict(answer_total=1, answer_correct=int(correct))
    if not q['answer']['is_answerable']:
        return counts
    es = {e['local_id']: e for e in q['evidences']}
    fs = {f['local_id']: f for f in q['facts']}
    if not fs:
        raise ValueError('Answerable question has no reference facts')
    links = {fid: set(f.get('evidence_local_ids') or [f['evidence_local_id']]) for fid, f in fs.items()}
    pages = set(inference['parsed']['predicted_pages'])
    hit = pages & {e['page'] for e in es.values()}
    fp = {fid: {es[e]['page'] for e in ids} for fid, ids in links.items()}
    eligible = {fid for fid in fs if fp[fid] & hit}
    cs = inference['parsed']['citations']
    pred = {boxkey(c) for c in cs if c['page'] in hit}
    matched = {}
    for r in judgments['regions']:
        c = cs[r['citation_index']]
        if not validbox(c) or c['page'] not in hit:
            raise ValueError('Region judgment refers to an invalid prediction')
        ids = set(r['covered_evidence_ids'])
        if any(e not in es or es[e]['page'] != c['page'] for e in ids):
            raise ValueError('Region match crosses pages or references unknown evidence')
        matched.setdefault(boxkey(c), set()).update(ids)
    expected = {boxkey(c) for c in cs if c['page'] in hit and validbox(c)}
    if set(matched) != expected:
        raise ValueError('Missing individual region judgments')
    covered_e = set().union(*matched.values()) if matched else set()
    coverage = set()
    linked = set()
    seen = set()
    for r in judgments['facts']:
        fid, pg = r['fact_id'], r['page']
        if (fid, pg) in seen or fid not in fs or pg not in fp[fid] & hit:
            raise ValueError('Invalid or duplicate fact-page judgment')
        seen.add((fid, pg))
        if r['label'] not in ('consistent', 'not_consistent'):
            raise ValueError('Unknown fact judgment')
        if r['label'] == 'consistent':
            coverage.add(fid)
        for i in r['citation_indices']:
            c = cs[i]
            if c['page'] != pg:
                raise ValueError('Attribution crosses pages')
            if links[fid] & matched.get(boxkey(c), set()):
                linked.add(fid)
    if seen != {(fid, pg) for fid in fs for pg in fp[fid] & hit}:
        raise ValueError('Missing fact or attribution judgments')
    counts.update(reference_facts=len(fs), predicted_pages=len(pages), valid_pages=len(hit),
        eligible_facts=len(eligible), predicted_regions=len(pred),
        valid_regions=sum(bool(ids) for ids in matched.values()),
        covered_region_facts=sum(bool(links[fid] & covered_e) for fid in eligible),
        covered_facts=len(coverage), correct_answerable=int(correct),
        complete_correct=int(correct and coverage & linked == set(fs)))
    return counts


def aggregate(rows):
    c = Counter()
    for r in rows:
        c.update(r)
    def rate(n, d):
        return 100 * c[n] / c[d] if c[d] else None
    def f1(p, r):
        return None if p is None or r is None else (2*p*r/(p+r) if p+r else 0.0)
    pp, pr = rate('valid_pages', 'predicted_pages'), rate('eligible_facts', 'reference_facts')
    rp, rr = rate('valid_regions', 'predicted_regions'), rate('covered_region_facts', 'eligible_facts')
    return dict(Page_P=pp, Page_R=pr, Page_F1=f1(pp, pr), Region_P=rp, Region_R=rr,
                Region_F1=f1(rp, rr), Fact_Coverage=rate('covered_facts', 'eligible_facts'),
                Acc=rate('answer_correct', 'answer_total'),
                Complete_at_Correct=rate('complete_correct', 'correct_answerable'), counts=dict(c))
