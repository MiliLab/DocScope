"""Reproduce published metrics from audited per-question counts (no API calls)."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def aggregate(rows, reference):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['model']].append(row)
    results = []
    for ref in reference:
        rr = groups[ref['model']]
        if len(rr) != 678 or len({r['question_id'] for r in rr}) != 678:
            raise ValueError('Expected 678 unique answerable test questions per model')
        def total(key):
            return sum(r[key] for r in rr)
        def rate(n, d):
            return 100 * n / d if d else None
        def f1(p, r):
            return 2*p*r/(p+r) if p is not None and r is not None and p+r else 0
        pp = rate(total('valid_pages'), total('predicted_pages'))
        pr = rate(total('eligible_facts'), total('reference_facts'))
        rp = rate(total('valid_regions'), total('predicted_regions'))
        rrec = rate(total('covered_region_facts'), total('eligible_facts'))
        results.append(dict(model=ref['model'], Page_P=pp, Page_R=pr,
            Page_F1=f1(pp, pr), Region_P=rp, Region_R=rrec, Region_F1=f1(rp, rrec),
            Fact_Coverage=rate(total('confirmed_covered_facts'), total('eligible_facts')),
            Acc=rate(ref['answer_correct'], ref['answer_total']),
            Complete_at_Correct=rate(sum(r['complete'] and r['correct_answer'] for r in rr), total('correct_answer'))))
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', type=Path, default=Path(__file__).with_name('reference_counts.jsonl.gz'))
    p.add_argument('--reference', type=Path, default=Path(__file__).with_name('reference_metrics.json'))
    a = p.parse_args()
    with gzip.open(a.counts, 'rt') as f:
        rows = [json.loads(line) for line in f]
    reference = json.loads(a.reference.read_text())
    result = aggregate(rows, reference)
    for actual, expected in zip(result, reference):
        for key, value in actual.items():
            if value != expected[key]:
                raise ValueError(f'Mismatch: {actual["model"]} {key}')
    print(json.dumps(result, indent=2))
