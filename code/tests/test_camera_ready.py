import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'eval/camera_ready'))
from metrics import aggregate, count_question


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.q={'id':'q','question':'Q','pdf':{'doc_id_str':'doc'},'answer':{'is_answerable':True,'answer_text':'yes'},
          'evidences':[{'local_id':'e1','page':1,'bbox':[0,0,10,10]},
                       {'local_id':'e2','page':2,'bbox':[0,0,10,10]}],
          'facts':[{'local_id':'f1','text_description':'F','evidence_local_id':'e1','evidence_local_ids':['e1','e2']}]}
        self.c={'page':2,'bbox':[0.1,0.1,0.2,0.2],'is_normalized':True,'span_start':2}
        self.inf={'status':'ok','model_raw':'F [citation]','parsed':{'answer':'yes','predicted_pages':[2,2], 'citations':[self.c]}}
        self.j={'answer_correct':True,'regions':[{'citation_index':0,'covered_evidence_ids':['e2']}],
                'facts':[{'fact_id':'f1','page':2,'label':'consistent','citation_indices':[0]}]}

    def test_alternative_support_and_duplicate_page(self):
        r=aggregate([count_question(self.q,self.inf,self.j)])
        for k,v in r.items():
            if k!='counts': self.assertEqual(v,100,k)

    def test_same_citation_box_required(self):
        other={**self.c,'bbox':[0.7,0.7,0.8,0.8]}
        self.inf['parsed']['citations'].append(other)
        self.j['regions'].append({'citation_index':1,'covered_evidence_ids':[]})
        self.j['facts'][0]['citation_indices']=[1]
        r=aggregate([count_question(self.q,self.inf,self.j)])
        self.assertEqual(r['Region_P'],50)
        self.assertEqual(r['Fact_Coverage'],100)
        self.assertEqual(r['Complete_at_Correct'],0)

    def test_duplicate_box_invalid_box_and_unanswerable(self):
        self.inf['parsed']['citations'] += [dict(self.c),{**self.c,'bbox':[0,0,2,2]}]
        c=count_question(self.q,self.inf,self.j)
        self.assertEqual(c['predicted_regions'],2)
        q={'answer':{'is_answerable':False}}
        r=aggregate([c,count_question(q,{}, {'answer_correct':False})])
        self.assertEqual(r['Acc'],50)
        self.assertEqual(r['Complete_at_Correct'],100)

    def test_missing_judgments_fail(self):
        self.j['regions']=[]
        with self.assertRaises(ValueError): count_question(self.q,self.inf,self.j)

    def test_coverage_or_over_pages(self):
        self.inf['parsed']['predicted_pages']=[1,2]
        self.j['facts'].append({'fact_id':'f1','page':1,'label':'not_consistent','citation_indices':[]})
        r=aggregate([count_question(self.q,self.inf,self.j)])
        self.assertEqual(r['Fact_Coverage'],100)
        self.assertEqual(r['Page_R'],100)

    def test_judge_pipeline_with_mock_responses(self):
        import score
        responses=[{'consistent':True},
           {'predictions':[{'id':'p1','matches':[{'gold_id':'g1','label':'covered'}]}]},
           {'items':[{'id':'f1','label':'consistent'}]},
           {'candidate_checks':[{'citation_id':'c1','associated':True}],
            'associations':[{'citation_id':'c1','evidence_quote':'F','context_quote':''}]}]
        def fake(*args,**kwargs): return responses.pop(0), {'raw':'mock','usage':{}}
        with tempfile.TemporaryDirectory() as d, patch.object(score,'judge',side_effect=fake), patch.object(score.render_eval_page,'render_pair',return_value=True):
            r=score.score_question(self.q,self.inf,Path(d),dict(text=None,region=None,attribution=None),dict(text='t',region='r',attribution='a'))
        self.assertEqual(r['counts']['complete_correct'],1)
        self.assertFalse(responses)


if __name__=='__main__': unittest.main()
