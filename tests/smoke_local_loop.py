"""Run two real Notebook experiments with synthetic data and deterministic model replies.

Run explicitly with runtime dependencies installed: python tests/smoke_local_loop.py
Only model responses are replaced; data staging, evaluation scripts, kernels,
workers, metric queues, history and cross-round artifact reuse run normally.
"""
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from bizmind.agent import Agent
from bizmind.config import Config
from bizmind.task_input import load_custom_task, stage_custom_data


SPLIT_CODE = '''import argparse, csv, json
from pathlib import Path
p = argparse.ArgumentParser()
for key in ('input_dir', 'public_dir', 'private_dir'):
    p.add_argument('--' + key)
a = p.parse_args()
source, public, private = map(Path, (a.input_dir, a.public_dir, a.private_dir))
rows = list(csv.DictReader((source / 'train_data/train.csv').open()))
train, validation = rows[:8], rows[8:]
def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
write(public / 'train_data/train.csv', ['id','x','y'], train)
write(public / 'validation_data/rows.csv', ['id','x'], [{'id':r['id'],'x':r['x']} for r in validation])
write(public / 'predict_data/predict.csv', ['id','x'], list(csv.DictReader((source / 'predict_data/predict.csv').open())))
write(public / 'validation_sample_predictions.csv', ['id','y'], [{'id':r['id'],'y':0} for r in validation])
(private / 'labels.json').write_text(json.dumps({r['id']:float(r['y']) for r in validation}))
print('Fixed 8/2 split; validation labels private.')
'''

EVALUATE_CODE = '''import argparse, csv, json
from pathlib import Path
p = argparse.ArgumentParser()
for key in ('public_dir', 'private_dir', 'pred'):
    p.add_argument('--' + key)
a = p.parse_args()
labels = json.loads((Path(a.private_dir) / 'labels.json').read_text())
rows = list(csv.DictReader(Path(a.pred).open()))
assert len(rows) == len(labels) and set(r['id'] for r in rows) == set(labels)
score = sum((float(r['y'])-labels[r['id']])**2 for r in rows)/len(rows)
report = {'score':score, 'success':True, 'message':''}
(Path(a.private_dir) / 'eval_report.json').write_text(json.dumps(report))
print(json.dumps(report))
'''


def training_code(linear):
    return '''import csv, json, statistics
from pathlib import Path
root = Path('../input/data')
train = list(csv.DictReader((root / 'train_data/train.csv').open()))
xs = [float(r['x']) for r in train]
ys = [float(r['y']) for r in train]
''' + ("slope, intercept = statistics.linear_regression(xs, ys)\n" if linear
       else "slope, intercept = 0.0, statistics.mean(ys)\n") + '''
for source, output in [('validation_data/rows.csv','validation_predictions.csv'), ('predict_data/predict.csv','predictions.csv')]:
    rows = list(csv.DictReader((root / source).open()))
    with open(output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['id','y'])
        writer.writeheader()
        writer.writerows({'id':r['id'], 'y':slope*float(r['x'])+intercept} for r in rows)
Path('model.json').write_text(json.dumps({'slope':slope, 'intercept':intercept}))
print('Model fitted; complete validation and business predictions written.')
'''


class EvaluationConversation:
    def __init__(self, *args, **kwargs):
        self.index = 0

    def add_message(self, **kwargs):
        pass

    def query(self, **kwargs):
        values = [('split_dataset.py', SPLIT_CODE), ('evaluate.py', EVALUATE_CODE), ('None', 'None')]
        filename, code = values[self.index]
        self.index += 1
        return {'current_file':[filename], 'code':[code]}


class CodingConversation:
    count = 0

    def __init__(self, *args, **kwargs):
        self.linear = type(self).count > 0
        type(self).count += 1

    def add_message(self, **kwargs):
        pass

    def query(self, required_fields, **kwargs):
        if 'code' in required_fields and 'goal' in required_fields:
            return {'code':[training_code(self.linear)], 'goal':['Fit and validate the local hypothesis'],
                    'prediction_file':['predictions.csv'], 'validation_prediction_file':['validation_predictions.csv']}
        return {'description':['Measured local linear regression' if self.linear else 'Measured mean baseline'],
                'code':[training_code(self.linear)], 'suggestions':['Compare the next method on the fixed split']}


planner_round = 0


def planner_reply(cfg, messages, required_fields, **kwargs):
    global planner_round
    if required_fields == ['is_lower_better']:
        return {'is_lower_better':['True']}
    if required_fields == ['solution']:
        return {'solution':['<description>Local regression hypothesis</description><idea>Fit a baseline</idea>']*4}
    if required_fields == ['idea']:
        return {'idea':['Fit baseline; then test a linear relationship']}
    parent = 'None' if planner_round == 0 else 'round-001-experiment-01'
    planner_round += 1
    return {'pipeline':[f'<title>Local model</title><description>Fit a local model and measure MSE</description>'
                        f'<codebase>{parent}</codebase><code>Fit model from local data</code>']}


def assemble_notebook(cfg, messages, **kwargs):
    cells = re.findall(r'<cell>\n(.*?)\n</cell>', messages[0]['content'], flags=re.S)
    return {'code':['\n\n'.join(cells)]}


def main():
    with TemporaryDirectory(prefix='bizmind-smoke-') as directory:
        root = Path(directory)
        (root / 'train.csv').write_text('id,x,y\n' + ''.join(f'{i},{i},{2*i+1}\n' for i in range(10)))
        (root / 'predict.csv').write_text('id,x\n10,10\n11,11\n')
        (root / 'task.yaml').write_text('objective: predict y\ndata: {train: train.csv, predict: predict.csv, target_column: y}\nevaluation: {primary_metric: mse, direction: minimize}\n')
        cfg = Config(task_input_file=root / 'task.yaml', agent_workspace_dir=root / 'run',
                     agent_num_iterations=2, agent_num_iterations_code_agents=3,
                     execution_device='cpu', execution_timeout=30, execution_inspect_interval=20)
        cfg.task_description = stage_custom_data(cfg, load_custom_task(cfg))
        with patch('bizmind.agent.query_llm', side_effect=planner_reply), \
             patch('bizmind.evaluate.Conversation', EvaluationConversation), \
             patch('bizmind.coder.Conversation', CodingConversation), \
             patch('bizmind.utils.jupyter_session.query_llm', side_effect=assemble_notebook):
            agent = Agent(cfg)
            agent.run()
        best = json.loads((root / 'run/best.json').read_text())
        history = json.loads((root / 'run/experiments.json').read_text())
        assert best['score'] == 0.0, best
        assert len(history['experiments']) == 2 and not history['failed_attempts'], history
        assert history['experiments'][0]['score'] > history['experiments'][1]['score']
        assert '10,21.0' in Path(best['prediction_file']).read_text()
        assert (root / 'run/round-002-experiment-01/working/model.json').is_file()
        assert 'y' not in (root / 'run/evaluation/public/validation_data/rows.csv').read_text().splitlines()[0]
        print('PASS: 2 real Notebook rounds, local MSE improved from',
              history['experiments'][0]['score'], 'to', best['score'], '; predictions and history verified.')


if __name__ == '__main__':
    main()
