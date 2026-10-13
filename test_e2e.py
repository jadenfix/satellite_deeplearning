import unittest,subprocess,tempfile,sys,json,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).parent
class E2E(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.start()
 def start(self):
  self.proc=subprocess.Popen([sys.executable,str(ROOT/'app.py'),'--port','0','--db',self.tmp.name+'/db.sqlite3'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True);self.url=self.proc.stdout.readline().strip()
 def tearDown(self):self.proc.terminate();self.proc.wait(timeout=5);self.proc.stdout.close();self.tmp.cleanup()
 def request(self,path,data=None):
  req=urllib.request.Request(self.url+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read())
 def restart(self):self.proc.terminate();self.proc.wait(timeout=5);self.proc.stdout.close();self.start()
 def test_train_holdout_restart(self):
  d=self.request('/api/runs',{'seed':42});self.assertGreater(d['water_iou'],.85);self.assertEqual(len(d['scenes']),4);self.assertFalse(set(d['train_image_seeds'])&set(d['heldout_image_seeds']));self.assertEqual(d['reload_max_logit_difference'],0.0);self.assertLess(d['training_loss'][-1],d['training_loss'][0]);self.assertEqual(d['model']['type'],'CPU_3_layer_CNN');self.assertEqual(d['test_pixels'],sum(d['confusion'].values()));self.restart();self.assertEqual(self.request('/api/runs/'+str(d['id']))['model'],d['model']);self.assertEqual(len(self.request('/api/runs')['items']),1)
 def test_validation_and_determinism(self):
  for data in [[],{'seed':True},{'seed':-1},{'seed':'42'}]:
   with self.assertRaises(urllib.error.HTTPError) as e:self.request('/api/runs',data)
   self.assertEqual(e.exception.code,400)
  a=self.request('/api/runs',{'seed':3});b=self.request('/api/runs',{'seed':3});self.assertEqual(a['water_iou'],b['water_iou']);self.assertEqual(a['model'],b['model'])
if __name__=='__main__':unittest.main()
