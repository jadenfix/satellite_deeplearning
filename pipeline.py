"""CPU land/water pixel classifier on explicitly generated RGB tiles, not satellite evidence."""
import base64,io,json,threading
import torch
from torch import nn
TRAIN_LOCK=threading.Lock()
torch.set_num_threads(1)
class Segmenter(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,2,1))
 def forward(self,x):return self.net(x)
def restore_model(model_data):
 model=Segmenter();model.load_state_dict(torch.load(io.BytesIO(base64.b64decode(model_data["state_dict_base64"])),map_location="cpu",weights_only=True));model.eval();return model
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

def fixture(seed,size=48):
 rng=np.random.default_rng(seed);y,x=np.mgrid[:size,:size];center=size*(.35+.2*rng.random())+size*.17*np.sin(y/size*6+rng.random()*5);water=abs(x-center)<size*(.09+.05*rng.random())
 image=np.where(water[...,None],np.array([.12,.32,.58]),np.array([.34,.48,.23]))+rng.normal(0,.07,(size,size,3))+rng.uniform(-.07,.07,(1,1,3))
 return np.clip(image,0,1),water.astype(int)
def png(array):
 b=io.BytesIO();Image.fromarray(np.uint8(np.clip(array,0,1)*255)).save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def _train(seed=42):
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=100000:raise ValueError('Seed must be an integer 0–100000')
 train_seeds=[seed*100+v for v in range(12)];test_seeds=[seed*100+v for v in range(100,104)]
 train=[fixture(s) for s in train_seeds];x=np.concatenate([im.reshape(-1,3) for im,_ in train]);y=np.concatenate([mask.ravel() for _,mask in train]);model=LogisticRegression(max_iter=250,random_state=seed).fit(x,y)
 torch.manual_seed(seed)
 network=Segmenter()
 tensors=torch.tensor(np.stack([im.transpose(2,0,1) for im,_ in train]),dtype=torch.float32)
 labels=torch.tensor(np.stack([mask for _,mask in train]),dtype=torch.long)
 optimizer=torch.optim.Adam(network.parameters(),lr=.012)
 losses=[]
 network.train()
 for _ in range(60):
  optimizer.zero_grad();loss=nn.functional.cross_entropy(network(tensors),labels);loss.backward();optimizer.step();losses.append(float(loss.detach()))
 network.eval()
 buffer=io.BytesIO();torch.save(network.state_dict(),buffer)
 model_data={'type':'CPU_3_layer_CNN','features':['R','G','B'],'channels':[3,12,12,2],'training_steps':60,'state_dict_base64':base64.b64encode(buffer.getvalue()).decode()}
 restored=restore_model(model_data)
 scenes=[];tp=fp=fn=tn=0;btp=bfp=bfn=0;max_reload_difference=0.0
 for s in test_seeds:
  im,target=fixture(s)
  input_tensor=torch.tensor(im.transpose(2,0,1)[None],dtype=torch.float32)
  with torch.no_grad():
   logits=network(input_tensor);reload_logits=restored(input_tensor);max_reload_difference=max(max_reload_difference,float(torch.max(torch.abs(logits-reload_logits))));pred=logits.argmax(1)[0].numpy()
  baseline=model.predict(im.reshape(-1,3)).reshape(target.shape)
  btp+=int(np.sum((target==1)&(baseline==1)));bfp+=int(np.sum((target==0)&(baseline==1)));bfn+=int(np.sum((target==1)&(baseline==0)))
  tp+=int(np.sum((target==1)&(pred==1)));fp+=int(np.sum((target==0)&(pred==1)));fn+=int(np.sum((target==1)&(pred==0)));tn+=int(np.sum((target==0)&(pred==0)));color=np.array([[.35,.53,.26],[.20,.57,.95]])
  scenes.append({'seed':s,'rgb':png(im),'truth':png(color[target]),'prediction':png(color[pred]),'baseline_prediction':png(color[baseline]),'water_iou':float(np.sum((target==1)&(pred==1))/np.sum((target==1)|(pred==1)))})
 return {'source':'generated_RGB_land_water_fixtures','seed':seed,'train_image_seeds':train_seeds,'heldout_image_seeds':test_seeds,'train_pixels':len(y),'test_pixels':tp+fp+fn+tn,'water_iou':tp/(tp+fp+fn),'accuracy':(tp+tn)/(tp+fp+fn+tn),'confusion':{'true_water':tp,'false_water':fp,'missed_water':fn,'true_land':tn},'model':model_data,'reload_max_logit_difference':max_reload_difference,'training_loss':losses,'baseline':{'type':'LogisticRegression','water_iou':btp/(btp+bfp+bfn),'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist()},'scenes':scenes,'limits':'Generated image holdout only. No real satellite sensor bands, cloud handling, geospatial alignment or cross-region generalization validated.'}
def train(seed=42):
 # Torch random generator and threading settings are process-wide; serialize local experiments.
 with TRAIN_LOCK:return _train(seed)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='result.json');a=p.parse_args();r=train(a.seed);open(a.output,'w').write(json.dumps(r));print(json.dumps({k:v for k,v in r.items() if k!='scenes'},indent=2))
