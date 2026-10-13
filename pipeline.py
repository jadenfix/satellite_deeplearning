"""CPU land/water pixel classifier on explicitly generated RGB tiles, not satellite evidence."""
import base64,io,json
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

def fixture(seed,size=48):
 rng=np.random.default_rng(seed);y,x=np.mgrid[:size,:size];center=size*(.35+.2*rng.random())+size*.17*np.sin(y/size*6+rng.random()*5);water=abs(x-center)<size*(.09+.05*rng.random())
 image=np.where(water[...,None],np.array([.12,.32,.58]),np.array([.34,.48,.23]))+rng.normal(0,.07,(size,size,3))+rng.uniform(-.07,.07,(1,1,3))
 return np.clip(image,0,1),water.astype(int)
def png(array):
 b=io.BytesIO();Image.fromarray(np.uint8(np.clip(array,0,1)*255)).save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def train(seed=42):
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=100000:raise ValueError('Seed must be an integer 0–100000')
 train_seeds=[seed*100+v for v in range(12)];test_seeds=[seed*100+v for v in range(100,104)]
 train=[fixture(s) for s in train_seeds];x=np.concatenate([im.reshape(-1,3) for im,_ in train]);y=np.concatenate([mask.ravel() for _,mask in train]);model=LogisticRegression(max_iter=250,random_state=seed).fit(x,y)
 scenes=[];tp=fp=fn=tn=0
 for s in test_seeds:
  im,target=fixture(s);pred=model.predict(im.reshape(-1,3)).reshape(target.shape);tp+=int(np.sum((target==1)&(pred==1)));fp+=int(np.sum((target==0)&(pred==1)));fn+=int(np.sum((target==1)&(pred==0)));tn+=int(np.sum((target==0)&(pred==0)));color=np.array([[.35,.53,.26],[.20,.57,.95]])
  scenes.append({'seed':s,'rgb':png(im),'truth':png(color[target]),'prediction':png(color[pred]),'water_iou':float(np.sum((target==1)&(pred==1))/np.sum((target==1)|(pred==1)))})
 return {'source':'generated_RGB_land_water_fixtures','seed':seed,'train_image_seeds':train_seeds,'heldout_image_seeds':test_seeds,'train_pixels':len(y),'test_pixels':tp+fp+fn+tn,'water_iou':tp/(tp+fp+fn),'accuracy':(tp+tn)/(tp+fp+fn+tn),'confusion':{'true_water':tp,'false_water':fp,'missed_water':fn,'true_land':tn},'model':{'type':'LogisticRegression','features':['R','G','B'],'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist()},'scenes':scenes,'limits':'Generated image holdout only. No real satellite sensor bands, cloud handling, geospatial alignment or cross-region generalization validated.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='result.json');a=p.parse_args();r=train(a.seed);open(a.output,'w').write(json.dumps(r));print(json.dumps({k:v for k,v in r.items() if k!='scenes'},indent=2))
