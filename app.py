"""Local CPU segmentation experiment viewer; generated fixtures only."""
import argparse,json,sqlite3
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pipeline import train
ROOT=Path(__file__).parent

def handler(db):
 def conn():
  c=sqlite3.connect(db);c.execute('CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY,result TEXT)');return c
 class H(BaseHTTPRequestHandler):
  def send(self,status,data,mime='application/json'):
   b=json.dumps(data,allow_nan=False).encode() if mime=='application/json' else data;self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(b)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'");self.end_headers();self.wfile.write(b)
  def do_GET(self):
   if self.path=='/':return self.send(200,(ROOT/'index.html').read_bytes(),'text/html')
   if self.path=='/app.js':return self.send(200,(ROOT/'app.js').read_bytes(),'text/javascript')
   if self.path=='/api/runs':
    with conn() as c:rows=c.execute('SELECT id,result FROM runs ORDER BY id DESC LIMIT 20').fetchall()
    return self.send(200,{'items':[{'id':i,'seed':json.loads(r)['seed']} for i,r in rows]})
   if self.path.startswith('/api/runs/') and self.path[10:].isdigit():
    with conn() as c:row=c.execute('SELECT result FROM runs WHERE id=?',(self.path[10:],)).fetchone()
    return self.send(200,json.loads(row[0])) if row else self.send(404,{'error':'Run not found'})
   self.send(404,{'error':'Not found'})
  def do_POST(self):
   if self.path!='/api/runs':return self.send(404,{'error':'Not found'})
   origin=self.headers.get('Origin')
   if origin and origin!='http://'+self.headers.get('Host',''):return self.send(403,{'error':'Cross-origin writes disabled'})
   try:
    length=int(self.headers.get('Content-Length','0'))
    if not 1<=length<=4096:return self.send(413,{'error':'Maximum request size 4096 bytes'})
    data=json.loads(self.rfile.read(length))
    if not isinstance(data,dict):raise ValueError('Expected JSON object')
    result=train(data.get('seed',42))
   except (ValueError,TypeError) as e:return self.send(400,{'error':str(e)})
   with conn() as c:
    cur=c.execute('INSERT INTO runs(result) VALUES(?)',(json.dumps(result),));result['id']=cur.lastrowid;c.execute('UPDATE runs SET result=? WHERE id=?',(json.dumps(result),result['id']))
   self.send(201,result)
 return H
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8084);p.add_argument('--db',default=str(ROOT/'experiments.sqlite3'));a=p.parse_args();s=ThreadingHTTPServer(('127.0.0.1',a.port),handler(a.db));print(f'http://127.0.0.1:{s.server_port}',flush=True);s.serve_forever()
if __name__=='__main__':main()
