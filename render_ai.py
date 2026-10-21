from PIL import Image, ImageDraw, ImageFont
import numpy as np, math, subprocess, os
from pathlib import Path
ROOT=Path(__file__).parent
W,H,FPS,CHAPTER=900,1200,20,27
DURATION=7*CHAPTER
WHITE='#eef5ff'; MUTED='#b9cbe2'; CYAN='#43e1ff'; PURPLE='#b99aff'; GOLD='#ffce66'; GREEN='#5aefb5'; PINK='#ff89b3'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fonts={}
def font(n,b=False):
 k=(n,b)
 if k not in fonts: fonts[k]=ImageFont.truetype(BOLD if b else FONT,n)
 return fonts[k]
def txt(d,xy,s,n=28,c=WHITE,b=False):d.text(xy,s,font=font(n,b),fill=c)
def lines(d,xy,s,n=29,c=WHITE,width=790,gap=10,b=False):
 x,y=xy
 for para in s.split('\n'):
  line=''
  for word in para.split():
   trial=(line+' '+word).strip()
   if d.textlength(trial,font=font(n,b))>width and line:
    txt(d,(x,y),line,n,c,b);y+=n+gap;line=word
   else:line=trial
  txt(d,(x,y),line,n,c,b);y+=n+gap
 return y
def box(d,rect,title,sub='',color=CYAN):
 x,y,x2,y2=rect
 d.rounded_rectangle((x+6,y+9,x2+6,y2+9),18,fill='#020712')
 d.rounded_rectangle(rect,18,fill='#15253b',outline='#3d536b',width=2)
 d.rounded_rectangle((x+1,y+15,x+5,y2-15),2,fill=color)
 txt(d,(x+21,y+15),title,28,color,True)
 if sub:lines(d,(x+21,y+57),sub,24,MUTED,x2-x-40,7)
def arrow(d,a,b,t,color=CYAN):
 d.line([a,b],fill='#34485f',width=3)
 dx,dy=b[0]-a[0],b[1]-a[1];l=math.hypot(dx,dy);ux,uy=dx/l,dy/l
 d.polygon([b,(b[0]-12*ux+6*uy,b[1]-12*uy-6*ux),(b[0]-12*ux-6*uy,b[1]-12*uy+6*ux)],fill=color)
 for offset in [0,.33,.66]:
  q=(t*.32+offset)%1;x=a[0]+q*dx;y=a[1]+q*dy
  d.ellipse((x-5,y-5,x+5,y+5),fill=color)
def chip(d,x,y,w,h,t,label='TRANSFORMER'):
 for i in range(12):
  px=x+20+i*(w-40)/11
  d.line((px,y-13,px,y),fill='#a5b9c9',width=5);d.line((px,y+h,px,y+h+13),fill='#a5b9c9',width=5)
 d.rounded_rectangle((x,y,x+w,y+h),12,fill='#223c54',outline='#93a8bc',width=3)
 d.rounded_rectangle((x+15,y+15,x+w-15,y+h-15),8,fill='#071525',outline=CYAN,width=2)
 for i in range(18):
  for j in range(5):
   v=(math.sin(i*.7+j+t*3)+1)/2
   d.rectangle((x+26+i*(w-52)/18,y+30+j*18,x+32+i*(w-52)/18,y+36+j*18),fill=(int(35+25*v),int(80+110*v),int(110+140*v)))
 txt(d,(x+25,y+h-55),label,23,CYAN,True)
titles=['TOKENS','EMBEDDINGS','TRANSFORMERS','PROMPTING','RAG: BUILD THE INDEX','RAG: RETRIEVE + GENERATE','HOW IT ALL FITS TOGETHER']
subs=['Text becomes discrete pieces','Numbers encode useful relationships','Context changes each token’s representation','Instructions guide the model’s response','Prepare documents before questions arrive','Find evidence, then write an answer','One question → a grounded response']
captions=[
 [('A tokenizer splits text into tokens: words, word fragments, punctuation, or other units.'),('Each token maps to an integer ID. These IDs select learned vectors inside the language model.'),('Splits and IDs depend on the tokenizer. This example is illustrative, not a real tokenizer output.')],
 [('Inside an LLM, each token ID selects a learned embedding: a vector of numbers.'),('Position information is added or incorporated so word order matters. Transformer layers then add context.'),('For search, a separate embedding model can encode a whole passage or query. Similarity helps find relevant text.')],
 [('A decoder transformer uses masked self-attention: each position can use earlier tokens, not future ones.'),('Query, key, and value vectors help compute attention. Feed-forward layers transform the result; residual paths and normalization stabilize it.'),('The output becomes next-token probabilities. A token is chosen, appended, and the generation cycle repeats.')],
 [('A prompt is the instructions and context supplied to the model. State the task, evidence, and desired format.'),('A clear prompt can ask for citations and an honest “not found” when evidence is missing.'),('Prompting steers behavior at inference time. It does not update model weights or guarantee a correct answer.')],
 [('RAG means retrieval-augmented generation. First, split source documents into manageable chunks.'),('An embedding model converts chunks into vectors. Store vectors with their text and source metadata.'),('The index can be refreshed without retraining the language model. Access permissions and document quality still matter.')],
 [('Embed the question using a compatible retrieval model, then search for relevant chunks. Keyword search and reranking may also help.'),('Put retrieved text into the prompt with the question. The transformer generates an answer from that supplied context.'),('Check citations against the actual source. RAG can reduce unsupported answers, but retrieval and generation can still fail.')],
 [('Prompt = task. Tokens = text units. Embeddings = numeric representations. Transformer = contextual processing and generation.'),('RAG adds external evidence before generation. It is an application workflow around the language model.'),('Learning happens during training. Here, inference uses fixed model weights to produce an answer, one token at a time.')]
]
yy=np.linspace(0,1,H)[:,None,None];bg=np.zeros((H,W,3),np.uint8)
bg[:]=np.array([6,13,25])*(1-yy)+np.array([14,29,47])*yy
base=Image.fromarray(bg)
def frame(t):
 im=base.copy();d=ImageDraw.Draw(im);s=min(6,int(t//CHAPTER));u=t-s*CHAPTER;p=u/CHAPTER
 for x in range(0,W,50):d.line((x,190,x,865),fill='#14283c')
 for y in range(215,866,50):d.line((40,y,860,y),fill='#14283c')
 txt(d,(48,34),'AI / INSIDE THE PIPELINE',23,CYAN,True)
 txt(d,(48,77),titles[s],36 if s in (4,5,6) else 48,WHITE,True)
 txt(d,(48,143),subs[s],25,MUTED)
 if s==0:
  box(d,(50,220,850,315),'QUESTION','How long does the robot battery last?')
  arrow(d,(450,320),(450,375),t)
  txt(d,(50,380),'ILLUSTRATIVE TOKEN SPLIT',24,GOLD,True)
  tokens=['How',' long',' does',' the',' robot',' battery',' last','?']
  ids=[31,84,57,12,205,318,96,9]
  for i,(token,tid) in enumerate(zip(tokens,ids)):
   x=50+(i%4)*202;y=435+(i//4)*145
   active=i<=int(u*1.2)
   box(d,(x,y,x+188,y+116),token.strip(),f'ID: {tid}',CYAN if active else MUTED)
  arrow(d,(450,706),(450,755),t)
  txt(d,(169,776),'TOKEN IDs → VECTOR LOOKUP',27,GREEN,True)
 elif s==1:
  box(d,(50,220,360,335),'TOKEN: battery','ID → learned vector',GOLD)
  arrow(d,(370,277),(490,277),t)
  box(d,(500,220,850,335),'EMBEDDING','[0.12, −0.44, 0.71, …]',CYAN)
  txt(d,(50,370),'RETRIEVAL EMBEDDING SPACE',25,CYAN,True)
  txt(d,(50,406),'Schematic 2D projection; real vectors have many dimensions.',21,MUTED)
  d.line((85,750,815,750),fill='#5d738b',width=2);d.line((85,750,85,465),fill='#5d738b',width=2)
  points=[(305,550,'battery',GOLD),(445,515,'runtime',GREEN),(390,637,'charging',CYAN),(700,670,'weather',PURPLE)]
  for x,y,label,c in points:
   r=10+2*math.sin(t*2);d.ellipse((x-r,y-r,x+r,y+r),fill=c);txt(d,(x+18,y-15),label,26,c)
  arrow(d,(305,550),(445,515),t,GREEN)
  txt(d,(140,795),'Nearby can mean semantically related.',28,WHITE,True)
 elif s==2:
  words=['robot','battery','lasts']
  for i,w in enumerate(words):box(d,(55+i*277,220,295+i*277,305),w,color=[CYAN,GOLD,GREEN][i])
  for i in range(3):
   for j in range(i+1):arrow(d,(175+j*277,315),(175+i*277,420),t+.3*i,PURPLE)
  box(d,(60,425,840,530),'MASKED MULTI-HEAD ATTENTION','Attention(Q,K,V) = softmax(QKᵀ / √dₖ + mask)V',PURPLE)
  arrow(d,(450,535),(450,570),t)
  box(d,(60,575,840,665),'FEED-FORWARD + RESIDUAL + NORM','Repeated blocks refine contextual representations.',CYAN)
  txt(d,(60,695),'NEXT TOKEN • illustrative probabilities',25,GOLD,True)
  for i,(label,v,c) in enumerate([('8',.65,GREEN),('6',.22,CYAN),('other',.13,PURPLE)]):
   y=738+i*42;txt(d,(65,y),label,25);d.rounded_rectangle((175,y+5,175+v*750,y+28),6,fill=c);txt(d,(720,y),f'{v:.0%}',23)
 elif s==3:
  data=[('TASK','Answer the battery-life question.',CYAN),('CONTEXT','Robot manual, section 4: up to 8 hours\nunder the stated test conditions.',GOLD),('FORMAT','Two sentences. Cite the manual section.',PURPLE),('IF EVIDENCE IS MISSING','Say “not found in the provided source.”',GREEN)]
  for i,(a,b,c) in enumerate(data):box(d,(55,215+i*145,845,345+i*145),a,b,c)
 elif s==4:
  box(d,(55,220,845,320),'1 / SOURCE DOCUMENT','Example robot manual + section metadata',GOLD)
  arrow(d,(450,325),(450,365),t)
  for i in range(3):
   x=55+i*268;box(d,(x,370,x+252,505),f'CHUNK {i+1}', ['Safety rules','Battery: 8 h','Motor specs'][i],[PURPLE,GREEN,CYAN][i])
   arrow(d,(x+126,510),(x+126,560),t)
  box(d,(55,565,845,670),'2 / EMBED EACH CHUNK','Passage → embedding vector',CYAN)
  arrow(d,(450,675),(450,725),t)
  for j in range(3):
   y=730+j*29;d.ellipse((180,y,720,y+45),fill='#173447',outline=GREEN,width=2)
  txt(d,(242,765),'3 / VECTOR INDEX',28,GREEN,True)
  txt(d,(120,840),'Vectors + original text + source references',26,MUTED)
 elif s==5:
  box(d,(55,215,845,310),'QUESTION','How long does the robot battery last?',CYAN)
  arrow(d,(450,315),(450,352),t)
  box(d,(55,357,845,450),'RETRIEVE RELEVANT CHUNK','Manual §4: up to 8 h in stated test conditions.',GREEN)
  arrow(d,(450,455),(450,492),t)
  box(d,(55,497,845,590),'AUGMENT THE PROMPT','Instructions + question + retrieved evidence',GOLD)
  arrow(d,(450,595),(450,635),t)
  chip(d,245,640,410,140,t,'GENERATE ANSWER')
  txt(d,(65,812),'“Up to 8 hours under the manual’s',29,WHITE,True)
  txt(d,(65,851),'test conditions. [Manual §4]”',29,GREEN,True)
 else:
  stages=[('PROMPT','Question + instructions',GOLD),('RETRIEVAL','Find document evidence',GREEN),('TOKENS + EMBEDDINGS','Represent augmented input',CYAN),('TRANSFORMER','Attend + transform + predict',PURPLE),('ANSWER + CITATION','Verify against the source',PINK)]
  for i,(a,b,c) in enumerate(stages):
   y=210+i*132;box(d,(100,y,800,y+102),a,b,c)
   if i<4:arrow(d,(450,y+105),(450,y+129),t,c)
 # Large open captions, with one thought on screen at a time.
 d.rounded_rectangle((32,895,868,1135),20,fill='#eaf2fb')
 cap=captions[s][min(2,int(u/(CHAPTER/3)))]
 txt(d,(55,915),'WHAT IS HAPPENING',21,'#275169',True)
 lines(d,(55,953),cap,29,'#10283c',785,10)
 for i in range(7):
  x=50+i*115;d.rounded_rectangle((x,1161,x+102,1167),3,fill='#32475e')
  if i<s:d.rounded_rectangle((x,1161,x+102,1167),3,fill=CYAN)
  if i==s:d.rounded_rectangle((x,1161,x+max(3,int(102*p)),1167),3,fill=CYAN)
 txt(d,(50,1180),f'{s+1}/7   •   Educational schematic   •   Example manual is fictional',14,MUTED)
 return im
if __name__=='__main__':
 if '--stills' in os.sys.argv:
  for s in range(7):frame(s*CHAPTER+CHAPTER/2).save(ROOT/f'qa_{s}.png')
 else:
  p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'ai_tokens_embeddings_rag.mp4')],stdin=subprocess.PIPE)
  for f in range(FPS*DURATION):
   p.stdin.write(frame(f/FPS).tobytes())
   if f%(FPS*CHAPTER)==0:print(f'Chapter {f//(FPS*CHAPTER)+1}/7 rendered',flush=True)
  p.stdin.close();assert p.wait()==0
