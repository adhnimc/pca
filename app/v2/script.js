
const DATASETS={iris:window.IRIS,wine:window.WINE};
const colors=["rgba(31,119,180,.8)","rgba(220,38,38,.8)","rgba(5,150,105,.8)","rgba(217,119,6,.8)"];
const el=id=>document.getElementById(id);
["dataset","n","noise","outliers","color"].forEach(id=>el(id).addEventListener("change",update));
el("rerun").addEventListener("click",update);

function rng(seed){return function(){let t=seed+=0x6D2B79F5;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function randn(r){let u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const std=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/a.length)||1}
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const tr=X=>X[0].map((_,j)=>X.map(r=>r[j]));
const z=a=>{const m=mean(a),s=std(a);return a.map(x=>(x-m)/s)}
function standardize(X){return tr(tr(X).map(c=>z(c)))}
function rankArray(a){const s=a.map((v,i)=>({v,i})).sort((x,y)=>x.v-y.v);const r=[];for(let i=0;i<s.length;){let j=i;while(j+1<s.length&&s[j+1].v===s[i].v)j++;const avg=(i+j+2)/2;for(let k=i;k<=j;k++)r[s[k].i]=avg;i=j+1}return r}
function rankCols(X){return tr(tr(X).map(rankArray))}
function cov(X){const n=X.length,p=X[0].length,M=Array.from({length:p},()=>Array(p).fill(0));for(let i=0;i<p;i++)for(let j=i;j<p;j++){let s=0;for(let r=0;r<n;r++)s+=X[r][i]*X[r][j];M[i][j]=M[j][i]=s/(n-1)}return M}
function matVec(A,v){return A.map(r=>dot(r,v))}
function norm(v){return Math.sqrt(dot(v,v))||1}
function outer(v){return v.map(a=>v.map(b=>a*b))}
function sub(A,B){return A.map((r,i)=>r.map((x,j)=>x-B[i][j]))}
function power(A){let v=Array(A.length).fill(1/Math.sqrt(A.length));for(let t=0;t<500;t++){let nv=matVec(A,v);const n=norm(nv);nv=nv.map(x=>x/n);if(nv.reduce((s,x,i)=>s+Math.abs(x-v[i]),0)<1e-10){v=nv;break}v=nv}return {vec:v,val:dot(v,matVec(A,v))}}
function pca2(X){const Xs=standardize(X),C=cov(Xs),e1=power(C),C2=sub(C,outer(e1.vec).map(r=>r.map(x=>x*e1.val))),e2=power(C2),s1=Xs.map(r=>dot(r,e1.vec)),s2=Xs.map(r=>dot(r,e2.vec)),tv=C.reduce((s,r,i)=>s+r[i],0);return{scores:[s1,s2],comp:[e1.vec,e2.vec],var:[e1.val/tv,e2.val/tv]}}
function corr(a,b){const za=z(a),zb=z(b);return dot(za,zb)/a.length}
function sim(){const n=+el("n").value,noise=+el("noise").value,out=+el("outliers").value,r=rng(42),X=[],mark=Array(n).fill(0);for(let i=0;i<n;i++){const z0=randn(r),e=()=>noise*randn(r);X.push([z0+e(),1.1*z0+e(),Math.exp(.5*z0)+e(),Math.tanh(1.2*z0)+e()])}for(let k=0;k<out&&k<n;k++){X[k][1]+=5+randn(r);mark[k]=1}return{X,cols:["x1","x2","x3","x4"],target:mark,names:["regular","outlier"]}}
function active(){const d=el("dataset").value;if(d==="sim")return sim();return DATASETS[d]}
function align(raw,rank){for(let i=0;i<2;i++)if(corr(raw.scores[i],rank.scores[i])<0){rank.scores[i]=rank.scores[i].map(x=>-x);rank.comp[i]=rank.comp[i].map(x=>-x)}return rank}
function clr(i,meta){if(!el("color").checked)return "rgba(31,119,180,.7)";return colors[(meta.target?.[i]||0)%colors.length]}
function svgClear(id){el(id).innerHTML=""}
function line(s,x1,y1,x2,y2,c,w){s.insertAdjacentHTML("beforeend",`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="${w}"/>`)}
function circ(s,x,y,r,c){s.insertAdjacentHTML("beforeend",`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`)}
function text(s,x,y,t,a="middle",fs=12,c="#111"){s.insertAdjacentHTML("beforeend",`<text x="${x}" y="${y}" text-anchor="${a}" font-size="${fs}" fill="${c}">${t}</text>`)}
function scatter(id,x,y,meta,xlab,ylab){const s=el(id);svgClear(id);const W=520,H=420,m={t:20,r:20,b:48,l:56},zx=z(x),zy=z(y),all=zx.concat(zy),mn=Math.min(...all)-.4,mx=Math.max(...all)+.4,sx=v=>m.l+(v-mn)/(mx-mn)*(W-m.l-m.r),sy=v=>H-m.b-(v-mn)/(mx-mn)*(H-m.t-m.b);line(s,sx(mn),sy(mn),sx(mx),sy(mx),"#2563eb",2);for(let i=0;i<zx.length;i++)circ(s,sx(zx[i]),sy(zy[i]),3.3,clr(i,meta));text(s,W/2,H-12,xlab);text(s,16,H/2,ylab,"middle",12);s.lastChild.setAttribute("transform",`rotate(-90 16 ${H/2})`)}
function biplot(id,p,meta,titleColor){const s=el(id);svgClear(id);const W=520,H=420,m={t:20,r:20,b:48,l:56},a=z(p.scores[0]),b=z(p.scores[1]),lim=Math.max(3,Math.max(...a.map(Math.abs),...b.map(Math.abs))*1.1),sx=v=>m.l+(v+lim)/(2*lim)*(W-m.l-m.r),sy=v=>H-m.b-(v+lim)/(2*lim)*(H-m.t-m.b);line(s,sx(-lim),sy(0),sx(lim),sy(0),"#9ca3af",1);line(s,sx(0),sy(-lim),sx(0),sy(lim),"#9ca3af",1);for(let i=0;i<a.length;i++)circ(s,sx(a[i]),sy(b[i]),3.1,clr(i,meta));for(let j=0;j<meta.cols.length;j++){const x=p.comp[0][j]*lim*.7,y=p.comp[1][j]*lim*.7;line(s,sx(0),sy(0),sx(x),sy(y),titleColor,2);circ(s,sx(x),sy(y),3.5,titleColor);text(s,sx(x)+4,sy(y)-4,meta.cols[j],"start",11,titleColor)}}
function table(meta,raw,rank){el("tbl").querySelector("tbody").innerHTML=meta.cols.map((c,i)=>`<tr><td>${c}</td><td>${raw.comp[0][i].toFixed(3)}</td><td>${raw.comp[1][i].toFixed(3)}</td><td>${rank.comp[0][i].toFixed(3)}</td><td>${rank.comp[1][i].toFixed(3)}</td></tr>`).join("")}
function update(){const meta=active(),raw=pca2(meta.X),rank=align(raw,pca2(rankCols(meta.X)));el("c1").textContent=corr(raw.scores[0],rank.scores[0]).toFixed(3);el("c2").textContent=corr(raw.scores[1],rank.scores[1]).toFixed(3);el("rv").textContent=raw.var.map(v=>v.toFixed(3)).join(", ");el("sv").textContent=rank.var.map(v=>v.toFixed(3)).join(", ");scatter("pc1",raw.scores[0],rank.scores[0],meta,"raw PC1","rank PC1");scatter("pc2",raw.scores[1],rank.scores[1],meta,"raw PC2","rank PC2");biplot("raw",raw,meta,"#2563eb");biplot("rank",rank,meta,"#7c3aed");table(meta,raw,rank)}
update();
