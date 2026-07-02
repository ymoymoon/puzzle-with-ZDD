r"""
ダイヤモンドゲーム（菱形盤面）— ZDD による最短手数解法
========================================================

【盤面】
  菱形 25マス（1-2-3-4-5-4-3-2-1）
  座標(q,r): q in [-2,2], r in [-2-q, 2-q]

  上頂点(-2,4)，下頂点(2,-4)，左頂点(-2,0)，右頂点(2,0)

【描画座標系】
  六角格子のピクセル座標に変換: x=px*(q+r/2), y=-px*r*sqrt(3)/2
  （r が増える方向を画面上方向にするため y は反転）

【初期配置】
  上頂点三角形 (r>=3): 4マス
【ゴール】
  下頂点三角形 (r<=-3): 4マス
"""

import time, json, math
from pathlib import Path
from kyotodd import ZDD
from collections import defaultdict as _dd


# ============================================================
# 盤面定義
# ============================================================

HEX_DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

# 盤面を直接 (q,r) 座標で定義する．
# q in [-2,2], r in [-2-q, 2-q] が「上から1,2,3,4,5,4,3,2,1マス」の菱形になる．
# 頂点は上(-2,4), 下(2,-4), 左(-2,0), 右(2,0)．
BOARD    = frozenset((q,r) for q in range(-2,3) for r in range(-2-q, 2-q+1))
POS_LIST = sorted(BOARD)
POS2ID   = {p: i+1 for i,p in enumerate(POS_LIST)}
ID2POS   = {v: p for p,v in POS2ID.items()}
N_POS    = len(POS_LIST)   # 25

# 上三角形 (r>=3) を初期配置，下三角形 (r<=-3) をゴールとする
INIT_PEGS = frozenset(POS2ID[(q,r)] for (q,r) in BOARD if r >= 3)
GOAL_PEGS = frozenset(POS2ID[(q,r)] for (q,r) in BOARD if r <= -3)
N_PEGS    = len(INIT_PEGS)   # 4

# 合法手
SINGLE_STEPS: list[tuple[int,int]] = []
SINGLE_JUMPS: list[tuple[int,int,int]] = []
for (q,r) in POS_LIST:
    for dq,dr in HEX_DIRS:
        nb = (q+dq, r+dr)
        if nb in POS2ID:
            SINGLE_STEPS.append((POS2ID[(q,r)], POS2ID[nb]))
        ov, to = (q+dq,r+dr), (q+2*dq,r+2*dr)
        if ov in POS2ID and to in POS2ID:
            SINGLE_JUMPS.append((POS2ID[(q,r)], POS2ID[ov], POS2ID[to]))

JUMPS_BY_FR: dict[int,list[tuple[int,int]]] = _dd(list)
for fr,ov,to in SINGLE_JUMPS:
    JUMPS_BY_FR[fr].append((ov,to))

_dummy = ZDD.power_set(N_POS)
del _dummy

print(f"盤面:{N_POS}マス  ペグ:{N_PEGS}個")
print(f"1歩:{len(SINGLE_STEPS)}  ジャンプ:{len(SINGLE_JUMPS)}")


# ============================================================
# ZDD 遷移演算
# ============================================================

def jump_closure(F: ZDD) -> ZDD:
    while True:
        active = set(F.support_vars())
        new = ZDD.empty
        for fr, ov_to_list in JUMPS_BY_FR.items():
            if fr not in active: continue
            F_fr = F.onset(fr)
            for ov,to in ov_to_list:
                new = new + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
        merged = F + new
        if int(merged.exact_count) == int(F.exact_count): break
        F = merged
    return F


def next_all(F: ZDD) -> ZDD:
    active = set(F.support_vars())
    result = ZDD.empty
    for fr,to in SINGLE_STEPS:
        if fr not in active: continue
        result = result + F.onset(fr).offset(to).change(fr).change(to)
    for fr, ov_to_list in JUMPS_BY_FR.items():
        if fr not in active: continue
        F_fr = F.onset(fr)
        af = ZDD.empty
        for ov,to in ov_to_list:
            af = af + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
        if int(af.exact_count) == 0: continue
        result = result + jump_closure(af)
    return result


def prev_all(F: ZDD) -> ZDD:
    """逆向き遷移（ゴールから逆算）"""
    active = set(F.support_vars())
    result = ZDD.empty
    for fr,to in SINGLE_STEPS:
        if to not in active: continue
        result = result + F.onset(to).offset(fr).change(to).change(fr)
    for fr, ov_to_list in JUMPS_BY_FR.items():
        for ov,to in ov_to_list:
            if to not in active: continue
            result = result + F.onset(to).offset(fr).offset(ov).change(to).change(fr).change(ov)
    return result


# ============================================================
# BFS 最短手数 + 手順追跡
# ============================================================

def solve_bfs():
    """BFS で最短手数と各ステップの到達盤面族を返す"""
    t0,c0 = time.perf_counter(), time.process_time()
    F = ZDD.from_sets([sorted(INIT_PEGS)])
    seen = F
    history = [F]   # history[t] = ちょうど t 手後に初めて到達した盤面族
    log = []

    for step in range(1, 100):
        F_next = next_all(F) - seen
        wall = time.perf_counter()-t0
        cpu  = time.process_time()-c0
        cnt  = int(F_next.exact_count)
        log.append({"step":step,"wall":wall,"cpu":cpu,
                    "count":cnt,"nodes":F_next.raw_size})
        print(f"  手{step:2d}  実{wall:7.3f}s  CPU{cpu:7.3f}s  "
              f"新規:{cnt:8,}  ノード:{F_next.raw_size:6,}")
        if cnt == 0: return -1, log, history
        seen = seen + F_next
        history.append(F_next)

        # ゴール到達確認
        chk = F_next
        for pid in sorted(GOAL_PEGS): chk = chk.onset(pid)
        for pid in POS2ID.values():
            if pid not in GOAL_PEGS: chk = chk.offset(pid)
        if int(chk.exact_count) > 0:
            print(f"\n  ★ 最短手数: {step} 手")
            return step, log, history
        F = F_next
    return -1, log, history


def trace_solution(history: list[ZDD], min_moves: int) -> list[dict]:
    """
    history を使って前向きに手順を1つ復元する．
    history[t] = ちょうど t 手後の新規到達盤面族
    """
    if min_moves <= 0:
        return []

    # ゴール盤面から出発して逆向きに辿る
    # ゴール盤面を history[min_moves] から取り出す
    chk = history[min_moves]
    for pid in sorted(GOAL_PEGS): chk = chk.onset(pid)
    for pid in POS2ID.values():
        if pid not in GOAL_PEGS: chk = chk.offset(pid)

    board = frozenset(history[min_moves].unrank(0))

    # ゴール盤面を特定
    goal_board = frozenset(GOAL_PEGS)

    # 前向きに手順を決定
    # t 手目の盤面から t+1 手目の手を選ぶ
    # history[t+1] に含まれる盤面が次の正解盤面

    # まず初期盤面を設定
    board = frozenset(INIT_PEGS)
    steps = []

    for t in range(min_moves):
        # history[t+1] = ちょうど t+1 手後の新規到達盤面族
        next_hist = history[t+1]
        found = False

        # 全合法手を試す
        all_moves = ([(fr, None, to, 'step') for fr,to in SINGLE_STEPS]
                    + [(fr, ov, to, 'jump') for fr,ov,to in SINGLE_JUMPS])

        for move in all_moves:
            if move[3] == 'step':
                fr, _, to, _ = move
                if fr not in board or to in board: continue
                next_board = (board - {fr}) | {to}
            else:
                fr, ov, to, _ = move
                if fr not in board or ov not in board or to in board: continue
                next_board = (board - {fr, ov}) | {to}

            # next_board が history[t+1] に含まれるか確認
            # ZDD の onset/offset で確認
            chk = next_hist
            for pid in sorted(next_board):
                chk = chk.onset(pid)
            for pid in POS2ID.values():
                if pid not in next_board:
                    chk = chk.offset(pid)

            if int(chk.exact_count) > 0:
                steps.append({
                    "step":   t+1,
                    "fr_user":  ID2POS[fr],
                    "ov_user":  ID2POS[ov] if move[3]=='jump' else None,
                    "to_user":  ID2POS[to],
                    "fr_pid":   fr,
                    "ov_pid":   ov if move[3]=='jump' else None,
                    "to_pid":   to,
                    "board":    frozenset(next_board),
                    "type":     move[3],
                })
                board = frozenset(next_board)
                found = True
                break

        if not found:
            print(f"  手{t+1}で手順が見つかりません")
            break

    return steps


# ============================================================
# HTML 生成
# ============================================================

def _user_pixel(q, r, px=32):
    """(q,r)座標を六角格子のピクセル座標に変換する．
    r が増える方向を画面上方向にするため y は反転する．
    六角格子なので x は q + r/2 のオフセットを持つ．
    """
    x = px * (q + r * 0.5)
    y = -px * r * (3**0.5) / 2   # r増加=上方向なのでy反転
    return x, y


def build_anim_json(move_steps: list[dict], px: int = 36) -> str:
    pts = [(_user_pixel(*pos, px=px), POS2ID[pos]) for pos in POS_LIST]
    xs=[x for (x,y),_ in pts]; ys=[y for (x,y),_ in pts]
    ox=min(xs)-px; oy=min(ys)-px
    W=max(xs)-ox+px+2; H=max(ys)-oy+px+2

    cells = [[pid, round(x-ox+1,1), round(y-oy+1,1)] for (x,y),pid in pts]

    frames = [{"pegs": sorted(INIT_PEGS), "fr": None, "ov": None, "to": None,
               "label": "初期盤面", "desc": ""}]
    for s in move_steps:
        desc = str(s['fr_user'])+" → "
        if s['ov_user']: desc += str(s['ov_user'])+" → "
        desc += str(s['to_user'])
        frames.append({
            "pegs": sorted(s["board"]),
            "fr":   s["fr_pid"],
            "ov":   s["ov_pid"],
            "to":   s["to_pid"],
            "label": "手 "+str(s["step"])+" ("+s["type"]+")",
            "desc":  desc,
        })

    return json.dumps({
        "cells": cells,
        "r_hole": round(px*0.38,1),
        "r_peg":  round(px*0.26,1),
        "board_w": round(W),
        "board_h": round(H),
        "frames": frames,
    }, ensure_ascii=False)


def board_svg_static(pegs: frozenset, px: int = 28) -> str:
    pts = [(_user_pixel(*pos,px=px), POS2ID[pos]) for pos in POS_LIST]
    xs=[x for (x,y),_ in pts]; ys=[y for (x,y),_ in pts]
    ox=min(xs)-px; oy=min(ys)-px
    W=max(xs)-ox+px+2; H=max(ys)-oy+px+2
    rh=round(px*0.38,1); rp=round(px*0.26,1)
    c=[]
    for (x,y),pid in pts:
        cx=round(x-ox+1,1); cy=round(y-oy+1,1)
        c.append('<circle cx="'+str(cx)+'" cy="'+str(cy)+'" r="'+str(rh)+
                 '" fill="#c8a97e" stroke="#9a7a55" stroke-width="0.8"/>')
        if pid in pegs:
            fill="#4a3728"
            if pid in INIT_PEGS and pid in GOAL_PEGS: fill="#8060a0"
            c.append('<circle cx="'+str(cx)+'" cy="'+str(cy)+'" r="'+str(rp)+
                     '" fill="'+fill+'" stroke="#2a1a10" stroke-width="0.8"/>')
    W2,H2=round(W),round(H)
    return ('<svg width="'+str(W2)+'" height="'+str(H2)+
            '" viewBox="0 0 '+str(W2)+' '+str(H2)+
            '" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="'+str(W2)+'" height="'+str(H2)+
            '" fill="#e8d5b0" rx="4"/>'+"".join(c)+"</svg>")


def generate_html(min_moves, log, wall_total, cpu_total,
                  move_steps: list[dict],
                  out="diamond_game_result.html"):

    init_svg  = board_svg_static(INIT_PEGS)
    goal_svg  = board_svg_static(GOAL_PEGS)
    anim_json = build_anim_json(move_steps)

    tbl_rows = "".join(
        '<tr><td>'+str(e['step'])+'</td>'
        '<td>'+f"{e['wall']:.3f}"+'</td><td>'+f"{e['cpu']:.3f}"+'</td>'
        '<td>'+f"{e['count']:,}"+'</td><td>'+f"{e['nodes']:,}"+'</td></tr>'
        for e in log)
    tbl = ('<div class="table-wrap"><table>'
           '<caption>BFS探索ログ · 実'+f"{wall_total:.3f}"+'s · CPU'
           +f"{cpu_total:.3f}"+'s</caption>'
           '<thead><tr><th>手</th><th>実時間(s)</th><th>CPU(s)</th>'
           '<th>新規盤面数</th><th>ZDDノード数</th></tr></thead>'
           '<tbody>'+tbl_rows+'</tbody></table></div>')

    # 手順テーブル
    step_rows = ""
    for s in move_steps:
        step_rows += ('<tr><td>'+str(s['step'])+'</td>'
                      '<td>'+str(s['fr_user'])+'</td>'
                      '<td>'+(str(s['ov_user']) if s['ov_user'] else '-')+'</td>'
                      '<td>'+str(s['to_user'])+'</td>'
                      '<td>'+s['type']+'</td></tr>')
    step_tbl = ('<div class="table-wrap"><table>'
                '<caption>解の手順（ユーザー座標 (a,b)）</caption>'
                '<thead><tr><th>手</th><th>移動元</th><th>飛越え</th>'
                '<th>移動先</th><th>種別</th></tr></thead>'
                '<tbody>'+step_rows+'</tbody></table></div>'
                if step_rows else '<p class="note">手順未取得</p>')

    res_val = str(min_moves) if min_moves > 0 else "?"

    CSS = """
<style>
:root{--bg:#0f1a2e;--surface:#162236;--card:#1c2d45;--border:#2a4a6b;
      --accent:#4a9eff;--accent2:#7bc4ff;--text:#d0e8ff;--muted:#6a8aaa;
      --radius:8px;--mono:"JetBrains Mono","Fira Code",monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
     font-family:"Noto Sans JP","Hiragino Sans",sans-serif;
     line-height:1.6;padding-bottom:60px}
header{background:var(--surface);border-bottom:2px solid var(--accent);padding:22px 36px 16px}
header h1{font-size:1.45rem;font-weight:700;color:var(--accent2);margin-bottom:3px}
header .sub{color:var(--muted);font-size:.83rem}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
         gap:11px;padding:22px 36px 0}
.kpi{background:var(--card);border:1px solid var(--border);
     border-radius:var(--radius);padding:12px 14px}
.kpi .label{font-size:.67rem;color:var(--muted);text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:3px}
.kpi .value{font-size:1.5rem;font-weight:700;color:var(--accent2);
            font-family:var(--mono)}
.kpi .unit{font-size:.70rem;color:var(--muted);margin-left:2px}
section{padding:26px 36px 0}
section h2{font-size:.93rem;font-weight:600;color:var(--accent);
           border-left:3px solid var(--accent);padding-left:9px;margin-bottom:12px}
.boards{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.board-card{background:var(--card);border:1px solid var(--border);
            border-radius:var(--radius);padding:12px;text-align:center}
.board-label{font-size:.70rem;color:var(--accent2);font-weight:600;
             margin-bottom:7px;font-family:var(--mono)}
.result-box{background:var(--card);border:2px solid var(--accent);
            border-radius:var(--radius);padding:14px 20px;font-family:var(--mono)}
.result-box .big{font-size:1.9rem;font-weight:700;color:var(--accent2);margin-right:4px}
.table-wrap{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border);
            margin-bottom:4px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.80rem}
caption{text-align:left;padding:6px 12px 3px;font-size:.71rem;
        color:var(--muted);background:var(--surface)}
thead tr{background:var(--surface)}
th{padding:6px 12px;text-align:right;color:var(--accent2);font-weight:600;
   font-size:.71rem;border-bottom:1px solid var(--border)}
th:first-child{text-align:center}
td{padding:5px 12px;text-align:right;border-bottom:1px solid #1c2d45}
td:first-child{text-align:center;color:var(--muted)}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(74,158,255,.07)}
.player{background:var(--card);border:1px solid var(--border);
        border-radius:var(--radius);padding:16px;
        display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
#anim-canvas{border-radius:5px;display:block}
.ctrl{flex:1;min-width:190px;display:flex;flex-direction:column;gap:9px}
.flabel{font-size:.92rem;font-weight:700;color:var(--accent2);font-family:var(--mono)}
.fdesc{font-size:.73rem;color:var(--muted);font-family:var(--mono)}
.fcnt{font-size:.68rem;color:var(--muted);margin-top:2px}
.btns{display:flex;gap:5px;flex-wrap:wrap}
.btn{background:var(--surface);border:1px solid var(--border);border-radius:5px;
     color:var(--accent2);font-family:var(--mono);font-size:.76rem;
     padding:5px 9px;cursor:pointer;transition:background .12s}
.btn:hover{background:var(--card);border-color:var(--accent)}
.btn.primary{background:var(--accent);color:#0f1a2e;
             border-color:var(--accent);font-weight:700}
.btn.primary:hover{background:var(--accent2)}
.btn:disabled{opacity:.3;cursor:not-allowed}
.spd{display:flex;align-items:center;gap:7px;font-size:.70rem;color:var(--muted)}
input[type=range]{flex:1;accent-color:var(--accent)}
.pbar{height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.pfill{height:100%;background:var(--accent);transition:width .15s;width:0%}
.legend{display:flex;gap:11px;flex-wrap:wrap}
.li{display:flex;align-items:center;gap:4px;font-size:.69rem;color:var(--muted)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.note{color:var(--muted);font-size:.78rem}
.coord-note{background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius);padding:10px 14px;
            font-family:var(--mono);font-size:.76rem;color:var(--muted);
            line-height:1.9;margin-bottom:4px}
</style>"""

    JS = """
<script>
(function(){
  var D=(""" + anim_json + """);
  var cv=document.getElementById('anim-canvas');
  var ctx=cv.getContext('2d');
  var R=window.devicePixelRatio||1;
  var W=D.board_w, H=D.board_h;
  cv.width=W*R; cv.height=H*R;
  cv.style.width=W+'px'; cv.style.height=H+'px';
  ctx.scale(R,R);
  var cm={};
  D.cells.forEach(function(c){ cm[c[0]]=[c[1],c[2]]; });
  function draw(fi){
    var f=D.frames[fi], ps=new Set(f.pegs);
    ctx.fillStyle='#e8d5b0';
    ctx.beginPath(); ctx.roundRect(0,0,W,H,4); ctx.fill();
    D.cells.forEach(function(c){
      ctx.beginPath(); ctx.arc(c[1],c[2],D.r_hole,0,Math.PI*2);
      ctx.fillStyle='#c8a97e'; ctx.fill();
      ctx.strokeStyle='#9a7a55'; ctx.lineWidth=0.8; ctx.stroke();
    });
    D.cells.forEach(function(c){
      if(!ps.has(c[0])) return;
      var cl='#4a3728';
      if(c[0]===f.fr) cl='#e05c5c';
      else if(c[0]===f.ov) cl='#e08c3c';
      else if(c[0]===f.to) cl='#5cb85c';
      ctx.beginPath(); ctx.arc(c[1],c[2],D.r_peg,0,Math.PI*2);
      ctx.fillStyle=cl; ctx.fill();
      ctx.strokeStyle='#2a1a10'; ctx.lineWidth=1.2; ctx.stroke();
    });
    if(f.fr!==null && f.ov!==null){
      var p1=cm[f.fr], pm=cm[f.ov], p2=cm[f.to];
      ctx.save(); ctx.setLineDash([3,2]); ctx.lineWidth=1.5;
      ctx.strokeStyle='rgba(123,196,255,.7)';
      ctx.beginPath(); ctx.moveTo(p1[0],p1[1]);
      ctx.lineTo(pm[0],pm[1]); ctx.lineTo(p2[0],p2[1]); ctx.stroke();
      var a=Math.atan2(p2[1]-pm[1],p2[0]-pm[0]), l=8;
      ctx.setLineDash([]); ctx.lineWidth=1.8; ctx.strokeStyle='#7bc4ff';
      ctx.beginPath();
      ctx.moveTo(p2[0],p2[1]);
      ctx.lineTo(p2[0]-l*Math.cos(a-.4),p2[1]-l*Math.sin(a-.4));
      ctx.moveTo(p2[0],p2[1]);
      ctx.lineTo(p2[0]-l*Math.cos(a+.4),p2[1]-l*Math.sin(a+.4));
      ctx.stroke(); ctx.restore();
    } else if(f.fr!==null){
      var p1=cm[f.fr], p2=cm[f.to];
      ctx.save(); ctx.setLineDash([3,2]); ctx.lineWidth=1.5;
      ctx.strokeStyle='rgba(123,196,255,.7)';
      ctx.beginPath(); ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]); ctx.stroke();
      var a=Math.atan2(p2[1]-p1[1],p2[0]-p1[0]), l=8;
      ctx.setLineDash([]); ctx.lineWidth=1.8; ctx.strokeStyle='#7bc4ff';
      ctx.beginPath();
      ctx.moveTo(p2[0],p2[1]);
      ctx.lineTo(p2[0]-l*Math.cos(a-.4),p2[1]-l*Math.sin(a-.4));
      ctx.moveTo(p2[0],p2[1]);
      ctx.lineTo(p2[0]-l*Math.cos(a+.4),p2[1]-l*Math.sin(a+.4));
      ctx.stroke(); ctx.restore();
    }
  }
  var cur=0, tot=D.frames.length-1, tm=null, iv=1000;
  function g(id){ return document.getElementById(id); }
  function ui(fi){
    cur=fi; var f=D.frames[fi];
    g('flabel').textContent=f.label;
    g('fdesc').textContent=f.desc;
    g('fcnt').textContent=fi+' / '+tot;
    g('pfill').style.width=(tot>0?fi/tot*100:0)+'%';
    g('b-prev').disabled=fi<=0; g('b-first').disabled=fi<=0;
    g('b-next').disabled=fi>=tot; g('b-last').disabled=fi>=tot;
    draw(fi);
  }
  function stop(){ if(tm){clearInterval(tm);tm=null;} g('b-play').textContent='▶ 再生'; }
  function play(){
    stop();
    tm=setInterval(function(){ if(cur>=tot){stop();return;} ui(cur+1); },iv);
    g('b-play').textContent='⏸ 停止';
  }
  g('b-play').addEventListener('click',function(){
    if(tm)stop(); else{ if(cur>=tot)ui(0); play(); }
  });
  g('b-first').addEventListener('click',function(){ stop(); ui(0); });
  g('b-last').addEventListener('click',function(){ stop(); ui(tot); });
  g('b-prev').addEventListener('click',function(){ stop(); ui(Math.max(0,cur-1)); });
  g('b-next').addEventListener('click',function(){ stop(); ui(Math.min(tot,cur+1)); });
  var sp=g('speed');
  sp.addEventListener('input',function(){
    iv=parseInt(sp.value);
    g('sv').textContent=(iv/1000).toFixed(1)+'s/手';
    if(tm){ stop(); play(); }
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'){ stop(); ui(Math.min(tot,cur+1)); }
    if(e.key==='ArrowLeft') { stop(); ui(Math.max(0,cur-1)); }
    if(e.key===' '){ e.preventDefault(); g('b-play').click(); }
  });
  ui(0);
})();
</script>"""

    HTML = ('<!DOCTYPE html>\n<html lang="ja">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            '<title>ダイヤモンドゲーム ZDD解析</title>\n'
            + CSS + '\n</head>\n<body>\n'
            '<header>\n'
            '<h1>ダイヤモンドゲーム（菱形盤面）— ZDD 最短手数解析</h1>\n'
            '<div class="sub">菱形 '+str(N_POS)+'マス（1-2-3-4-5-4-3-2-1）'
            ' &nbsp;|&nbsp; ペグ '+str(N_PEGS)+'個'
            ' &nbsp;|&nbsp; 1歩:'+str(len(SINGLE_STEPS))
            +' ジャンプ:'+str(len(SINGLE_JUMPS))+'</div>\n</header>\n'
            '<div class="summary">\n'
            '<div class="kpi"><div class="label">最短手数</div>'
            '<div class="value">'+res_val+'<span class="unit">手</span></div></div>\n'
            '<div class="kpi"><div class="label">実時間</div>'
            '<div class="value">'+f"{wall_total:.3f}"+'<span class="unit">s</span></div></div>\n'
            '<div class="kpi"><div class="label">CPU時間</div>'
            '<div class="value">'+f"{cpu_total:.3f}"+'<span class="unit">s</span></div></div>\n'
            '<div class="kpi"><div class="label">盤面マス数</div>'
            '<div class="value">'+str(N_POS)+'</div></div>\n'
            '<div class="kpi"><div class="label">ペグ数</div>'
            '<div class="value">'+str(N_PEGS)+'</div></div>\n'
            '</div>\n'
            '<section>\n<h2>座標系</h2>\n'
            '<div class="coord-note">'
            '座標 (q,r): q∈[-2,2],  r∈[-2-q, 2-q] &nbsp;|&nbsp;'
            ' 上頂点:(-2,4)  下頂点:(2,-4)  左頂点:(-2,0)  右頂点:(2,0) &nbsp;|&nbsp;'
            ' 上三角形: r&ge;3 &nbsp; 下三角形: r&le;-3'
            '</div>\n</section>\n'
            '<section>\n<h2>初期盤面 / ゴール盤面</h2>\n'
            '<div class="boards">\n'
            '<div class="board-card"><div class="board-label">初期盤面（上三角形 r≥3）</div>'
            +init_svg+'</div>\n'
            '<div class="board-card"><div class="board-label">ゴール盤面（下三角形 r≤-3）</div>'
            +goal_svg+'</div>\n'
            '</div>\n</section>\n'
            '<section>\n<h2>最短手数の結果</h2>\n'
            '<div class="result-box"><span class="big">'+res_val+'</span>手'
            '（実: '+f"{wall_total:.3f}"+'s，CPU: '+f"{cpu_total:.3f}"+'s）'
            '</div>\n</section>\n'
            '<section>\n<h2>解の手順（アニメーション）</h2>\n'
            '<div class="player">\n'
            '<canvas id="anim-canvas"></canvas>\n'
            '<div class="ctrl">\n'
            '<div>\n'
            '<div class="flabel" id="flabel">初期盤面</div>\n'
            '<div class="fdesc"  id="fdesc"></div>\n'
            '<div class="fcnt"   id="fcnt">0 / '+str(len(move_steps))+'</div>\n'
            '</div>\n'
            '<div class="pbar"><div class="pfill" id="pfill"></div></div>\n'
            '<div class="btns">\n'
            '<button class="btn" id="b-first">⏮</button>\n'
            '<button class="btn" id="b-prev">◀</button>\n'
            '<button class="btn primary" id="b-play">▶ 再生</button>\n'
            '<button class="btn" id="b-next">▶</button>\n'
            '<button class="btn" id="b-last">⏭</button>\n'
            '</div>\n'
            '<div class="spd"><span>速度</span>'
            '<input type="range" id="speed" min="200" max="3000" value="1000" step="100">'
            '<span id="sv">1.0s/手</span></div>\n'
            '<div class="legend">\n'
            '<div class="li"><span class="dot" style="background:#4a3728;border:1px solid #2a1a10"></span>ペグ</div>\n'
            '<div class="li"><span class="dot" style="background:#e05c5c"></span>移動元</div>\n'
            '<div class="li"><span class="dot" style="background:#e08c3c"></span>飛び越え</div>\n'
            '<div class="li"><span class="dot" style="background:#5cb85c"></span>移動先</div>\n'
            '</div>\n</div>\n</div>\n</section>\n'
            '<section>\n<h2>手順詳細</h2>\n'+step_tbl+'\n</section>\n'
            '<section>\n<h2>BFS 探索ログ</h2>\n'+tbl+'\n</section>\n'
            + JS + '\n</body>\n</html>')

    Path(out).write_text(HTML, encoding="utf-8")
    print(f"HTML 出力: {out}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("="*55)
    print("  ダイヤモンドゲーム（菱形25マス）— ZDD BFS")
    print("  上頂点(-2,4) → 下頂点(2,-4)")
    print("="*55)

    t0, c0 = time.perf_counter(), time.process_time()
    min_moves, log, history = solve_bfs()
    wall_total = time.perf_counter()-t0
    cpu_total  = time.process_time()-c0

    print(f"\n  実時間: {wall_total:.3f}s  CPU時間: {cpu_total:.3f}s")

    move_steps = []
    if min_moves > 0:
        print("手順を追跡中...")
        move_steps = trace_solution(history, min_moves)
        print(f"  {len(move_steps)} 手の手順を取得")

    generate_html(min_moves, log, wall_total, cpu_total, move_steps)