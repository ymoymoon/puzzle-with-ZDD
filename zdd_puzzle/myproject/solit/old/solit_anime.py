r"""
ペグソリティア — reachable ZDD 高速解法
任意の盤面定義に対応した汎用実装
"""

import time
import html as _html
from pathlib import Path
from kyotodd import ZDD
from collections import defaultdict as _dd


# ============================================================
# 盤面定義（ここを変更するだけで任意の盤面に対応）
# ============================================================

def make_small_board():
    valid = set()
    for r in range(1, 6):
        for c in range(1, 6):
            valid.add((r, c))
    init = (2, 3)
    goal = (4, 3)
    grid_size = 5
    return valid, init, goal, grid_size

def make_english_board():
    """英国式 33マス（十字型 7×7）"""
    valid = set()
    for r in range(1, 8):
        for c in range(1, 8):
            if (r < 3 or r > 5) and (c < 3 or c > 5):
                continue
            valid.add((r, c))
    center = (4, 4)
    grid_size = 7
    return valid, center, center, grid_size


def make_french_board():
    """フランス式 37マス（角を含む十字型）"""
    valid = set()
    for r in range(1, 8):
        for c in range(1, 8):
            if (abs(r-4) + abs(c-4) <= 4):
                valid.add((r, c))
    center = (4, 4)
    grid_size = 7
    return valid, center, center, grid_size


def make_diamond_board():
    """ダイヤモンド型 41マス"""
    valid = set()
    for r in range(1, 10):
        for c in range(1, 10):
            if abs(r - 5) + abs(c - 5) <= 4:
                valid.add((r, c))
    center = (5, 5)
    grid_size = 9
    return valid, center, center, grid_size


# ── 使用する盤面を選択 ──
VALID, INIT_HOLE, GOAL_POS, GRID_SIZE = make_small_board()

# ============================================================
# 盤面から自動的に計算される定数
# ============================================================

POS_LIST = sorted(VALID)
POS2ID   = {p: i+1 for i, p in enumerate(POS_LIST)}
ID2POS   = {v: p for p, v in POS2ID.items()}

# 合法手
MOVES: list[tuple[int,int,int]] = []
for (r, c) in POS_LIST:
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        over = (r+dr, c+dc)
        to   = (r+2*dr, c+2*dc)
        if over in VALID and to in VALID:
            MOVES.append((POS2ID[(r,c)], POS2ID[over], POS2ID[to]))

MOVES_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_FR[_fr].append((_ov, _to))

MOVES_BY_TO: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_TO[_to].append((_fr, _ov))

N_POS   = len(POS_LIST)          # マス数
N_PEGS  = N_POS - 1              # 初期ペグ数（空きマス1マス）
N_STEPS = N_PEGS - 1             # 手数（N_PEGS個 → 1個）

# ── 初期の空きマスとゴールのペグ位置を独立して指定 ──────────

INIT_EMPTY = POS2ID[INIT_HOLE]
GOAL_PEG   = POS2ID[GOAL_POS]

INIT_STATE = frozenset(POS2ID.values()) - {INIT_EMPTY}
GOAL_STATE = frozenset({GOAL_PEG})

# 変数空間の初期化（offset/change が var_count() 超で失敗するのを防ぐ）
_dummy = ZDD.power_set(N_POS)
del _dummy

print(f"盤面: {N_POS}マス  初期ペグ: {N_PEGS}個  手数: {N_STEPS}手")
print(f"合法手数: {len(MOVES)}  初期空き: {INIT_HOLE}  ゴール: {GOAL_POS}")


# ============================================================
# 遷移演算
# ============================================================

def next_all(F: ZDD) -> ZDD:
    active = set(F.support_vars())
    result = ZDD.empty
    for fr, ov_to_list in MOVES_BY_FR.items():
        if fr not in active:
            continue
        F_fr = F.onset(fr)
        for ov, to in ov_to_list:
            result = result + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
    return result


def prev_all(F: ZDD) -> ZDD:
    active = set(F.support_vars())
    result = ZDD.empty
    for to, fr_ov_list in MOVES_BY_TO.items():
        if to not in active:
            continue
        F_to = F.onset(to)
        for fr, ov in fr_ov_list:
            result = result + F_to.offset(fr).offset(ov).change(to).change(fr).change(ov)
    return result


# ============================================================
# Phase 1: 逆向き探索
# ============================================================

def build_reachable():
    t0 = time.perf_counter()
    F  = ZDD.from_sets([sorted(GOAL_STATE)])
    reachable: list[ZDD] = [F]
    log = [{"step": 0, "elapsed": 0.0, "count": int(F.exact_count), "nodes": F.raw_size}]

    for step in range(N_STEPS):
        F = prev_all(F)
        elapsed = time.perf_counter() - t0
        log.append({"step": step+1, "elapsed": elapsed,
                    "count": int(F.exact_count), "nodes": F.raw_size})
        reachable.append(F)

    return reachable, log, time.perf_counter() - t0


# ============================================================
# Phase 2: 前向き高速探索
# ============================================================

def solve_with_reachable(reachable: list[ZDD]):
    t0 = time.perf_counter()
    F  = ZDD.from_sets([sorted(INIT_STATE)])
    F  = F & reachable[N_STEPS]
    log = [{"step": 0, "elapsed": 0.0, "count": int(F.exact_count), "nodes": F.raw_size}]

    for step in range(N_STEPS):
        F = next_all(F)
        F = F & reachable[N_STEPS - (step+1)]
        elapsed = time.perf_counter() - t0
        log.append({"step": step+1, "elapsed": elapsed,
                    "count": int(F.exact_count), "nodes": F.raw_size})
        if int(F.exact_count) == 0:
            break

    return F, log, time.perf_counter() - t0


# ============================================================
# Phase 3: 手順復元
# ============================================================

def trace_with_reachable(reachable: list[ZDD]) -> list[dict]:
    board = INIT_STATE
    steps = []

    for t in range(N_STEPS):
        future_filter = reachable[N_STEPS - (t+1)]
        found         = False

        for fr, ov, to in MOVES:
            if fr not in board or ov not in board or to in board:
                continue
            next_board = (board - {fr, ov}) | {to}
            candidate  = future_filter
            for pid in sorted(next_board):
                candidate = candidate.onset(pid)
            for pid in POS2ID.values():
                if pid not in next_board:
                    candidate = candidate.offset(pid)
            if int(candidate.exact_count) > 0:
                board = frozenset(next_board)
                steps.append({
                    "step":   t+1,
                    "fr_pos": ID2POS[fr],
                    "ov_pos": ID2POS[ov],
                    "to_pos": ID2POS[to],
                    "board":  frozenset(board),
                })
                found = True
                break

        if not found:
            break

    return steps


# ============================================================
# 盤面データの JSON シリアライズ
# ============================================================

def _cell_coords(px_per_cell: int = 22):
    """各マスの (cx, cy) 座標を返す dict {pid: (cx,cy)}"""
    r_off = px_per_cell // 2
    return {
        POS2ID[(r,c)]: ((c-1)*px_per_cell + r_off, (r-1)*px_per_cell + r_off)
        for (r,c) in POS_LIST
    }

def build_animation_data(move_steps: list[dict], px: int = 26) -> str:
    """
    アニメーション用の JSON データを生成する．

    形式:
      {
        "cells":  [[pid, cx, cy], ...],      # 全マスの座標
        "r_hole": r_hole,                    # 穴の半径
        "r_peg":  r_peg,                     # ペグの半径
        "board_w": board_px,
        "board_h": board_px,
        "frames": [
          { "pegs": [pid,...],               # このフレームでのペグ位置
            "fr": pid|null, "ov": pid|null, "to": pid|null,  # 移動ハイライト
            "label": "手 N",
            "desc":  "(r,c) → (r,c) → (r,c)" },
          ...
        ]
      }
    """
    import json
    coords = _cell_coords(px)
    r_off  = px // 2
    board_px = GRID_SIZE * px

    cells = [[pid, cx, cy] for pid,(cx,cy) in coords.items()]

    frames = []
    # フレーム0: 初期盤面
    frames.append({
        "pegs":  sorted(INIT_STATE),
        "fr": None, "ov": None, "to": None,
        "label": "初期盤面", "desc": ""
    })
    for s in move_steps:
        fr_id = POS2ID[s["fr_pos"]]
        ov_id = POS2ID[s["ov_pos"]]
        to_id = POS2ID[s["to_pos"]]
        frames.append({
            "pegs":  sorted(s["board"]),
            "fr":    fr_id,
            "ov":    ov_id,
            "to":    to_id,
            "label": f"手 {s['step']}",
            "desc":  f"{s['fr_pos']} → {s['ov_pos']} → {s['to_pos']}"
        })

    return json.dumps({
        "cells":   cells,
        "r_hole":  r_off - 2,
        "r_peg":   r_off - 5,
        "board_w": board_px,
        "board_h": board_px,
        "frames":  frames,
    }, ensure_ascii=False)


def small_board_svg(state: frozenset[int], px_per_cell: int = 14) -> str:
    """到達可能初期盤面用の小さい静的SVG"""
    cell     = px_per_cell
    r_off    = cell // 2
    board_px = GRID_SIZE * cell
    circles  = []
    for (r, c) in POS_LIST:
        cx  = (c-1)*cell + r_off
        cy  = (r-1)*cell + r_off
        pid = POS2ID[(r,c)]
        circles.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r_off-1}" '
            f'fill="#c8a97e" stroke="#9a7a55" stroke-width="0.8"/>'
        )
        if pid in state:
            circles.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_off-3}" '
                f'fill="#4a3728" stroke="#2a1a10" stroke-width="1"/>'
            )
    return (
        f'<svg width="{board_px}" height="{board_px}" '
        f'viewBox="0 0 {board_px} {board_px}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<rect width="{board_px}" height="{board_px}" fill="#e8d5b0" rx="3"/>'
        + "".join(circles) + "</svg>"
    )


# ============================================================
# HTML 生成
# ============================================================

def log_table(log: list[dict], caption: str) -> str:
    rows = "".join(
        f"<tr><td>{e['step']}</td><td>{e['elapsed']:.2f}</td>"
        f"<td>{e['count']:,}</td><td>{e['nodes']:,}</td></tr>"
        for e in log
    )
    return f"""
<div class="table-wrap">
  <table>
    <caption>{_html.escape(caption)}</caption>
    <thead><tr>
      <th>ステップ</th><th>経過時間 (s)</th><th>盤面数</th><th>ZDDノード数</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def generate_html(bwd_log, bwd_time, 
                  reachable_inits, num_reachable, move_steps,
                  out_path="anime2.html"):
    from pathlib import Path

    anim_json = build_animation_data(move_steps, px=26)
    board_name = f"{N_POS}マス  初期空き:{INIT_HOLE}  ゴール:{GOAL_POS}"

    init_html = "".join(
        f'<div class="init-card"><div class="init-label">#{i+1}</div>'
        f'{small_board_svg(st)}</div>'
        for i, st in enumerate(reachable_inits)
    )
    if num_reachable > len(reachable_inits):
        init_html += f'<p class="more">他 {num_reachable - len(reachable_inits):,} 件</p>'

    page = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ペグソリティア ZDD解析 — {board_name}</title>
<style>
  :root {{
    --bg:#1a1208;--surface:#2c1f0e;--card:#3a2a14;--border:#6b4c1e;
    --accent:#c8860a;--accent2:#e8b84b;--text:#f0e2c8;--muted:#a08060;
    --radius:8px;--mono:"JetBrains Mono","Fira Code",monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);
        font-family:"Noto Sans JP","Hiragino Sans",sans-serif;
        line-height:1.6;padding-bottom:60px}}
  header{{background:var(--surface);border-bottom:2px solid var(--accent);
          padding:28px 40px 20px}}
  header h1{{font-size:1.8rem;font-weight:700;color:var(--accent2);
             letter-spacing:.04em;margin-bottom:4px}}
  header .sub{{color:var(--muted);font-size:.88rem}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:14px;padding:28px 40px 0}}
  .kpi{{background:var(--card);border:1px solid var(--border);
        border-radius:var(--radius);padding:16px 18px}}
  .kpi .label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:5px}}
  .kpi .value{{font-size:1.6rem;font-weight:700;color:var(--accent2);
               font-family:var(--mono)}}
  .kpi .unit{{font-size:.78rem;color:var(--muted);margin-left:3px}}
  section{{padding:36px 40px 0}}
  section h2{{font-size:1.05rem;font-weight:600;color:var(--accent);
              border-left:3px solid var(--accent);padding-left:10px;
              margin-bottom:16px;letter-spacing:.04em}}
  .table-wrap{{overflow-x:auto;border-radius:var(--radius);
               border:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;
         font-family:var(--mono);font-size:.83rem}}
  caption{{text-align:left;padding:9px 14px 5px;font-size:.78rem;
           color:var(--muted);background:var(--surface)}}
  thead tr{{background:var(--surface)}}
  th{{padding:9px 14px;text-align:right;color:var(--accent2);
      font-weight:600;font-size:.76rem;letter-spacing:.06em;
      border-bottom:1px solid var(--border)}}
  th:first-child{{text-align:center}}
  td{{padding:6px 14px;text-align:right;
      border-bottom:1px solid #3a2a14;color:var(--text)}}
  td:first-child{{text-align:center;color:var(--muted)}}
  tr:last-child td{{border-bottom:none}}
  tbody tr:hover{{background:rgba(200,134,10,.08)}}

  /* アニメーションプレーヤー */
  .player{{background:var(--card);border:1px solid var(--border);
           border-radius:var(--radius);padding:24px;
           display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap}}
  .player-board{{flex:0 0 auto}}
  #anim-canvas{{border-radius:6px;display:block}}
  .player-controls{{flex:1;min-width:220px;display:flex;
                    flex-direction:column;gap:14px}}
  .frame-info{{font-family:var(--mono)}}
  .frame-label{{font-size:1.1rem;font-weight:700;color:var(--accent2);
                margin-bottom:2px}}
  .frame-desc{{font-size:.82rem;color:var(--muted)}}
  .frame-counter{{font-size:.75rem;color:var(--muted);margin-top:4px}}
  .ctrl-btns{{display:flex;gap:8px;flex-wrap:wrap}}
  .btn{{background:var(--surface);border:1px solid var(--border);
        border-radius:6px;color:var(--accent2);font-family:var(--mono);
        font-size:.82rem;padding:7px 14px;cursor:pointer;
        transition:background .15s,border-color .15s}}
  .btn:hover{{background:var(--card);border-color:var(--accent)}}
  .btn.primary{{background:var(--accent);color:#1a1208;border-color:var(--accent);
                font-weight:700}}
  .btn.primary:hover{{background:var(--accent2);border-color:var(--accent2)}}
  .btn:disabled{{opacity:.35;cursor:not-allowed}}
  .speed-row{{display:flex;align-items:center;gap:10px;font-size:.78rem;
              color:var(--muted)}}
  input[type=range]{{flex:1;accent-color:var(--accent)}}
  .legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:4px}}
  .legend-item{{display:flex;align-items:center;gap:5px;
                font-size:.76rem;color:var(--muted)}}
  .dot{{width:11px;height:11px;border-radius:50%;display:inline-block}}
  .progress-bar{{height:4px;background:var(--border);
                 border-radius:2px;overflow:hidden}}
  .progress-fill{{height:100%;background:var(--accent);
                  transition:width .2s;width:0%}}

  /* 到達可能初期盤面 */
  .init-scroll{{display:flex;gap:12px;overflow-x:auto;
                padding-bottom:10px;flex-wrap:wrap}}
  .init-card{{background:var(--card);border:1px solid var(--border);
              border-radius:var(--radius);padding:10px;text-align:center}}
  .init-label{{font-size:.72rem;color:var(--muted);margin-bottom:4px;
               font-family:var(--mono)}}
  .more{{color:var(--muted);font-size:.82rem;padding-top:8px}}
</style>
</head>
<body>

<header>
  <h1>ペグソリティア — ZDD解析結果</h1>
  <div class="sub">盤面: {_html.escape(board_name)} &nbsp;|&nbsp;
    reachable ZDD 高速解法 &nbsp;|&nbsp; 合法手数: {len(MOVES)}
  </div>
</header>

<div class="summary">
  <div class="kpi"><div class="label">逆向き探索時間</div>
    <div class="value">{bwd_time:.1f}<span class="unit">s</span></div></div>
  <div class="kpi"><div class="label">総手数</div>
    <div class="value">{len(move_steps)}<span class="unit">手</span></div></div>
  <div class="kpi"><div class="label">到達可能初期盤面数</div>
    <div class="value">{num_reachable:,}<span class="unit">盤面</span></div></div>
</div>

<section>
  <h2>解の手順アニメーション（{len(move_steps)} 手）</h2>
  <div class="player">
    <div class="player-board">
      <canvas id="anim-canvas"></canvas>
    </div>
    <div class="player-controls">
      <div class="frame-info">
        <div class="frame-label" id="frame-label">初期盤面</div>
        <div class="frame-desc"  id="frame-desc"></div>
        <div class="frame-counter" id="frame-counter">0 / {len(move_steps)}</div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
      <div class="ctrl-btns">
        <button class="btn" id="btn-first"  title="最初">⏮</button>
        <button class="btn" id="btn-prev"   title="前へ">◀</button>
        <button class="btn primary" id="btn-play" title="再生/停止">▶ 再生</button>
        <button class="btn" id="btn-next"   title="次へ">▶</button>
        <button class="btn" id="btn-last"   title="最後">⏭</button>
      </div>
      <div class="speed-row">
        <span>速度</span>
        <input type="range" id="speed" min="100" max="2000" value="800" step="100">
        <span id="speed-val">0.8s/手</span>
      </div>
      <div class="legend">
        <div class="legend-item">
          <span class="dot" style="background:#4a3728;border:1px solid #2a1a10"></span>ペグ</div>
        <div class="legend-item">
          <span class="dot" style="background:#e05c5c"></span>移動元</div>
        <div class="legend-item">
          <span class="dot" style="background:#e08c3c"></span>飛び越え</div>
        <div class="legend-item">
          <span class="dot" style="background:#5cb85c"></span>移動先</div>
      </div>
    </div>
  </div>
</section>

<section>
  <h2>到達可能な初期盤面（ペグ {N_PEGS} 個，空き: {INIT_HOLE}，ゴール: {GOAL_POS}）</h2>
  <div class="init-scroll">{init_html}</div>
</section>

<section>
  <h2>逆向き探索（reachable ZDD 構築）</h2>
  {log_table(bwd_log, f"ゴールから逆向きに展開 · 総時間 {bwd_time:.2f}s")}
</section>

<script>
(function() {{
  const DATA = {anim_json};

  // Canvas セットアップ
  const canvas = document.getElementById('anim-canvas');
  const ctx    = canvas.getContext('2d');
  const DPR    = window.devicePixelRatio || 1;
  const W      = DATA.board_w;
  const H      = DATA.board_h;
  canvas.width  = W * DPR;
  canvas.height = H * DPR;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(DPR, DPR);

  // cells を pid→[cx,cy] に変換
  const cellMap = {{}};
  DATA.cells.forEach(([pid,cx,cy]) => {{ cellMap[pid] = [cx,cy]; }});

  const COL = {{
    bg:    '#e8d5b0',
    hole:  '#c8a97e',
    hstr:  '#9a7a55',
    peg:   '#4a3728',
    pstr:  '#2a1a10',
    from:  '#e05c5c',
    over:  '#e08c3c',
    to:    '#5cb85c',
  }};

  function drawFrame(fi) {{
    const f = DATA.frames[fi];
    const pegSet = new Set(f.pegs);

    ctx.clearRect(0, 0, W, H);

    // 背景
    ctx.fillStyle = COL.bg;
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 4);
    ctx.fill();

    // 穴
    DATA.cells.forEach(([pid,cx,cy]) => {{
      ctx.beginPath();
      ctx.arc(cx, cy, DATA.r_hole, 0, Math.PI*2);
      ctx.fillStyle = COL.hole;
      ctx.fill();
      ctx.strokeStyle = COL.hstr;
      ctx.lineWidth = 1;
      ctx.stroke();
    }});

    // ペグ（移動アニメーション中は fr/ov を薄く表示）
    DATA.cells.forEach(([pid,cx,cy]) => {{
      if (!pegSet.has(pid)) return;
      let fill = COL.peg;
      if      (pid === f.fr) fill = COL.from;
      else if (pid === f.ov) fill = COL.over;
      else if (pid === f.to) fill = COL.to;

      ctx.beginPath();
      ctx.arc(cx, cy, DATA.r_peg, 0, Math.PI*2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = COL.pstr;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }});

    // 矢印: fr → ov → to
    if (f.fr !== null) {{
      const [x1,y1] = cellMap[f.fr];
      const [xm,ym] = cellMap[f.ov];
      const [x2,y2] = cellMap[f.to];
      ctx.save();
      ctx.setLineDash([4,3]);
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = 'rgba(232,184,75,0.7)';
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(xm, ym);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      // 矢先
      const dx = x2 - xm, dy = y2 - ym;
      const ang = Math.atan2(dy, dx);
      const al = 8;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - al*Math.cos(ang-0.4), y2 - al*Math.sin(ang-0.4));
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - al*Math.cos(ang+0.4), y2 - al*Math.sin(ang+0.4));
      ctx.strokeStyle = '#e8b84b';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }}
  }}

  // UI 更新
  let curFrame = 0;
  const total  = DATA.frames.length - 1;

  function updateUI(fi) {{
    curFrame = fi;
    const f = DATA.frames[fi];
    document.getElementById('frame-label').textContent   = f.label;
    document.getElementById('frame-desc').textContent    = f.desc;
    document.getElementById('frame-counter').textContent = fi + ' / ' + total;
    document.getElementById('progress').style.width      = (fi/total*100) + '%';
    document.getElementById('btn-prev').disabled  = fi <= 0;
    document.getElementById('btn-first').disabled = fi <= 0;
    document.getElementById('btn-next').disabled  = fi >= total;
    document.getElementById('btn-last').disabled  = fi >= total;
    drawFrame(fi);
  }}

  // 自動再生
  let timer    = null;
  let interval = 800;

  function stopPlay() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
    document.getElementById('btn-play').textContent = '▶ 再生';
  }}

  function startPlay() {{
    stopPlay();
    timer = setInterval(() => {{
      if (curFrame >= total) {{ stopPlay(); return; }}
      updateUI(curFrame + 1);
    }}, interval);
    document.getElementById('btn-play').textContent = '⏸ 停止';
  }}

  document.getElementById('btn-play').addEventListener('click', () => {{
    if (timer) stopPlay();
    else {{
      if (curFrame >= total) updateUI(0);
      startPlay();
    }}
  }});
  document.getElementById('btn-first').addEventListener('click', () => {{ stopPlay(); updateUI(0); }});
  document.getElementById('btn-last' ).addEventListener('click', () => {{ stopPlay(); updateUI(total); }});
  document.getElementById('btn-prev' ).addEventListener('click', () => {{ stopPlay(); updateUI(Math.max(0, curFrame-1)); }});
  document.getElementById('btn-next' ).addEventListener('click', () => {{ stopPlay(); updateUI(Math.min(total, curFrame+1)); }});

  const speedEl = document.getElementById('speed');
  speedEl.addEventListener('input', () => {{
    interval = parseInt(speedEl.value);
    document.getElementById('speed-val').textContent = (interval/1000).toFixed(1) + 's/手';
    if (timer) {{ stopPlay(); startPlay(); }}
  }});

  // キーボード操作
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight') {{ stopPlay(); updateUI(Math.min(total, curFrame+1)); }}
    if (e.key === 'ArrowLeft')  {{ stopPlay(); updateUI(Math.max(0, curFrame-1)); }}
    if (e.key === ' ')          {{ e.preventDefault(); document.getElementById('btn-play').click(); }}
  }});

  updateUI(0);
}})();
</script>
</body>
</html>"""

    Path(out_path).write_text(page, encoding="utf-8")
    print(f"HTML 出力: {out_path}")
# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("逆向き探索...")
    reachable, bwd_log, bwd_time = build_reachable()
    print(f"  完了 ({bwd_time:.2f}s)")

    # print("Phase 2: 前向き探索...")
    # goal_f, fwd_log, fwd_time = solve_with_reachable(reachable)
    # print(f"  完了 ({fwd_time:.2f}s)  ゴール盤面数: {int(goal_f.exact_count)}")

    print("手順復元...")
    move_steps = trace_with_reachable(reachable)
    print(f"  {len(move_steps)} 手")

    print("到達可能初期盤面を列挙...")
    F_init        = reachable[N_STEPS].choose(N_PEGS)
    num_reachable = int(F_init.exact_count)
    limit         = min(num_reachable, 20)
    reachable_inits = [frozenset(F_init.unrank(i)) for i in range(limit)]

    print("HTML 生成中...")
    generate_html(
        bwd_log=bwd_log, bwd_time=bwd_time,
        reachable_inits=reachable_inits,
        num_reachable=num_reachable,
        move_steps=move_steps,
    )