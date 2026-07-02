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
    return valid, center, grid_size


def make_french_board():
    """フランス式 37マス（角を含む十字型）"""
    valid = set()
    for r in range(1, 8):
        for c in range(1, 8):
            if (abs(r-4) + abs(c-4) <= 4):
                valid.add((r, c))
    center = (4, 4)
    grid_size = 7
    return valid, center, grid_size


def make_diamond_board():
    """ダイヤモンド型 41マス"""
    valid = set()
    for r in range(1, 10):
        for c in range(1, 10):
            if abs(r - 5) + abs(c - 5) <= 4:
                valid.add((r, c))
    center = (5, 5)
    grid_size = 9
    return valid, center, grid_size


# ── 使用する盤面を選択 ──
VALID, CENTER_POS, GRID_SIZE = make_french_board()

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
N_PEGS  = N_POS - 1              # 初期ペグ数（中央1マスが空き）
N_STEPS = N_PEGS - 1             # 手数

CENTER     = POS2ID[CENTER_POS]
INIT_STATE = frozenset(POS2ID.values()) - {CENTER}
GOAL_STATE = frozenset({CENTER})

# 変数空間の初期化（offset/change が var_count() 超で失敗するのを防ぐ）
_dummy = ZDD.power_set(N_POS)
del _dummy

print(f"盤面: {N_POS}マス  初期ペグ: {N_PEGS}個  手数: {N_STEPS}手")
print(f"合法手数: {len(MOVES)}  中央: {CENTER_POS}")


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
        print(f"  逆ステップ {step+1:2d} ({elapsed:.2f}s)")
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
# SVG 描画（盤面サイズ自動計算）
# ============================================================

def _board_svg_inner(state: frozenset[int],
                     highlight: dict | None,
                     cell: int, r_off: int) -> str:
    circles = []
    for (r, c) in POS_LIST:
        cx  = (c - 1) * cell + r_off
        cy  = (r - 1) * cell + r_off
        pid = POS2ID[(r, c)]

        circles.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r_off-2}" '
            f'fill="#c8a97e" stroke="#9a7a55" stroke-width="1"/>'
        )
        if pid in state:
            if   highlight and pid == highlight.get("from"): fill = "#e05c5c"
            elif highlight and pid == highlight.get("over"): fill = "#e08c3c"
            elif highlight and pid == highlight.get("to"):   fill = "#5cb85c"
            else:                                             fill = "#4a3728"
            circles.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_off-4}" '
                f'fill="{fill}" stroke="#2a1a10" stroke-width="1.5"/>'
            )
    return "".join(circles)


def board_svg(state: frozenset[int],
              highlight: dict | None = None,
              px_per_cell: int = 22) -> str:
    cell      = px_per_cell
    r_off     = cell // 2
    board_px  = GRID_SIZE * cell
    inner     = _board_svg_inner(state, highlight, cell, r_off)
    return (
        f'<svg width="{board_px}" height="{board_px}" '
        f'viewBox="0 0 {board_px} {board_px}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<rect width="{board_px}" height="{board_px}" fill="#e8d5b0" rx="4"/>'
        + inner + "</svg>"
    )


def small_board_svg(state: frozenset[int], px_per_cell: int = 14) -> str:
    cell     = px_per_cell
    r_off    = cell // 2
    board_px = GRID_SIZE * cell
    inner    = _board_svg_inner(state, None, cell, r_off)
    return (
        f'<svg width="{board_px}" height="{board_px}" '
        f'viewBox="0 0 {board_px} {board_px}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<rect width="{board_px}" height="{board_px}" fill="#e8d5b0" rx="3"/>'
        + inner + "</svg>"
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
                  out_path="solution.html"):

    # 手順HTML
    move_html = f"""
    <div class="move-card">
      <div class="move-label">初期盤面</div>
      {board_svg(INIT_STATE)}
    </div>"""
    for s in move_steps:
        hl = {"from": POS2ID[s["fr_pos"]],
              "over": POS2ID[s["ov_pos"]],
              "to":   POS2ID[s["to_pos"]]}
        move_html += f"""
    <div class="move-card">
      <div class="move-label">手 {s['step']}</div>
      <div class="move-desc">{s['fr_pos']} → {s['ov_pos']} → {s['to_pos']}</div>
      {board_svg(s['board'], highlight=hl)}
    </div>"""

    # 到達可能初期盤面HTML
    init_html = "".join(
        f'<div class="init-card"><div class="init-label">#{i+1}</div>'
        f'{small_board_svg(st)}</div>'
        for i, st in enumerate(reachable_inits)
    )
    if num_reachable > len(reachable_inits):
        init_html += f'<p class="more">他 {num_reachable - len(reachable_inits):,} 件</p>'

    board_name = f"{N_POS}マス (中央 {CENTER_POS})"

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
    --red:#e05c5c;--orange:#e08c3c;--green:#5cb85c;
    --radius:8px;--mono:"JetBrains Mono","Fira Code",monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);
        font-family:"Noto Sans JP","Hiragino Sans",sans-serif;
        line-height:1.6;padding-bottom:60px}}
  header{{background:var(--surface);border-bottom:2px solid var(--accent);
          padding:32px 40px 24px}}
  header h1{{font-size:1.8rem;font-weight:700;color:var(--accent2);
             letter-spacing:.04em;margin-bottom:4px}}
  header .sub{{color:var(--muted);font-size:.9rem}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:16px;padding:32px 40px 0}}
  .kpi{{background:var(--card);border:1px solid var(--border);
        border-radius:var(--radius);padding:18px 20px}}
  .kpi .label{{font-size:.75rem;color:var(--muted);text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:6px}}
  .kpi .value{{font-size:1.6rem;font-weight:700;color:var(--accent2);
               font-family:var(--mono)}}
  .kpi .unit{{font-size:.8rem;color:var(--muted);margin-left:4px}}
  section{{padding:40px 40px 0}}
  section h2{{font-size:1.1rem;font-weight:600;color:var(--accent);
              border-left:3px solid var(--accent);padding-left:12px;
              margin-bottom:20px;letter-spacing:.04em}}
  .table-wrap{{overflow-x:auto;border-radius:var(--radius);
               border:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;
         font-family:var(--mono);font-size:.85rem}}
  caption{{text-align:left;padding:10px 14px 6px;font-size:.8rem;
           color:var(--muted);background:var(--surface)}}
  thead tr{{background:var(--surface)}}
  th{{padding:10px 16px;text-align:right;color:var(--accent2);
      font-weight:600;font-size:.78rem;letter-spacing:.06em;
      border-bottom:1px solid var(--border)}}
  th:first-child{{text-align:center}}
  td{{padding:7px 16px;text-align:right;
      border-bottom:1px solid #3a2a14;color:var(--text)}}
  td:first-child{{text-align:center;color:var(--muted)}}
  tr:last-child td{{border-bottom:none}}
  tbody tr:hover{{background:rgba(200,134,10,.08)}}
  .move-scroll{{display:flex;gap:16px;overflow-x:auto;
                padding-bottom:12px;scroll-snap-type:x mandatory}}
  .move-card{{flex:0 0 auto;background:var(--card);
              border:1px solid var(--border);border-radius:var(--radius);
              padding:14px;scroll-snap-align:start;text-align:center}}
  .move-label{{font-size:.75rem;color:var(--accent2);font-weight:700;
               letter-spacing:.06em;margin-bottom:4px;font-family:var(--mono)}}
  .move-desc{{font-size:.72rem;color:var(--muted);margin-bottom:8px;
              font-family:var(--mono)}}
  .legend{{display:flex;gap:20px;margin-bottom:14px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:6px;
                font-size:.8rem;color:var(--muted)}}
  .dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
  .init-scroll{{display:flex;gap:12px;overflow-x:auto;
                padding-bottom:10px;flex-wrap:wrap}}
  .init-card{{background:var(--card);border:1px solid var(--border);
              border-radius:var(--radius);padding:10px;text-align:center}}
  .init-label{{font-size:.72rem;color:var(--muted);margin-bottom:4px;
               font-family:var(--mono)}}
  .more{{color:var(--muted);font-size:.85rem;padding-top:8px}}
</style>
</head>
<body>

<header>
  <h1>ペグソリティア — ZDD解析結果</h1>
  <div class="sub">盤面: {_html.escape(board_name)} &nbsp;|&nbsp;
    reachable ZDD 高速解法 &nbsp;|&nbsp;
    合法手数: {len(MOVES)}
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
  <h2>逆向き探索（reachable ZDD 構築）</h2>
  {log_table(bwd_log, f"ゴールから逆向きに展開 · 総時間 {bwd_time:.2f}s")}
</section>

<section>
  <h2>到達可能な初期盤面（ペグ {N_PEGS} 個，空き1マス）</h2>
  <div class="init-scroll">{init_html}</div>
</section>

<section>
  <h2>解の手順（{len(move_steps)} 手）</h2>
  <div class="legend">
    <div class="legend-item">
      <span class="dot" style="background:#4a3728;border:1.5px solid #2a1a10"></span>ペグ</div>
    <div class="legend-item">
      <span class="dot" style="background:#e05c5c"></span>移動元</div>
    <div class="legend-item">
      <span class="dot" style="background:#e08c3c"></span>飛び越え</div>
    <div class="legend-item">
      <span class="dot" style="background:#5cb85c"></span>移動先</div>
  </div>
  <div class="move-scroll">{move_html}</div>
</section>

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

    # 不要
    # print("前向き探索...")
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
        # fwd_log=fwd_log, fwd_time=fwd_time,
        reachable_inits=reachable_inits,
        num_reachable=num_reachable,
        move_steps=move_steps,
    )