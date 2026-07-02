r"""
ダイヤモンドゲーム（中国式チェッカー1人用）— ZDD による最短手数解法
===================================================================

【ルール】
  盤面: 六芒星（121マス）
  初期: 上頂点三角形（10個のペグ）
  ゴール: 下頂点三角形（10個の位置）に全ペグを移動
  移動:
    (A) 1歩移動: 隣接する空きマスへ移動
    (B) ジャンプ: 隣接するペグを飛び越えて空きマスへ
        ジャンプは1ターン内に連続して何回でも可能
  目標: 最短ターン数で初期配置からゴール配置へ移動

【ZDD エンコーディング】
  変数 p = pos_id (1..121)
  ZDDの各集合 = ペグのある位置IDの集合 = 1つの盤面状態
  ZDD全体 = n手後に到達可能な全盤面の族

【アルゴリズム】
  盤面族ZDDによる状態遷移BFS:
    F_0 = {初期盤面}
    F_{n+1} = next_all(F_n)   (全移動を適用)
  ゴール盤面が F_t に初めて現れた t が最短手数

【移動の ZDD 表現】
  (A) 1歩移動 (fr → to):
      F.onset(fr).offset(to).change(fr).change(to)
      「fr にペグがあり to が空き」の盤面で fr↔to を入れ替え

  (B) ジャンプ (fr → to, ov を飛び越え):
      F.onset(fr).onset(ov).offset(to).change(fr).change(ov).change(to)

  連続ジャンプ: 1ターン内に複数ジャンプを組み合わせる
    各ペグについて到達可能な全終点を求め，
    それを1回の ZDD 演算として適用する
"""

import time
from kyotodd import ZDD
from collections import defaultdict as _dd


# ============================================================
# 盤面定義
# ============================================================

HEX_DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

def make_board_and_arms(center_N=5, arm_N=4):
    """
    中国式チェッカー盤（六芒星）を生成する．
    center_N=5, arm_N=4 → 121マス（標準盤）
    """
    # 中央正六角形
    central = set()
    for q in range(-(center_N-1), center_N):
        for r in range(-(center_N-1), center_N):
            if abs(q+r) <= center_N-1:
                central.add((q,r))

    # 6頂点三角形
    arm_tips = [
        (0, -(center_N+arm_N-1),  1, 0,  0, 1),
        (center_N+arm_N-1, -(center_N+arm_N-1), -1, 0, 0, 1),
        (center_N+arm_N-1, 0,  0,-1, -1, 1),
        (0,  center_N+arm_N-1, -1, 0,  0,-1),
        (-(center_N+arm_N-1), center_N+arm_N-1, 1, 0, 0,-1),
        (-(center_N+arm_N-1), 0,  0, 1,  1,-1),
    ]
    arms = []
    for tq,tr,d1q,d1r,d2q,d2r in arm_tips:
        arm = frozenset(
            (tq+d1q*i+d2q*j, tr+d1r*i+d2r*j)
            for i in range(arm_N) for j in range(arm_N-i)
        )
        arms.append(arm)

    board = central | set().union(*arms)
    return frozenset(board), central, arms

BOARD, CENTRAL, ARMS = make_board_and_arms(5, 4)
POS_LIST = sorted(BOARD)
POS2ID   = {p: i+1 for i, p in enumerate(POS_LIST)}
ID2POS   = {v: p for p, v in POS2ID.items()}

N_POS  = len(POS_LIST)   # 121
N_PEGS = len(ARMS[0])    # 10

# 初期: 腕0（上），ゴール: 腕3（下）
INIT_STATE = frozenset(POS2ID[p] for p in ARMS[0])
GOAL_STATE = frozenset(POS2ID[p] for p in ARMS[3])

# 変数空間の初期化
_dummy = ZDD.power_set(N_POS)
del _dummy

print(f"盤面: {N_POS}マス  ペグ数: {N_PEGS}")
print(f"初期位置（腕0）: {len(INIT_STATE)}個")
print(f"ゴール位置（腕3）: {len(GOAL_STATE)}個")


# ============================================================
# 移動の事前計算
# ============================================================

# 1歩移動: (fr_id, to_id)
SINGLE_MOVES: list[tuple[int,int]] = []
for (q,r) in POS_LIST:
    for dq,dr in HEX_DIRS:
        to = (q+dq, r+dr)
        if to in POS2ID:
            SINGLE_MOVES.append((POS2ID[(q,r)], POS2ID[to]))

# 単発ジャンプ: (fr_id, ov_id, to_id)
SINGLE_JUMPS: list[tuple[int,int,int]] = []
for (q,r) in POS_LIST:
    for dq,dr in HEX_DIRS:
        ov = (q+dq,  r+dr)
        to = (q+2*dq, r+2*dr)
        if ov in POS2ID and to in POS2ID:
            SINGLE_JUMPS.append((POS2ID[(q,r)], POS2ID[ov], POS2ID[to]))

# 連続ジャンプの事前計算
# 各マスから連続ジャンプで到達可能な全終点リストを求める（盤面固定時）
# 実際の盤面状態は実行時に変わるので，ZDD演算で対応する

# 連続ジャンプの ZDD 表現:
# 1ターンに「fr → [連続ジャンプ] → to」を全て列挙する代わりに，
# ジャンプを何度でも繰り返せる不動点演算を使う:
#   jump_closure(F) = F と「1ジャンプ適用後の族」の和集合の不動点
# これにより連続ジャンプ後の全到達盤面が得られる

# fr でグループ化
MOVES_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for fr, to in SINGLE_MOVES:
    MOVES_BY_FR[fr].append(('step', to))
for fr, ov, to in SINGLE_JUMPS:
    MOVES_BY_FR[fr].append(('jump', ov, to))

JUMPS_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for fr, ov, to in SINGLE_JUMPS:
    JUMPS_BY_FR[fr].append((ov, to))


# ============================================================
# ZDD 遷移演算
# ============================================================

def apply_single_step(F: ZDD, fr: int, to: int) -> ZDD:
    """1歩移動"""
    return F.onset(fr).offset(to).change(fr).change(to)


def apply_single_jump(F: ZDD, fr: int, ov: int, to: int) -> ZDD:
    """1回ジャンプ"""
    return F.onset(fr).onset(ov).offset(to).change(fr).change(ov).change(to)


def jump_closure(F: ZDD) -> ZDD:
    """
    連続ジャンプの不動点演算:
    ジャンプを繰り返し適用し，新たな盤面が生まれなくなるまで続ける.
    F_0 = F
    F_{k+1} = F_k ∪ {1ジャンプ適用後の全盤面}
    収束したら F_k を返す．

    これにより「任意回数の連続ジャンプ後の全到達盤面」が得られる．
    """
    current = F
    active = set(current.support_vars())
    while True:
        new_states = ZDD.empty
        for fr, ov_to_list in JUMPS_BY_FR.items():
            if fr not in active:
                continue
            F_fr = current.onset(fr)
            for ov, to in ov_to_list:
                new_states = new_states + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
        merged = current + new_states
        if int(merged.exact_count) == int(current.exact_count):
            break
        current = merged
        active = set(current.support_vars())
    return current


def next_all(F: ZDD) -> ZDD:
    """
    F の全盤面から1ターンの全移動を適用した盤面族を返す．

    1ターン = 1歩移動 OR 任意回数の連続ジャンプ

    実装:
      (1) 1歩移動の適用
      (2) 各ペグについて連続ジャンプを適用（jump_closure）
      (3) 両者の和集合
    """
    active = set(F.support_vars())
    result = ZDD.empty

    # (1) 1歩移動
    for fr, to in SINGLE_MOVES:
        if fr not in active:
            continue
        result = result + apply_single_step(F, fr, to)

    # (2) 連続ジャンプ: 各ペグを起点に jump_closure を計算
    # ただし全ペグ同時に jump_closure を取ると混在するため，
    # 各ペグ fr について「fr を動かした後の盤面」に jump_closure を適用
    for fr in list(JUMPS_BY_FR.keys()):
        if fr not in active:
            continue
        # fr を1回以上ジャンプさせた盤面族
        # まず fr を動かした直後の盤面を収集
        F_fr = F.onset(fr)
        after_first = ZDD.empty
        for ov, to in JUMPS_BY_FR[fr]:
            after_first = after_first + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
        if int(after_first.exact_count) == 0:
            continue
        # その後も fr（移動後の位置）から連続してジャンプできる
        # → jump_closure で全到達盤面を求める（fr 以外のペグも使える）
        closed = jump_closure(after_first)
        result = result + closed

    return result


# ============================================================
# BFS による最短手数探索
# ============================================================

def solve_bfs() -> tuple[int, list[dict]]:
    """
    ZDD BFS で最短手数を求める．
    戻り値: (最短手数, 各ステップのログ)
    """
    t0   = time.perf_counter()
    F    = ZDD.from_sets([sorted(INIT_STATE)])
    seen = F   # 既訪問盤面の和集合（重複排除用）
    log  = []

    for step in range(1, 200):
        F_next = next_all(F)
        # 既訪問盤面を除去（新規盤面のみ）
        F_next = F_next - seen
        elapsed = time.perf_counter() - t0

        cnt = int(F_next.exact_count)
        log.append({
            "step":    step,
            "elapsed": elapsed,
            "count":   cnt,
            "nodes":   F_next.raw_size,
        })
        print(f"  ステップ {step:3d} ({elapsed:.2f}s)  新規盤面数: {cnt:8,}  "
              f"ZDDノード: {F_next.raw_size:6,}")

        if cnt == 0:
            print("  到達可能な新規盤面なし → 探索終了")
            return -1, log

        seen = seen + F_next

        # ゴール到達確認
        # GOAL_STATE が F_next に含まれるか
        goal_check = F_next
        for pid in sorted(GOAL_STATE):
            goal_check = goal_check.onset(pid)
        for pid in POS2ID.values():
            if pid not in GOAL_STATE:
                goal_check = goal_check.offset(pid)

        if int(goal_check.exact_count) > 0:
            print(f"\n  ★ 最短手数: {step} 手 ({time.perf_counter()-t0:.2f}s)")
            return step, log

        F = F_next

    return -1, log


# ============================================================
# 盤面表示
# ============================================================

def format_board(state: frozenset[int]) -> list[str]:
    """盤面を文字列リストとして返す"""
    qs = sorted(set(q for q,r in BOARD))
    rs = sorted(set(r for q,r in BOARD))
    lines = []
    for r in rs:
        off = abs(r)
        row = " " * off
        for q in qs:
            if (q,r) in POS2ID:
                pid = POS2ID[(q,r)]
                row += "o " if pid in state else ". "
            else:
                row += "  "
        lines.append(row.rstrip())
    return lines


# ============================================================
# HTML 生成
# ============================================================

def board_svg(state: frozenset[int], px: int = 12) -> str:
    """六角グリッドの SVG"""
    qs = sorted(set(q for q,r in BOARD))
    rs = sorted(set(r for q,r in BOARD))

    # 六角グリッドを正方形グリッドに射影（axial→pixel）
    # x = px*(q + r/2), y = px*(r*sqrt(3)/2)
    import math
    h = math.sqrt(3)/2

    cells = []
    xs_all, ys_all = [], []
    for r in rs:
        for q in qs:
            if (q,r) not in POS2ID:
                continue
            x = px * (q + r * 0.5)
            y = px * (r * h)
            xs_all.append(x); ys_all.append(y)

    if not xs_all:
        return ""

    min_x, min_y = min(xs_all)-px, min(ys_all)-px
    max_x, max_y = max(xs_all)+px, max(ys_all)+px
    W = max_x - min_x + 2
    H = max_y - min_y + 2

    for r in rs:
        for q in qs:
            if (q,r) not in POS2ID:
                continue
            pid = POS2ID[(q,r)]
            x = px*(q + r*0.5) - min_x + 1
            y = px*(r*h) - min_y + 1

            # 穴
            cells.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{px*0.42:.1f}" '
                          f'fill="#c8a97e" stroke="#9a7a55" stroke-width="0.8"/>')
            if pid in state:
                # ペグの色: GOAL_STATE は緑，INIT_STATE は青，それ以外は茶
                if pid in GOAL_STATE:
                    fill = "#5cb85c"
                elif pid in INIT_STATE:
                    fill = "#5b9bd5"
                else:
                    fill = "#4a3728"
                cells.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{px*0.3:.1f}" '
                              f'fill="{fill}" stroke="#2a1a10" stroke-width="0.8"/>')

    return (f'<svg width="{W:.0f}" height="{H:.0f}" '
            f'viewBox="0 0 {W:.0f} {H:.0f}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
            f'<rect width="{W:.0f}" height="{H:.0f}" fill="#e8d5b0" rx="4"/>'
            + "".join(cells) + "</svg>")


def log_table_html(log: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{e['step']}</td><td>{e['elapsed']:.2f}</td>"
        f"<td>{e['count']:,}</td><td>{e['nodes']:,}</td></tr>"
        for e in log
    )
    return f"""
<div class="table-wrap"><table>
  <thead><tr>
    <th>ステップ</th><th>経過時間 (s)</th><th>新規盤面数</th><th>ZDDノード数</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table></div>"""


def generate_html(min_moves: int, log: list[dict], total_time: float,
                  out: str = "diamond_game_result.html"):
    from pathlib import Path

    init_svg = board_svg(INIT_STATE, px=10)
    goal_svg = board_svg(GOAL_STATE, px=10)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ダイヤモンドゲーム ZDD解析結果</title>
<style>
  :root{{--bg:#0f1a2e;--surface:#162236;--card:#1c2d45;--border:#2a4a6b;
        --accent:#4a9eff;--accent2:#7bc4ff;--text:#d0e8ff;--muted:#6a8aaa;
        --green:#5cb85c;--blue:#5b9bd5;--radius:8px;
        --mono:"JetBrains Mono","Fira Code",monospace}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);
        font-family:"Noto Sans JP","Hiragino Sans",sans-serif;
        line-height:1.6;padding-bottom:60px}}
  header{{background:var(--surface);border-bottom:2px solid var(--accent);
          padding:28px 40px 20px}}
  header h1{{font-size:1.7rem;font-weight:700;color:var(--accent2);
             letter-spacing:.04em;margin-bottom:4px}}
  header .sub{{color:var(--muted);font-size:.88rem}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
            gap:14px;padding:28px 40px 0}}
  .kpi{{background:var(--card);border:1px solid var(--border);
        border-radius:var(--radius);padding:16px 18px}}
  .kpi .label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:5px}}
  .kpi .value{{font-size:1.7rem;font-weight:700;color:var(--accent2);
               font-family:var(--mono)}}
  .kpi .unit{{font-size:.78rem;color:var(--muted);margin-left:3px}}
  section{{padding:36px 40px 0}}
  section h2{{font-size:1.05rem;font-weight:600;color:var(--accent);
              border-left:3px solid var(--accent);padding-left:10px;
              margin-bottom:16px;letter-spacing:.04em}}
  .boards{{display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start}}
  .board-card{{background:var(--card);border:1px solid var(--border);
               border-radius:var(--radius);padding:16px;text-align:center}}
  .board-label{{font-size:.78rem;color:var(--accent2);font-weight:600;
                margin-bottom:10px;font-family:var(--mono)}}
  .table-wrap{{overflow-x:auto;border-radius:var(--radius);
               border:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.83rem}}
  thead tr{{background:var(--surface)}}
  th{{padding:9px 14px;text-align:right;color:var(--accent2);font-weight:600;
      font-size:.76rem;letter-spacing:.06em;border-bottom:1px solid var(--border)}}
  th:first-child{{text-align:center}}
  td{{padding:6px 14px;text-align:right;border-bottom:1px solid #1c2d45;color:var(--text)}}
  td:first-child{{text-align:center;color:var(--muted)}}
  tr:last-child td{{border-bottom:none}}
  tbody tr:hover{{background:rgba(74,158,255,.07)}}
  .legend{{display:flex;gap:18px;margin-bottom:12px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:6px;font-size:.78rem;color:var(--muted)}}
  .dot{{width:11px;height:11px;border-radius:50%;display:inline-block}}
  .result-box{{background:var(--card);border:2px solid var(--accent);
               border-radius:var(--radius);padding:20px 28px;
               font-size:1.05rem;color:var(--text);font-family:var(--mono)}}
  .result-box .big{{font-size:2.2rem;font-weight:700;color:var(--accent2);
                    margin-right:8px}}
  .rules{{background:var(--card);border:1px solid var(--border);
          border-radius:var(--radius);padding:16px 20px;font-size:.84rem;
          color:var(--muted);line-height:1.9}}
  .rules b{{color:var(--accent2)}}
</style>
</head>
<body>
<header>
  <h1>ダイヤモンドゲーム — ZDD 最短手数解析</h1>
  <div class="sub">中国式チェッカー 1人用 | 六芒星 {N_POS}マス | 10ペグ × 6腕</div>
</header>

<div class="summary">
  <div class="kpi"><div class="label">最短手数</div>
    <div class="value">{min_moves if min_moves>0 else "?"}<span class="unit">手</span></div></div>
  <div class="kpi"><div class="label">総計算時間</div>
    <div class="value">{total_time:.1f}<span class="unit">s</span></div></div>
  <div class="kpi"><div class="label">盤面マス数</div>
    <div class="value">{N_POS}<span class="unit">マス</span></div></div>
  <div class="kpi"><div class="label">ペグ数</div>
    <div class="value">{N_PEGS}<span class="unit">個</span></div></div>
  <div class="kpi"><div class="label">合法手数（1歩）</div>
    <div class="value">{len(SINGLE_MOVES):,}</div></div>
  <div class="kpi"><div class="label">合法手数（ジャンプ）</div>
    <div class="value">{len(SINGLE_JUMPS):,}</div></div>
</div>

<section>
  <h2>ルール</h2>
  <div class="rules">
    <b>盤面:</b> 六芒星（中央正六角形61マス + 6頂点三角形各10マス = 121マス）<br>
    <b>初期配置:</b> 上頂点三角形（腕0）に10個のペグ<br>
    <b>ゴール:</b> 下頂点三角形（腕3）に10個全てのペグを移動<br>
    <b>移動:</b> (A) 1歩移動: 隣接する空きマスへ &nbsp;
               (B) ジャンプ: 隣接ペグを飛び越えて空きマスへ（1ターン内に連続可能）<br>
    <b>目標:</b> 最短ターン数でゴールへ
  </div>
</section>

<section>
  <h2>初期盤面 / ゴール盤面</h2>
  <div class="legend">
    <div class="legend-item"><span class="dot" style="background:#5b9bd5"></span>初期配置（腕0）</div>
    <div class="legend-item"><span class="dot" style="background:#5cb85c"></span>ゴール位置（腕3）</div>
    <div class="legend-item"><span class="dot" style="background:#c8a97e;border:1px solid #9a7a55"></span>空きマス</div>
  </div>
  <div class="boards">
    <div class="board-card">
      <div class="board-label">初期盤面</div>
      {init_svg}
    </div>
    <div class="board-card">
      <div class="board-label">ゴール盤面</div>
      {goal_svg}
    </div>
  </div>
</section>

<section>
  <h2>最短手数の結果</h2>
  <div class="result-box">
    <span class="big">{min_moves if min_moves>0 else "?"}</span>手
    &nbsp;で上頂点三角形から下頂点三角形への移動が可能
    &nbsp;（総計算時間: {total_time:.2f}s）
  </div>
</section>

<section>
  <h2>BFS 探索ログ（各ステップの新規到達盤面数 / ZDD ノード数）</h2>
  {log_table_html(log)}
</section>

</body>
</html>"""

    Path(out).write_text(html, encoding="utf-8")
    print(f"HTML 出力: {out}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ダイヤモンドゲーム — ZDD BFS 最短手数探索")
    print("=" * 60)
    print(f"  盤面: {N_POS}マス  ペグ: {N_PEGS}個")
    print(f"  1歩移動: {len(SINGLE_MOVES)}手  単発ジャンプ: {len(SINGLE_JUMPS)}手")
    print()

    t_start = time.perf_counter()
    min_moves, log = solve_bfs()
    total_time = time.perf_counter() - t_start

    print("\nHTML 生成中...")
    generate_html(min_moves, log, total_time)