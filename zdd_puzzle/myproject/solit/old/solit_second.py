"""
ペグソリティア — 盤面族ZDDによる状態遷移アルゴリズム
=====================================================

【PDFのアルゴリズム】
  盤面全体を V = {(1,3),...,(7,5)} とし，
  盤面 A（ペグのある位置の集合）から遷移できる盤面の集合を next(A) とする:

    next(A) = ⋃ { A ∩ {V \ {(r,c),(r,c±1)} ∪ {(r,c±2)} }
                | (r,c),(r,c±1) ∈ A, (r,c±2) ∉ A }
            ∪ ⋃ { A ∩ {V \ {(r,c),(r±1,c)} ∪ {(r±2,c)} }
                | (r,c),(r±1,c) ∈ A, (r±2,c) ∉ A }

  これをZDDの集合族に拡張:
    F をn手後の全盤面の族とすると，
    F_{n+1} = ⋃_{手 m} { A' | A ∈ F, A' ∈ next_m(A) }

  各手 m = (from, over, to) に対して:
    next_m(F) = F を onset(from) → onset(over) → offset(to) で絞り込み，
                change(from), change(over), change(to) で盤面を更新

【変数エンコーディング】
  変数 p = pos2id[(r,c)]   (1-indexed, 1..33)
  ZDDの各集合 = ペグのある位置IDの集合 = 1つの盤面状態
  ZDD全体 = n手後に到達可能な全盤面の族
"""

import time
from kyotodd import ZDD


# ============================================================
# 盤面定義
# ============================================================

VALID: set[tuple[int,int]] = set()
for _r in range(1, 8):
    for _c in range(1, 8):
        if (_r < 3 or _r > 5) and (_c < 3 or _c > 5):
            continue
        VALID.add((_r, _c))

POS_LIST = sorted(VALID)
POS2ID   = {p: i+1 for i, p in enumerate(POS_LIST)}   # (r,c) → 1..33
ID2POS   = {v: p for p, v in POS2ID.items()}

# 合法手: (from_pos, over_pos, to_pos) の位置ID
MOVES: list[tuple[int,int,int]] = []
for (r, c) in POS_LIST:
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        over = (r+dr, c+dc)
        to   = (r+2*dr, c+2*dc)
        if over in VALID and to in VALID:
            MOVES.append((POS2ID[(r,c)], POS2ID[over], POS2ID[to]))

# fr ごとに手をグループ化: onset(fr) を1回だけ呼んで共有するため
from collections import defaultdict as _dd
MOVES_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_FR[_fr].append((_ov, _to))

N_POS   = len(POS_LIST)   # 33
N_MOVES = len(MOVES)      # 76
N_STEPS = N_POS - 2       # 31  (ペグ32個→31手→ペグ1個)

CENTER     = POS2ID[(4, 4)]
INIT_STATE = frozenset(POS2ID.values()) - {CENTER}
GOAL_STATE = frozenset({CENTER})


# ============================================================
# 盤面表示
# ============================================================

def show_board(state: frozenset, label: str = ""):
    if label:
        print(f"  {label}")
    grid = [['.' if (r,c) in VALID else ' ' for c in range(1,8)] for r in range(1,8)]
    for pid in state:
        r, c = ID2POS[pid]
        grid[r-1][c-1] = 'o'
    for row in grid:
        print("  " + " ".join(row))


# ============================================================
# PDFアルゴリズム: ZDD上での next 演算
# ============================================================

def next_all(F: ZDD) -> ZDD:
    """
    盤面族 F から全合法手を適用した結果の族（和集合）を返す．

    最適化1: support_vars() で F に実際に存在する変数（ペグ位置）を取得し
             fr がその中にない手はスキップ → onset が空を返す無駄な呼び出しを排除

    最適化2: MOVES_BY_FR で fr ごとに onset(fr) を1回だけ呼んで共有
             同じ fr から出る複数の手で ZDD の絞り込み結果を再利用
    """
    active = set(F.support_vars())   # F に実際に存在するペグ位置ID

    result = ZDD.empty
    for fr, ov_to_list in MOVES_BY_FR.items():
        if fr not in active:         # fr のペグが F に存在しなければ全スキップ
            continue
        F_fr = F.onset(fr)           # onset(fr) を1回だけ呼ぶ
        for ov, to in ov_to_list:
            result = result + F_fr.onset(ov).offset(to).change(fr).change(ov).change(to)
    return result


# ============================================================
# メインソルバー
# ============================================================

def solve():
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.2f}s"

    # 初期盤面族: 初期状態1つだけを含むZDD
    F = ZDD.from_sets([sorted(INIT_STATE)])
    print(f"  初期盤面数: {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = next_all(F)
        cnt = int(F.exact_count)
        print(f"  ステップ {step+1:2d} ({elapsed()})  到達盤面数: {cnt}")
        if cnt == 0:
            print("  到達可能な盤面がなくなりました")
            break

    # ゴール状態が F に含まれるか確認
    # ゴール = CENTER のみペグ → onset(CENTER) で絞り，
    # 他の全変数を offset してゴールだけを抽出
    goal_f = F.onset(CENTER)
    for pid in POS2ID.values():
        if pid != CENTER:
            goal_f = goal_f.offset(pid)

    num_sol = int(goal_f.exact_count)
    print(f"\n  完了 ({elapsed()})")
    print(f"  ゴール到達盤面数: {num_sol}")

    return F, goal_f


# ============================================================
# 解の検証
# ============================================================

def verify_goal(goal_f: ZDD):
    """ゴール盤面族を列挙して表示"""
    sols = goal_f.enumerate()
    print(f"\n  ゴール盤面一覧 ({len(sols)} 件):")
    for s in sols:
        show_board(frozenset(s))


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  ペグソリティア — 盤面族ZDD 状態遷移アルゴリズム")
    print("=" * 55)
    show_board(INIT_STATE, label="初期盤面")
    print()
    show_board(GOAL_STATE, label="ゴール盤面")
    print()

    print("  ZDD 状態遷移中...")
    F, goal_f = solve()

    verify_goal(goal_f)