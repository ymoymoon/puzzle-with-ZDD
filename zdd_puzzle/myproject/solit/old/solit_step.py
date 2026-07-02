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
# 6/12 具体的な解く手順を出力させる

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

def solve() -> list[ZDD]:
    """
    前向きに31ステップ展開し，各ステップの盤面族ZDDをリストで返す．
    history[0] = 初期盤面族，history[t] = t手後の盤面族．
    """
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.2f}s"

    F = ZDD.from_sets([sorted(INIT_STATE)])
    history: list[ZDD] = [F]
    print(f"  初期盤面数: {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = next_all(F)
        cnt = int(F.exact_count)
        print(f"  ステップ {step+1:2d} ({elapsed()})  到達盤面数: {cnt}")
        history.append(F)
        if cnt == 0:
            print("  到達可能な盤面がなくなりました")
            break

    print(f"\n  完了 ({elapsed()})")
    return history


def extract_goal(F: ZDD) -> ZDD:
    """F からゴール盤面（CENTER のみペグ）を抽出する"""
    g = F.onset(CENTER)
    for pid in POS2ID.values():
        if pid != CENTER:
            g = g.offset(pid)
    return g


def trace_solution(history: list[ZDD]) -> list[tuple[int,int,int]]:
    """
    history を使ってゴールから逆向きに1手ずつ手を特定し，
    手順リスト [(fr,ov,to), ...] を返す．

    方針:
      ステップ t の盤面 B_t から手 (fr,ov,to) を打って B_{t+1} に遷移したなら，
      B_t = (B_{t+1} △ {fr,ov,to}) であり，
      B_{t+1} には to が含まれ fr,ov が含まれない．

      逆向き操作:
        B_{t+1} を含む族 F_{t+1} に対して，
        手 (fr,ov,to) の逆適用 = onset(to).offset(fr).offset(ov).change(fr).change(ov).change(to)
        で B_t を復元し，それが history[t] に含まれるかを & で確認する．
    """
    # ゴール盤面から出発
    current: ZDD = extract_goal(history[N_STEPS])
    if int(current.exact_count) == 0:
        print("  ゴール到達盤面なし")
        return []

    # ゴール盤面を1つ選ぶ
    board_vars = current.unrank(0)   # 変数番号リスト = ペグ位置IDのリスト
    board: frozenset[int] = frozenset(board_vars)

    move_seq: list[tuple[int,int,int]] = []

    for t in range(N_STEPS, 0, -1):
        found = False
        for fr, ov, to in MOVES:
            # board (ステップt の盤面) から逆算して，ステップt-1の盤面を復元
            # 手(fr,ov,to)の逆: to にペグ，fr/ov が空き → fr/ov にペグ，to が空き
            if to not in board or fr in board or ov in board:
                continue
            prev_board = (board - {to}) | {fr, ov}
            # prev_board が history[t-1] に含まれるか確認
            # ZDDで確認: history[t-1] に prev_board が含まれるか
            # → prev_board の全変数を onset，補集合を offset した ZDD が空でないか
            candidate = history[t-1]
            for pid in sorted(prev_board):
                candidate = candidate.onset(pid)
            for pid in POS2ID.values():
                if pid not in prev_board:
                    candidate = candidate.offset(pid)
            if int(candidate.exact_count) > 0:
                move_seq.append((fr, ov, to))
                board = frozenset(prev_board)
                found = True
                break
        if not found:
            print(f"  ステップ {t} で手が見つかりません")
            break

    move_seq.reverse()
    return move_seq


def print_solution(move_seq: list[tuple[int,int,int]]):
    """手順を盤面アニメーションで表示"""
    state = set(INIT_STATE)
    show_board(frozenset(state), label="初期盤面")
    for t, (fr, ov, to) in enumerate(move_seq):
        state.remove(fr)
        state.remove(ov)
        state.add(to)
        fr_pos = ID2POS[fr]
        ov_pos = ID2POS[ov]
        to_pos = ID2POS[to]
        print(f"\n  手 {t+1:2d}: {fr_pos} → {ov_pos} → {to_pos}")
        show_board(frozenset(state))


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
    history = solve()

    goal_f = extract_goal(history[N_STEPS])
    num_sol = int(goal_f.exact_count)
    print(f"  ゴール到達盤面数: {num_sol}")

    if num_sol > 0:
        print("\n  手順を逆追跡中...")
        move_seq = trace_solution(history)
        print(f"  手順長: {len(move_seq)} 手\n")
        print_solution(move_seq)