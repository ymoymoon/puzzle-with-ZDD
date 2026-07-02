r"""
ペグソリティア — 盤面族ZDDによる状態遷移アルゴリズム
=====================================================

【PDFのアルゴリズム】
  盤面全体を V = {(1,3),...,(7,5)} とし，
  盤面 A（ペグのある位置の集合）から遷移できる盤面の集合を next(A) とする:

    next(A) = U { A ∩ (V \ {(r,c),(r,c±1)} ∪ {(r,c±2)} )
                | (r,c),(r,c±1) ∈ A, (r,c±2) ∉ A }
            ∪ U { A ∩ (V \ {(r,c),(r±1,c)} ∪ {(r±2,c)} )
                | (r,c),(r±1,c) ∈ A, (r±2,c) ∉ A }

【変数エンコーディング】
  変数 p = pos2id[(r,c)]   (1-indexed, 1..33)
  ZDDの各集合 = ペグのある位置IDの集合 = 1つの盤面状態
  ZDD全体     = n手後に到達可能な全盤面の族

【reachable ZDD を利用した高速解法】
  reachable[t] = ゴールからちょうど t 手逆向きに戻った全盤面の族

  前向き探索時に毎ステップ:
    F = next_all(F) & reachable[N_STEPS - step]
  とすることで，「今後ゴールに到達できない盤面」を即座に除去できる．
  F のサイズが常に最小に保たれ，探索全体が高速化される．

  手順追跡も reachable を直接利用し，history の保存が不要になる．
"""
# 6/12 逆方向からの探索で到達可能な盤面を列挙，その後それを利用してとく

import time
from kyotodd import ZDD
from collections import defaultdict as _dd


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
POS2ID   = {p: i+1 for i, p in enumerate(POS_LIST)}
ID2POS   = {v: p for p, v in POS2ID.items()}

MOVES: list[tuple[int,int,int]] = []
for (r, c) in POS_LIST:
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        over = (r+dr, c+dc)
        to   = (r+2*dr, c+2*dc)
        if over in VALID and to in VALID:
            MOVES.append((POS2ID[(r,c)], POS2ID[over], POS2ID[to]))

# onset 共有のためのグループ化
MOVES_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_FR[_fr].append((_ov, _to))

MOVES_BY_TO: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_TO[_to].append((_fr, _ov))

N_POS   = len(POS_LIST)   # 33
N_MOVES = len(MOVES)      # 76
N_STEPS = N_POS - 2       # 31

CENTER     = POS2ID[(4, 4)]
INIT_STATE = frozenset(POS2ID.values()) - {CENTER}
GOAL_STATE = frozenset({CENTER})

# offset/change/onset は var_count() 以下の変数番号しか受け付けない．
# cost_bound_ge 等が内部で var_count() を変化させる可能性があるため，
# 最初に power_set(N_POS) を呼んで変数空間を 1..N_POS に確定させる．
# これ以降 N_POS=33 以下の変数番号のみ使うので範囲外エラーが起きない．
_dummy = ZDD.power_set(N_POS)
del _dummy


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
# 遷移演算
# ============================================================

def next_all(F: ZDD) -> ZDD:
    """前向き: 全合法手を適用した盤面族の和集合"""
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
    """逆向き: 全合法手を逆適用した盤面族の和集合"""
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
# Phase 1: 逆向き探索で reachable ZDD を構築
# ============================================================

def build_reachable() -> list[ZDD]:
    """
    ゴールから逆向きに 31 ステップ展開し，reachable リストを返す．

    reachable[0] = ゴール盤面族（1盤面）
    reachable[t] = ゴールから t 手逆向きに到達できる全盤面の族

    reachable[t] はステップ (N_STEPS - t) 後の前向き盤面族と
    & を取ることで，到達不能な盤面を即座に除去できる．
    """
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.2f}s"

    F = ZDD.from_sets([sorted(GOAL_STATE)])
    reachable: list[ZDD] = [F]
    print(f"  逆ステップ  0: ゴール盤面数 = {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = prev_all(F)
        cnt = int(F.exact_count)
        print(f"  逆ステップ {step+1:2d} ({elapsed()})  到達盤面数: {cnt}")
        reachable.append(F)

    # reachable を逆順にする:
    # reachable[t] が「前向きステップ t 後」に対応するよう並べ替え
    # reachable_bwd[0]=ゴール=前向きN_STEPS後，reachable_bwd[N_STEPS]=初期相当
    # → そのままの順で「前向きステップ t 後に使うフィルタ = reachable[N_STEPS - t]」
    print(f"\n  逆向き探索完了 ({elapsed()})")
    print(f"  到達可能初期盤面数（ペグ32個）: "
          f"{int(reachable[N_STEPS].choose(32).exact_count)}")
    return reachable


# ============================================================
# Phase 2: reachable を使った前向き高速探索
# ============================================================

def solve_with_reachable(reachable: list[ZDD]) -> ZDD:
    """
    reachable ZDD を使って前向き探索を行う．

    各ステップで:
      F = next_all(F) & reachable[N_STEPS - (step+1)]

    これにより「今後ゴールに到達できない盤面」を毎ステップ除去する．
    最終的に F はゴール到達可能な盤面のみになる．

    戻り値: 最終ステップの F（= ゴール盤面族）
    """
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.2f}s"

    F = ZDD.from_sets([sorted(INIT_STATE)])
    # 初期盤面を reachable[N_STEPS] と & で検証
    F = F & reachable[N_STEPS]
    print(f"  ステップ  0: 有効初期盤面数 = {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = next_all(F)
        # reachable[N_STEPS - (step+1)] でフィルタリング
        F = F & reachable[N_STEPS - (step + 1)]
        cnt = int(F.exact_count)
        print(f"  ステップ {step+1:2d} ({elapsed()})  有効盤面数: {cnt}")
        if cnt == 0:
            print("  ゴール到達不能")
            break

    print(f"\n  前向き探索完了 ({elapsed()})")
    return F


# ============================================================
# Phase 3: reachable を使った手順追跡
# ============================================================

def trace_with_reachable(reachable: list[ZDD]) -> list[tuple[int,int,int]]:
    """
    reachable ZDD を使って初期盤面からゴールまでの手順を1つ復元する．

    方針:
      ステップ t の盤面 B_t を1つ保持し，
      「B_t に次の手 m を適用した B_{t+1} が reachable[N_STEPS-(t+1)] に含まれるか」
      を確認しながら前向きに1手ずつ決定する．

    reachable[k] との & を使うことで，各ステップで
    「この手を打った後もゴールに到達できるか」を ZDD 演算で高速に判定できる．
    """
    # 初期盤面から出発
    board: frozenset[int] = INIT_STATE

    move_seq: list[tuple[int,int,int]] = []

    for t in range(N_STEPS):
        # 残り (N_STEPS - t - 1) 手後にゴールに到達できる盤面のフィルタ
        future_filter = reachable[N_STEPS - (t + 1)]

        found = False
        for fr, ov, to in MOVES:
            # 合法性チェック
            if fr not in board or ov not in board or to in board:
                continue
            next_board = (board - {fr, ov}) | {to}

            # next_board が future_filter に含まれるか ZDD で確認
            candidate = future_filter
            for pid in sorted(next_board):
                candidate = candidate.onset(pid)
            for pid in POS2ID.values():
                if pid not in next_board:
                    candidate = candidate.offset(pid)

            if int(candidate.exact_count) > 0:
                move_seq.append((fr, ov, to))
                board = frozenset(next_board)
                found = True
                break

        if not found:
            print(f"  ステップ {t+1} で手が見つかりません")
            break

    return move_seq


# ============================================================
# 表示ユーティリティ
# ============================================================

def print_solution(move_seq: list[tuple[int,int,int]]):
    state = set(INIT_STATE)
    show_board(frozenset(state), label="初期盤面")
    for t, (fr, ov, to) in enumerate(move_seq):
        state.remove(fr)
        state.remove(ov)
        state.add(to)
        print(f"\n  手 {t+1:2d}: {ID2POS[fr]} → {ID2POS[ov]} → {ID2POS[to]}")
        show_board(frozenset(state))


def print_reachable_inits(reachable: list[ZDD]):
    """ゴールに到達可能な初期盤面（ペグ32個）を列挙して表示"""
    F32 = reachable[N_STEPS].choose(32)
    num = int(F32.exact_count)
    print(f"\n  到達可能初期盤面数: {num}")
    limit = min(num, 5)
    print(f"  (最初の {limit} 件を表示)")
    for i in range(limit):
        board_vars = F32.unrank(i)
        show_board(frozenset(board_vars), label=f"初期盤面 #{i+1}")
        print()


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  ペグソリティア — reachable ZDD 利用高速解法")
    print("=" * 55)
    show_board(INIT_STATE, label="標準初期盤面（中央空き）")
    print()
    show_board(GOAL_STATE, label="ゴール盤面（中央のみペグ）")
    print()

    # ── Phase 1: 逆向き探索で reachable ZDD を構築 ───────────
    print("【逆向き探索 — reachable ZDD 構築】")
    print("-" * 55)
    reachable = build_reachable()

    # ── 到達可能初期盤面の列挙 ────────────────────────────────
    print("\n【到達可能初期盤面】")
    print("-" * 55)
    print_reachable_inits(reachable)

    # ── Phase 2: reachable を使った前向き高速探索 ────────────
    # 不要．
    # print("【Phase 2】前向き高速探索（reachable フィルタ使用）")
    # print("-" * 55)
    # goal_f = solve_with_reachable(reachable)
    # print(f"  最終ゴール盤面数: {int(goal_f.exact_count)}")

    # ── Phase 3: 手順復元 ─────────────────────────────────────
    print("\n【手順復元（reachable を使って前向きに決定）】")
    print("-" * 55)
    move_seq = trace_with_reachable(reachable)
    print(f"  手順長: {len(move_seq)} 手\n")
    print_solution(move_seq)