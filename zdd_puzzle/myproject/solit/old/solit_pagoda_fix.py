r"""
ペグソリティア — Pagoda関数 + reachable ZDD 高速解法
=====================================================

【Pagoda関数による高速化】
  f: V -> R が Pagoda 関数 ⟺
    任意の合法手 (fr, ov, to) に対して f(fr) + f(ov) >= f(to)

  性質: Pagoda値 Σ_{p∈A} f(p) は手を打つたびに非増加
    → t手後の盤面 B について Pagoda(B) >= Pagoda(ゴール) が必要条件
    → この条件を満たさない盤面は到達不能として即座に除去できる

  使用する Pagoda 関数:
    f(r,c) = round(φ^(-|r-4|-|c-4|) × 1000)   (φ = 黄金比)
    全76手で f(fr)+f(ov) >= f(to) を満たすことを検証済み

  ZDD への適用:
    cost_bound_ge(weights, pagoda_goal) で Pagoda 値が
    ゴールを下回る盤面を毎ステップ除去する

【3段階フィルタリング】
  各ステップで以下を順番に適用:
    1. next_all(F)                          前向き遷移
    2. & reachable[N_STEPS-(step+1)]        到達可能性フィルタ
    3. cost_bound_ge(weights, pagoda_goal)  Pagoda フィルタ
"""
# 6/22 pagoda の修正

import math
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

MOVES_BY_FR: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_FR[_fr].append((_ov, _to))

MOVES_BY_TO: dict[int, list[tuple[int,int]]] = _dd(list)
for _fr, _ov, _to in MOVES:
    MOVES_BY_TO[_to].append((_fr, _ov))

N_POS   = len(POS_LIST)   # 33
N_STEPS = N_POS - 2       # 31

CENTER     = POS2ID[(4, 4)]
INIT_STATE = frozenset(POS2ID.values()) - {CENTER}
GOAL_STATE = frozenset({CENTER})


# ============================================================
# Pagoda 関数
# ============================================================

_phi = (1 + math.sqrt(5)) / 2

# f(r,c) = round(phi^(-|r-4|-|c-4|) * 1000)
PAGODA: dict[tuple[int,int], int] = {
    (r, c): round(_phi ** (-(abs(r-4) + abs(c-4))) * 1000)
    for r, c in VALID
}

# ZDD weights ベクトル: インデックス = 位置ID (1-indexed)
# インデックス 0 は未使用，size は max_var+1 以上が必要
_WEIGHTS_LIST = [0] * (N_POS + 2)
for pos, pid in POS2ID.items():
    _WEIGHTS_LIST[pid] = PAGODA[pos]

PAGODA_GOAL = PAGODA[(4, 4)]   # = 1000
PAGODA_INIT = sum(PAGODA[ID2POS[p]] for p in INIT_STATE)

print(f"Pagoda: goal={PAGODA_GOAL}, init={PAGODA_INIT}")

# ============================================================
# 変数空間の初期化
# ============================================================
# offset/change/onset は var_count() 以下の変数番号しか受け付けない．
# cost_bound_ge 等が内部で var_count() を変化させる可能性があるため，
# 最初に power_set(N_POS) を呼んで変数空間を 1..N_POS に確定させる．
# これ以降 N_POS=33 以下の変数番号のみ使うので範囲外エラーが起きない．
_dummy = ZDD.power_set(N_POS)
del _dummy


def pagoda_weights(feasible: ZDD) -> list[int]:
    """
    feasible の support_vars() に合わせた weights を返す．
    cost_bound_ge が要求する「size > top_var」を保証する．
    """
    sv = feasible.support_vars()
    max_var = max(sv) if sv else N_POS
    size = max(N_POS + 2, max_var + 2)
    w = [0] * size
    for pid, val in enumerate(_WEIGHTS_LIST):
        if pid < size:
            w[pid] = val
    return w


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
# Phase 1: 逆向き探索で reachable ZDD を構築
# ============================================================

def build_reachable() -> list[ZDD]:
    """
    reachable[t] = ゴールから t 手逆向きに到達できる全盤面の族
    """
    t0 = time.perf_counter()
    def elapsed(): return f"{time.perf_counter()-t0:.2f}s"

    F = ZDD.from_sets([sorted(GOAL_STATE)])
    reachable: list[ZDD] = [F]
    print(f"  逆ステップ  0: ゴール盤面数 = {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = prev_all(F)
        # Pagoda フィルタ: Pagoda値 >= PAGODA_GOAL の盤面のみ残す
        w = pagoda_weights(F)
        F = F.cost_bound_ge(w, PAGODA_GOAL)
        cnt = int(F.exact_count)
        print(f"  逆ステップ {step+1:2d} ({elapsed()})  到達盤面数: {cnt}")
        reachable.append(F)

    print(f"\n  逆向き探索完了 ({elapsed()})")
    reachable_inits = reachable[N_STEPS].choose(32)
    print(f"  到達可能初期盤面数（ペグ32個）: {int(reachable_inits.exact_count)}")
    return reachable


# ============================================================
# Phase 2: reachable + Pagoda フィルタによる前向き高速探索
# ============================================================

def solve_with_reachable(reachable: list[ZDD]) -> ZDD:
    """
    各ステップで 3 段階フィルタを適用:
      1. next_all          前向き遷移
      2. & reachable[...]  到達可能性フィルタ
      3. cost_bound_ge     Pagoda フィルタ
    """
    t0 = time.perf_counter()
    def elapsed(): return f"{time.perf_counter()-t0:.2f}s"

    F = ZDD.from_sets([sorted(INIT_STATE)])
    F = F & reachable[N_STEPS]
    w = pagoda_weights(F)
    F = F.cost_bound_ge(w, PAGODA_GOAL)
    print(f"  ステップ  0: 有効初期盤面数 = {int(F.exact_count)}")

    for step in range(N_STEPS):
        F = next_all(F)
        F = F & reachable[N_STEPS - (step + 1)]
        w = pagoda_weights(F)
        F = F.cost_bound_ge(w, PAGODA_GOAL)
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
    reachable を使って初期盤面からゴールまでの手順を前向きに1つ復元する．
    各手の選択も Pagoda 条件で候補を絞る．
    """
    board: frozenset[int] = INIT_STATE
    move_seq: list[tuple[int,int,int]] = []

    for t in range(N_STEPS):
        future_filter = reachable[N_STEPS - (t + 1)]
        cur_pagoda = sum(PAGODA[ID2POS[p]] for p in board)

        found = False
        for fr, ov, to in MOVES:
            if fr not in board or ov not in board or to in board:
                continue

            # Pagoda 事前チェック（ZDD演算前に高速除去）
            next_pagoda = cur_pagoda - PAGODA[ID2POS[fr]] - PAGODA[ID2POS[ov]] + PAGODA[ID2POS[to]]
            if next_pagoda < PAGODA_GOAL:
                continue

            next_board = (board - {fr, ov}) | {to}

            # reachable で確認
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
        state.remove(fr); state.remove(ov); state.add(to)
        print(f"\n  手 {t+1:2d}: {ID2POS[fr]} → {ID2POS[ov]} → {ID2POS[to]}")
        show_board(frozenset(state))


def print_reachable_inits(reachable: list[ZDD]):
    F32 = reachable[N_STEPS].choose(32)
    num = int(F32.exact_count)
    print(f"\n  到達可能初期盤面数（ペグ32個）: {num}")
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
    print("=" * 58)
    print("  ペグソリティア — Pagoda + reachable ZDD 高速解法")
    print("=" * 58)
    show_board(INIT_STATE, label="標準初期盤面（中央空き）")
    print()
    show_board(GOAL_STATE, label="ゴール盤面（中央のみペグ）")
    print()

    # Phase 1: 逆向き探索
    print("【Phase 1】逆向き探索 — reachable ZDD 構築（Pagodaフィルタ付き）")
    print("-" * 58)
    reachable = build_reachable()

    # 到達可能初期盤面
    print("\n【到達可能初期盤面】")
    print("-" * 58)
    print_reachable_inits(reachable)

    # Phase 2: 前向き高速探索
    print("【Phase 2】前向き高速探索（reachable + Pagoda 3段階フィルタ）")
    print("-" * 58)
    goal_f = solve_with_reachable(reachable)
    print(f"  最終ゴール盤面数: {int(goal_f.exact_count)}")

    # Phase 3: 手順復元
    print("\n【Phase 3】手順復元")
    print("-" * 58)
    move_seq = trace_with_reachable(reachable)
    print(f"  手順長: {len(move_seq)} 手\n")
    print_solution(move_seq)