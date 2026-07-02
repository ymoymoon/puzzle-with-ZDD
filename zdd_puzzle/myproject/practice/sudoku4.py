"""
4×4数独ソルバー — kyotodd.ZDD による完全解法
=============================================

【盤面】
  4×4 グリッド，数字 1〜4，2×2 ボックス4個

【変数エンコーディング】
  v(r, c, d) = (r-1)*16 + (c-1)*4 + d   (1-indexed, 1..64)

【exactly_one の実装】
  vars_list のうちちょうど1つを含む集合を f から取り出す．

  各 vi について:
    step1: f_ge1.onset0(vi)   → vi を含む集合を選び，vi を除いて返す
    step2: .offset(vj≠vi)     → 他の全変数を除去
    step3: .change(vi)        → 全集合に vi を追加（onset0で除いたものを戻す）

  * ではなく change を使う理由:
    onset0 後は全集合が vi を含まない状態になっている．
    change(vi) は「vi を含まない集合に vi を追加」するので
    正確に vi だけを復元できる．
    join(*) は全組み合わせを取るため複数集合がある場合に
    意図しない集合が生成されてしまう．
"""

import time
from kyotodd import ZDD

N = 4
BOX_R, BOX_C = 2, 2


# ============================================================
# 変数インデックス
# ============================================================

def var(r: int, c: int, d: int) -> int:
    return (r - 1) * N * N + (c - 1) * N + d   # 1..64


def decode_var(v: int) -> tuple[int, int, int]:
    v -= 1
    d = v % N + 1
    c = (v // N) % N + 1
    r = v // (N * N) + 1
    return r, c, d


# ============================================================
# exactly_one
# ============================================================

def exactly_one(vars_list: list[int], f: ZDD) -> ZDD:
    """
    f の中で vars_list のうちちょうど1つを含む集合のみを残す．

    onset0(vi) + offset(vj) + change(vi) の組み合わせで
    「vi だけを含み他は含まない」集合を正確に構築する．
    """
    # 1つも含まない集合を除去 → f_ge1 は1つ以上含む集合
    none_sel = f
    for v in vars_list:
        none_sel = none_sel.offset(v)
    f_ge1 = f - none_sel

    result = ZDD.empty
    for vi in vars_list:
        # vi を含む集合を選び，vi を除いて返す
        branch = f_ge1.onset0(vi)
        # vi 以外の全変数を除去
        for vj in vars_list:
            if vj != vi:
                branch = branch.offset(vj)
        # vi を復元: onset0 で全集合から vi が除かれているので
        # change(vi) で正確に vi を追加できる
        branch = branch.change(vi)
        result = result + branch
    return result


# ============================================================
# ZDD 構築
# ============================================================

def build_zdd(hints: list[tuple[int, int, int]]) -> ZDD:
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.1f}s"

    total_vars = N * N * N   # 64
    print(f"  初期冪集合を構築中 ({total_vars} 変数) ...")
    f = ZDD.power_set(total_vars)
    print(f"    完了 ({elapsed()})")

    # C5: ヒント固定
    print("  C5: ヒント固定 ...")
    hint_set: dict[tuple[int,int], int] = {(r,c): d for r,c,d in hints}
    for (r, c), d in hint_set.items():
        f = f.onset(var(r, c, d))
        for d2 in range(1, N+1):
            if d2 != d:
                f = f.offset(var(r, c, d2))
    print(f"    完了 ({elapsed()})  候補: {f.exact_count:.2e}")

    # C1: 各セルにちょうど1数字
    print(f"  C1: 各セル ({N*N}) ...")
    for r in range(1, N+1):
        for c in range(1, N+1):
            if (r, c) in hint_set:
                continue
            f = exactly_one([var(r, c, d) for d in range(1, N+1)], f)
    print(f"    完了 ({elapsed()})  候補: {f.exact_count:.2e}")

    # C2: 各行×各数字はちょうど1列
    print(f"  C2: 各行×各数字 ({N*N}) ...")
    for r in range(1, N+1):
        for d in range(1, N+1):
            f = exactly_one([var(r, c, d) for c in range(1, N+1)], f)
    print(f"    完了 ({elapsed()})  候補: {f.exact_count:.2e}")

    # C3: 各列×各数字はちょうど1行
    print(f"  C3: 各列×各数字 ({N*N}) ...")
    for c in range(1, N+1):
        for d in range(1, N+1):
            f = exactly_one([var(r, c, d) for r in range(1, N+1)], f)
    print(f"    完了 ({elapsed()})  候補: {f.exact_count:.2e}")

    # C4: 各ボックス×各数字はちょうど1セル
    num_boxes = (N // BOX_R) * (N // BOX_C)
    print(f"  C4: 各ボックス×各数字 ({num_boxes * N}) ...")
    for br in range(N // BOX_R):
        for bc in range(N // BOX_C):
            for d in range(1, N+1):
                box_vars = [
                    var(br*BOX_R+dr+1, bc*BOX_C+dc+1, d)
                    for dr in range(BOX_R) for dc in range(BOX_C)
                ]
                f = exactly_one(box_vars, f)
    print(f"    完了 ({elapsed()})  候補: {f.exact_count:.2e}")

    return f


# ============================================================
# デコード・表示・検証
# ============================================================

def decode_solution(sol_vars: list[int]) -> list[list[int]]:
    grid = [[0]*N for _ in range(N)]
    for v in sol_vars:
        r, c, d = decode_var(v)
        grid[r-1][c-1] = d
    return grid


def print_grid(grid: list[list[int]], title: str = ""):
    if title:
        print(f"\n  {title}")
    print("  ┌─────┬─────┐")
    for r in range(N):
        if r == BOX_R:
            print("  ├─────┼─────┤")
        row = "  │"
        for c in range(N):
            d = grid[r][c]
            row += f" {d if d else '.'}"
            if (c + 1) % BOX_C == 0:
                row += " │"
        print(row)
    print("  └─────┴─────┘")


def print_hint_grid(hints):
    grid = [[0]*N for _ in range(N)]
    for r, c, d in hints:
        grid[r-1][c-1] = d
    print_grid(grid, title="ヒント (入力)")


def verify_solution(grid: list[list[int]]) -> bool:
    digits = set(range(1, N+1))
    for r in range(N):
        if set(grid[r]) != digits:
            return False
    for c in range(N):
        if {grid[r][c] for r in range(N)} != digits:
            return False
    for br in range(N // BOX_R):
        for bc in range(N // BOX_C):
            box = {
                grid[br*BOX_R+dr][bc*BOX_C+dc]
                for dr in range(BOX_R) for dc in range(BOX_C)
            }
            if box != digits:
                return False
    return True


# ============================================================
# メインソルバー
# ============================================================

def solve(hints, puzzle_name="4×4数独", enumerate_all=False):
    print(f"\n{'='*45}")
    print(f"  {puzzle_name}")
    print(f"{'='*45}")
    print_hint_grid(hints)

    print("\n  ZDD 構築中...")
    t0 = time.perf_counter()
    f  = build_zdd(hints)
    elapsed = time.perf_counter() - t0

    num_sol = int(f.exact_count)
    print(f"\n  構築完了: {elapsed:.3f} 秒")
    print(f"  ZDD ノード数: {f.raw_size}")
    print(f"  解の総数    : {num_sol}")

    if num_sol == 0:
        print("  *** 解なし ***")
        return

    grid = decode_solution(f.unrank(0))
    ok   = verify_solution(grid)
    print_grid(grid, title=f"解 #1  ({'✓ 正解' if ok else '✗ 不正'})")

    if enumerate_all and num_sol > 1:
        for i, sol_vars in enumerate(f.enumerate()):
            g = decode_solution(sol_vars)
            print_grid(g, title=f"解 #{i+1}  ({'✓' if verify_solution(g) else '✗'})")
    elif num_sol > 1:
        print(f"  (他に {num_sol-1} 解あり．enumerate_all=True で全解表示)")


# ============================================================
# テストケース
# ============================================================

# ── ケース1: 一意解
#   . 2 | . 4
#   . . | 1 .
#   ----+----
#   . 4 | . .
#   3 . | 2 .
HINTS_UNIQUE = [
    (1,2,2),(1,4,4),
    (2,3,1),
    (3,2,4),
    (4,1,3),(4,3,2),
]

# ── ケース2: ヒント最小（多解）
HINTS_MULTI = [
    (1,1,1),(2,2,2),(3,3,3),(4,4,4),
]

# ── ケース3: ヒントなし（全解）
HINTS_EMPTY = []

if __name__ == "__main__":
    solve(HINTS_UNIQUE, puzzle_name="4×4数独 (一意解)",  enumerate_all=False)
    solve(HINTS_MULTI,  puzzle_name="4×4数独 (多解)",    enumerate_all=True)
    solve(HINTS_EMPTY,  puzzle_name="4×4数独 (ヒントなし・全解)", enumerate_all=False)