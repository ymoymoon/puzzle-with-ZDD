"""
数独ソルバー — kyotodd.ZDD による完全解法
==========================================

【変数エンコーディング】
  変数番号 v(r, c, d) = (r-1)*81 + (c-1)*9 + d   (1-indexed)
    r ∈ {1..9}: 行
    c ∈ {1..9}: 列
    d ∈ {1..9}: 数字

  合計 9×9×9 = 729 変数．

【制約の種類】
  C1: 各セル (r,c) にはちょうど1つの数字が入る
  C2: 各行 r で，数字 d はちょうど1列に現れる
  C3: 各列 c で，数字 d はちょうど1行に現れる
  C4: 各 3×3 ボックスで，数字 d はちょうど1セルに現れる
  C5: ヒント (確定セル) を固定

  「ちょうど1つ」= 少なくとも1つ(onset) かつ 高々1つ(2変数以上の同時選択を禁止)

【ZDD の使い方】
  全 729 変数の冪集合から，制約を違反する選択を差集合で取り除いていく．
  最終的に残った族の各要素が，数独の完全な解の1つ．
"""

import time
from itertools import combinations


try:
    from kyotodd import ZDD
except ImportError:
    raise ImportError("kyotodd が必要です: pip install kyotodd")


# ============================================================
# 変数インデックス
# ============================================================

def var(r: int, c: int, d: int) -> int:
    """セル(r,c)に数字dを置く変数番号 (1-indexed)"""
    return (r - 1) * 81 + (c - 1) * 9 + d   # 1..729


def decode_var(v: int) -> tuple[int, int, int]:
    """変数番号 → (r, c, d)"""
    v -= 1
    d = v % 9 + 1
    c = (v // 9) % 9 + 1
    r = v // 81 + 1
    return r, c, d


# ============================================================
# 制約ビルダー
# ============================================================

def exactly_one(vars_list: list[int], universe: ZDD) -> ZDD:
    """
    変数集合 vars_list の中からちょうど1つを選ぶ選択のみ残す．

    実装:
      at_least_one : vars_list の全変数を onset で絞り込む前に
                     「少なくとも1つ選ぶ」= 全変数が未選択の組を除外
      at_most_one  : 2つ以上の組み合わせを差集合で除去

    ここでは onset/offset + 差集合で構成する．
    """
    # 少なくとも1つ選ぶ: すべてを選ばない選択(=all offset)を除去
    none_selected = ZDD.power_set_vars(
        [v for v in range(1, 730) if v not in vars_list]
    )
    at_least_one = universe - none_selected

    # 高々1つ: 2つ以上を同時に選ぶ選択を除去
    result = at_least_one
    for v1, v2 in combinations(vars_list, 2):
        # v1 と v2 を両方含む選択を除去
        both = result.onset(v1).onset(v2)
        # onset は v1,v2 を含んだまま返すので，それをそのまま差集合
        result = result - both

    return result


def build_sudoku_zdd(hints: list[tuple[int, int, int]]) -> ZDD:
    """
    数独の全制約を満たす選択の族を ZDD で構築する．

    hints: [(r, c, d), ...] — 確定セルのリスト
    戻り値: 解を表す ZDD
    """

    print("  初期冪集合を構築中 (729 変数) ...")
    t0 = time.perf_counter()

    # 全変数の冪集合からスタート
    universe = ZDD.power_set(729)
    f = universe
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)")

    # ── C5: ヒント固定 ────────────────────────────────────────
    # ヒントのある変数は必ず選択する (onset)
    # 同じセルの他の数字は選ばない (offset)
    print("  C5: ヒント固定 ...")
    hint_set = {}
    for r, c, d in hints:
        hint_set[(r, c)] = d

    for (r, c), d in hint_set.items():
        # (r,c) に d を置く変数を固定
        f = f.onset(var(r, c, d))
        # (r,c) の他の数字変数を排除
        for d2 in range(1, 10):
            if d2 != d:
                f = f.offset(var(r, c, d2))
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)  解候補: {f.exact_count:.2e}")

    # ── C1: 各セルにちょうど1つの数字 ────────────────────────
    print("  C1: 各セル (81 セル) にちょうど1数字 ...")
    for r in range(1, 10):
        for c in range(1, 10):
            if (r, c) in hint_set:
                continue  # ヒントセルは C5 で処理済み
            cell_vars = [var(r, c, d) for d in range(1, 10)]
            f = exactly_one(cell_vars, f)
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)  解候補: {f.exact_count:.2e}")

    # ── C2: 各行で各数字はちょうど1列 ───────────────────────
    print("  C2: 各行×各数字 (81 制約) ...")
    for r in range(1, 10):
        for d in range(1, 10):
            row_vars = [var(r, c, d) for c in range(1, 10)]
            f = exactly_one(row_vars, f)
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)  解候補: {f.exact_count:.2e}")

    # ── C3: 各列で各数字はちょうど1行 ───────────────────────
    print("  C3: 各列×各数字 (81 制約) ...")
    for c in range(1, 10):
        for d in range(1, 10):
            col_vars = [var(r, c, d) for r in range(1, 10)]
            f = exactly_one(col_vars, f)
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)  解候補: {f.exact_count:.2e}")

    # ── C4: 各 3×3 ボックスで各数字はちょうど1セル ──────────
    print("  C4: 各ボックス×各数字 (81 制約) ...")
    for br in range(3):           # ボックス行 0,1,2
        for bc in range(3):       # ボックス列 0,1,2
            for d in range(1, 10):
                box_vars = [
                    var(br * 3 + dr + 1, bc * 3 + dc + 1, d)
                    for dr in range(3)
                    for dc in range(3)
                ]
                f = exactly_one(box_vars, f)
    print(f"    完了 ({time.perf_counter()-t0:.2f}s)  解候補: {f.exact_count:.2e}")

    return f


# ============================================================
# 解のデコード・表示
# ============================================================

def decode_solution(sol_vars: list[int]) -> list[list[int]]:
    """変数番号のリスト → 9×9 グリッド"""
    grid = [[0] * 9 for _ in range(9)]
    for v in sol_vars:
        r, c, d = decode_var(v)
        grid[r - 1][c - 1] = d
    return grid


def print_grid(grid: list[list[int]], title: str = ""):
    if title:
        print(f"\n  {title}")
    print("  ┌───────┬───────┬───────┐")
    for r in range(9):
        if r in (3, 6):
            print("  ├───────┼───────┼───────┤")
        row_str = "  │"
        for c in range(9):
            d = grid[r][c]
            row_str += f" {d if d else '.'}"
            if c in (2, 5):
                row_str += " │"
        row_str += " │"
        print(row_str)
    print("  └───────┴───────┴───────┘")


def print_hint_grid(hints: list[tuple[int, int, int]]):
    grid = [[0] * 9 for _ in range(9)]
    for r, c, d in hints:
        grid[r - 1][c - 1] = d
    print_grid(grid, title="ヒント (入力)")


# ============================================================
# 解の検証
# ============================================================

def verify_solution(grid: list[list[int]]) -> bool:
    """数独の解が正しいか確認する"""
    digits = set(range(1, 10))
    # 行
    for r in range(9):
        if set(grid[r]) != digits:
            return False
    # 列
    for c in range(9):
        if {grid[r][c] for r in range(9)} != digits:
            return False
    # ボックス
    for br in range(3):
        for bc in range(3):
            box = {grid[br*3+dr][bc*3+dc] for dr in range(3) for dc in range(3)}
            if box != digits:
                return False
    return True


# ============================================================
# メインソルバー
# ============================================================

def solve_sudoku(
    hints: list[tuple[int, int, int]],
    puzzle_name: str = "数独",
    enumerate_all: bool = False,
) -> None:
    """
    数独を ZDD で解く．

    hints: [(r, c, d), ...]
    enumerate_all: True にすると全解を列挙（解が少ない場合向け）
    """
    print(f"\n{'='*60}")
    print(f"  {puzzle_name}")
    print(f"{'='*60}")
    print_hint_grid(hints)

    print("\n  ZDD 構築中...")
    t_start = time.perf_counter()
    f = build_sudoku_zdd(hints)
    t_build = time.perf_counter() - t_start

    num_solutions = int(f.exact_count)
    print(f"\n  構築完了: {t_build:.3f} 秒")
    print(f"  ZDD ノード数: {f.raw_size}")
    print(f"  解の総数: {num_solutions}")

    if num_solutions == 0:
        print("  *** 解なし ***")
        return

    # 最初の解を表示
    first_sol_vars = f.unrank(0)
    grid = decode_solution(first_sol_vars)
    ok = verify_solution(grid)
    print_grid(grid, title=f"解 #1  (検証: {'✓ 正解' if ok else '✗ 不正'})")

    if enumerate_all and num_solutions > 1:
        print(f"\n  全 {num_solutions} 解を列挙:")
        for i, sol_vars in enumerate(f.enumerate()):
            g = decode_solution(sol_vars)
            v = verify_solution(g)
            print_grid(g, title=f"解 #{i+1}  ({'✓' if v else '✗'})")

    elif num_solutions > 1:
        print(f"\n  (残り {num_solutions-1} 解あり．enumerate_all=True で全解表示)")


# ============================================================
# テストケース
# ============================================================

# ── ケース 1: 易しめの数独 ──────────────────────────────────
# 出典: 一般的なサンプル問題
EASY_HINTS = [
    (1,1,5),(1,2,3),(1,5,7),
    (2,1,6),(2,4,1),(2,5,9),(2,6,5),
    (3,2,9),(3,3,8),(3,8,6),
    (4,1,8),(4,5,6),(4,9,3),
    (5,1,4),(5,4,8),(5,6,3),(5,9,1),
    (6,1,7),(6,5,2),(6,9,6),
    (7,2,6),(7,7,2),(7,8,8),
    (8,4,4),(8,5,1),(8,6,9),(8,9,5),
    (9,5,8),(9,8,7),(9,9,9),
]

# ── ケース 2: 解が一意でない問題 (ヒントが少ない) ───────────
MULTI_SOL_HINTS = [
    (1,1,1),
    (2,2,2),
    (3,3,3),
    (4,4,4),
    (5,5,5),
]

# ── ケース 3: 世界最難問とされる Al Escargot ─────────────────
HARDEST_HINTS = [
    (1,1,1),(1,2,2),(1,4,3),(1,6,4),
    (2,1,6),(2,3,4),(2,5,7),(2,8,2),
    (3,3,5),(3,5,9),(3,6,8),
    (4,2,4),(4,4,6),(4,8,8),(4,9,1),
    (5,1,9),(5,4,2),(5,6,3),(5,9,5),
    (6,1,5),(6,2,6),(6,6,7),(6,8,4),
    (7,4,8),(7,5,1),(7,7,3),
    (8,2,7),(8,5,2),(8,7,6),(8,9,4),
    (9,4,4),(9,6,5),(9,8,7),(9,9,9),
]


if __name__ == "__main__":

    # 易しい問題: 解を求め，検証
    solve_sudoku(EASY_HINTS, puzzle_name="易しい数独", enumerate_all=False)

    # 解が複数ある問題: 全解を列挙
    solve_sudoku(MULTI_SOL_HINTS, puzzle_name="ヒントが少ない数独 (多解)", enumerate_all=True)

    # 最難問: ZDD の威力を確認
    solve_sudoku(HARDEST_HINTS, puzzle_name="Al Escargot (最難問)", enumerate_all=False)