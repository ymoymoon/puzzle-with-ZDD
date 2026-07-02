"""
9×9数独ソルバー — ボトムアップ構築型 ZDD
==========================================

【方針】
  ZDD.single（空集合のみ）から出発し，セルごとに join で拡張する．
  join 後に即座に行・列・ボックス制約の違反を supersets_of で除去することで
  ZDD を常に小さく保つ．

  power_set(729) のような巨大な初期ZDDを作らないため
  メモリ効率が根本的に改善される．

【セル処理順とインクリメンタル制約除去】
  セル (r,c) を join した直後に:
    - 行制約: (r, c, d) と (r, c2, d) が共存する集合を除去 (c2 < c)
    - 列制約: (r, c, d) と (r2, c, d) が共存する集合を除去 (r2 < r)
    - ボックス制約: 同ボックス内の既処理セルと (r,c,d) が共存する集合を除去

  これにより join で追加された直後に違反を刈り込み，
  次の join に渡す ZDD を最小限に保つ．

【変数エンコーディング】
  v(r, c, d) = (r-1)*81 + (c-1)*9 + d   (1-indexed, 1..729)
"""

import time
from kyotodd import ZDD

N = 9
BOX_R, BOX_C = 3, 3


# ============================================================
# 変数インデックス
# ============================================================

def var(r: int, c: int, d: int) -> int:
    return (r - 1) * N * N + (c - 1) * N + d


def decode_var(v: int) -> tuple[int, int, int]:
    v -= 1
    d = v % N + 1
    c = (v // N) % N + 1
    r = v // (N * N) + 1
    return r, c, d


def box_id(r: int, c: int) -> tuple[int, int]:
    """セル(r,c)が属するボックスの (br, bc)"""
    return (r - 1) // BOX_R, (c - 1) // BOX_C


# ============================================================
# ZDD 構築（ボトムアップ）
# ============================================================

def build_zdd(hints: list[tuple[int, int, int]]) -> ZDD:
    t0 = time.perf_counter()

    def elapsed():
        return f"{time.perf_counter()-t0:.1f}s"

    hint_set: dict[tuple[int,int], int] = {(r,c): d for r,c,d in hints}

    # 処理済みセルの記録（制約チェック用）
    processed: set[tuple[int,int]] = set()

    # ZDD.single = {∅} からスタート
    f = ZDD.single
    print(f"  ボトムアップ構築開始 (ZDD.single からスタート)")

    for r in range(1, N+1):
        for c in range(1, N+1):

            # ── このセルの選択肢ZDDを作る ──────────────────
            if (r, c) in hint_set:
                d = hint_set[(r, c)]
                cell_zdd = ZDD.from_sets([[var(r, c, d)]])
            else:
                cell_zdd = ZDD.from_sets([
                    [var(r, c, d)] for d in range(1, N+1)
                ])

            # ── join で f に追加 ────────────────────────────
            # join: f の各集合 S と cell_zdd の各集合 T に対して S∪T を作る
            # = 「既存の割り当て」×「このセルの割り当て」の全組み合わせ
            f = f * cell_zdd

            # ── インクリメンタル制約除去 ────────────────────
            # (r,c) を追加した直後に，既処理セルとの違反を除去する

            for d in range(1, N+1):
                v_new = var(r, c, d)

                # 行制約: 同じ行・同じ数字の既処理セルと共存する集合を除去
                for c2 in range(1, c):
                    if (r, c2) in processed:
                        v_old = var(r, c2, d)
                        f = f - f.supersets_of([v_old, v_new])

                # 列制約: 同じ列・同じ数字の既処理セルと共存する集合を除去
                for r2 in range(1, r):
                    if (r2, c) in processed:
                        v_old = var(r2, c, d)
                        f = f - f.supersets_of([v_old, v_new])

                # ボックス制約: 同ボックス内の既処理セルと共存する集合を除去
                br, bc = box_id(r, c)
                for r2 in range(br*BOX_R+1, br*BOX_R+BOX_R+1):
                    for c2 in range(bc*BOX_C+1, bc*BOX_C+BOX_C+1):
                        if (r2, c2) == (r, c):
                            continue
                        if (r2, c2) in processed:
                            v_old = var(r2, c2, d)
                            f = f - f.supersets_of([v_old, v_new])

            processed.add((r, c))

            if c == N:  # 行末にログ出力
                print(f"    行 {r} 完了 ({elapsed()})  "
                      f"ノード数: {f.raw_size}  "
                      f"候補数: {f.exact_count:.2e}")

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
    print("  ┌───────┬───────┬───────┐")
    for r in range(N):
        if r in (3, 6):
            print("  ├───────┼───────┼───────┤")
        row = "  │"
        for c in range(N):
            d = grid[r][c]
            row += f" {d if d else '.'}"
            if (c + 1) % BOX_C == 0:
                row += " │"
        print(row)
    print("  └───────┴───────┴───────┘")


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

def solve(hints, puzzle_name="9×9数独", enumerate_all=False):
    print(f"\n{'='*55}")
    print(f"  {puzzle_name}")
    print(f"{'='*55}")
    print_hint_grid(hints)

    print("\n  ZDD 構築中...")
    t0      = time.perf_counter()
    f       = build_zdd(hints)
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

EASY = [
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

HARDEST = [
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
    solve(EASY,    puzzle_name="9×9数独 (易しい)",    enumerate_all=False)
    solve(HARDEST, puzzle_name="9×9数独 Al Escargot", enumerate_all=False)