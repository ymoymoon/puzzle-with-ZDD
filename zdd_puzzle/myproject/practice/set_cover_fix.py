"""
集合被覆問題 (Set Cover Problem) — kyotodd.ZDD による完全解法
=============================================================

【変数エンコーディング】
  変数 v = i + 1 (1-indexed) が「部分集合 S_i を選ぶ」選択変数．
  n 個の部分集合 → 変数 1..n．

【グローバル変数空間問題への対処】
  ZDD ライブラリはプロセス内でグローバルな変数空間を持つ．
  power_set(n) を呼ぶたびに変数空間が max(n, 現在値) に拡張される．
  一度拡張された空間は縮小できないため，

    inst1 (n=5)  → 変数空間 1..5
    inst2 (n=12) → 変数空間 1..12 に拡張
    inst3 (n=5)  → power_set(5) を呼んでも空間は 1..12 のまま
                   cost_bound_le 等が要求する weights のサイズが
                   max_var+1=13 になるためサイズ 6 の weights ではエラー

  対処: weights を作る際は power_set(n) ではなく，
        ZDD オブジェクトの support_vars() で実際の最大変数番号を取得し，
        weights を max(support_vars())+1 以上の長さで作る．
        これにより変数空間の状態に依存しない安全な実装になる．
"""

import random
import time
from dataclasses import dataclass, field
from itertools import combinations

from kyotodd import ZDD


# ============================================================
# データ構造
# ============================================================

@dataclass
class SetCoverInstance:
    universe: list[int]
    subsets:  list[list[int]]
    costs:    list[int]
    name:     str = "unnamed"

    @property
    def n(self) -> int:
        return len(self.subsets)

    @property
    def m(self) -> int:
        return len(self.universe)

    def display(self):
        print(f"\n{'='*60}")
        print(f"  問題: {self.name}")
        print(f"{'='*60}")
        print(f"  Universe U = {self.universe}  (|U|={self.m})")
        print(f"  部分集合数 n = {self.n}")
        print()
        for i, (s, c) in enumerate(zip(self.subsets, self.costs)):
            print(f"    S_{i+1} = {s}   コスト={c}")
        print()


@dataclass
class SetCoverResult:
    instance:          SetCoverInstance
    feasible_zdd:      ZDD
    optimal_solutions: list[list[int]]
    optimal_cost:      int
    num_feasible:      int
    solve_time:        float
    stats:             dict = field(default_factory=dict)


# ============================================================
# インスタンス生成
# ============================================================

def make_instance_manual() -> SetCoverInstance:
    """
    小規模手作りインスタンス
      U = {1,2,3,4,5}
      S1={1,2,3} c=3,  S2={2,4} c=2,  S3={3,4,5} c=3
      S4={1,5}   c=2,  S5={1,2,4} c=4
    """
    return SetCoverInstance(
        universe=[1, 2, 3, 4, 5],
        subsets=[[1,2,3],[2,4],[3,4,5],[1,5],[1,2,4]],
        costs=[3, 2, 3, 2, 4],
        name="手作り小規模 (n=5, m=5)",
    )


def make_instance_random(n: int, m: int, seed: int = 0) -> SetCoverInstance:
    rng = random.Random(seed)
    universe = list(range(1, m + 1))
    mandatory = [[] for _ in range(n)]
    for elem in universe:
        mandatory[rng.randrange(n)].append(elem)
    subsets = []
    for i in range(n):
        base   = set(mandatory[i])
        extras = rng.sample(universe, k=rng.randint(0, max(0, m // 3)))
        subsets.append(sorted(base | set(extras)))
    costs = [rng.randint(1, 10) for _ in range(n)]
    return SetCoverInstance(
        universe=universe, subsets=subsets, costs=costs,
        name=f"ランダム (n={n}, m={m}, seed={seed})",
    )


# ============================================================
# weights ベクトル生成
# ============================================================

def make_weights(inst: SetCoverInstance, feasible: ZDD) -> list[int]:
    """
    cost_bound_le 等が要求する weights を生成する．

    API 仕様: weights のサイズ > ZDD が使う最大変数番号
    support_vars() でそのZDDが実際に使う変数番号リストを取得し，
    その最大値+1 を下限として weights を作る．
    これにより，グローバル変数空間の拡張に依存しない．
    """
    svars = feasible.support_vars()   # 実際に使われている変数番号のリスト
    max_var = max(svars) if svars else inst.n
    size = max(inst.n + 1, max_var + 1)
    weights = [0] * size
    for i, c in enumerate(inst.costs):
        if i + 1 < size:
            weights[i + 1] = c
    return weights


# ============================================================
# ZDD 構築: 全実行可能解
# ============================================================

def build_feasibility_zdd(inst: SetCoverInstance) -> ZDD:
    """
    全実行可能解を ZDD で構築する．

    全 2^n 通りの選択 (power_set_vars([1..n])) から，
    Universe のある要素を被覆しない選択を差集合で除去する．

    power_set_vars に変数リストを陽に渡すことで，
    グローバル変数空間の状態に関わらず正しい ZDD が得られる．
    """
    n    = inst.n
    vars = list(range(1, n + 1))

    # 変数 1..n の冪集合 (陽に変数リストを指定)
    feasible = ZDD.power_set_vars(vars)

    for e in inst.universe:
        covering_vars     = [i + 1 for i, s in enumerate(inst.subsets) if e in s]
        non_covering_vars = [v for v in vars if v not in covering_vars]

        if not covering_vars:
            return ZDD.empty

        # e を一切被覆しない選択の集合 = non_covering_vars だけの冪集合
        if non_covering_vars:
            not_cover_e = ZDD.power_set_vars(non_covering_vars)
        else:
            not_cover_e = ZDD.empty

        feasible = feasible - not_cover_e

    return feasible


# ============================================================
# メインソルバー
# ============================================================

def solve(inst: SetCoverInstance, verbose: bool = True) -> SetCoverResult:
    if verbose:
        inst.display()

    t0 = time.perf_counter()

    if verbose:
        print("  [Step 1] 全実行可能解を ZDD で構築中...")
    feasible     = build_feasibility_zdd(inst)
    num_feasible = int(feasible.exact_count)
    if verbose:
        print(f"    実行可能解の総数: {num_feasible}")

    if num_feasible == 0:
        raise ValueError("実行可能解が存在しません")

    # weights: support_vars() で ZDD の実際の最大変数番号を取得して生成
    weights = make_weights(inst, feasible)

    if verbose:
        print("  [Step 2] 最小コストを探索中...")
    opt_cost = feasible.min_weight(weights)
    if verbose:
        print(f"    最適コスト: {opt_cost}")

    if verbose:
        print("  [Step 3] 最適解を抽出中...")
    optimal_zdd = feasible.cost_bound_eq(weights, opt_cost)
    num_optimal = int(optimal_zdd.exact_count)
    if verbose:
        print(f"    最適解の数: {num_optimal}")

    optimal_raw = optimal_zdd.enumerate()
    t1          = time.perf_counter()
    stats       = _compute_stats(inst, feasible, weights)

    return SetCoverResult(
        instance=inst,
        feasible_zdd=feasible,
        optimal_solutions=optimal_raw,
        optimal_cost=opt_cost,
        num_feasible=num_feasible,
        solve_time=t1 - t0,
        stats=stats,
    )


# ============================================================
# 統計
# ============================================================

def _compute_stats(inst, feasible, weights) -> dict:
    stats = {}
    stats["min_cost"] = feasible.min_weight(weights)
    stats["max_cost"] = feasible.max_weight(weights)
    profile = feasible.profile()
    stats["size_profile"] = [(k, cnt) for k, cnt in enumerate(profile) if cnt > 0]
    freq = feasible.element_frequency()
    stats["var_frequency"] = {f"S{i+1}": freq[i+1] for i in range(inst.n)}
    stats["greedy_cost"]   = _greedy_cost(inst)
    min_c = stats["min_cost"]
    stats["greedy_approx_ratio"] = stats["greedy_cost"] / min_c if min_c > 0 else 1.0
    return stats


def _greedy_cost(inst: SetCoverInstance) -> int:
    uncovered = set(inst.universe)
    total, used = 0, [False] * inst.n
    while uncovered:
        best_i, best_ratio = -1, float("inf")
        for i, (s, c) in enumerate(zip(inst.subsets, inst.costs)):
            if used[i]:
                continue
            gain = len(uncovered & set(s))
            if gain > 0 and c / gain < best_ratio:
                best_ratio, best_i = c / gain, i
        if best_i == -1:
            break
        used[best_i] = True
        total += inst.costs[best_i]
        uncovered -= set(inst.subsets[best_i])
    return total


# ============================================================
# レポート
# ============================================================

def print_report(result: SetCoverResult):
    inst = result.instance
    print(f"\n{'='*60}")
    print(f"  求解結果: {inst.name}")
    print(f"{'='*60}")
    print(f"  求解時間         : {result.solve_time:.4f} 秒")
    print(f"  実行可能解の総数 : {result.num_feasible}")
    print(f"  最適コスト       : {result.optimal_cost}")
    print(f"  最適解の数       : {len(result.optimal_solutions)}")

    print("\n--- 最適解 ---")
    for sol_vars in result.optimal_solutions:
        chosen = [f"S{v}" for v in sol_vars]
        sets   = [inst.subsets[v-1] for v in sol_vars]
        cost   = sum(inst.costs[v-1] for v in sol_vars)
        print(f"  {chosen}  集合={sets}  合計コスト={cost}")

    s = result.stats
    print(f"\n--- 統計 ---")
    print(f"  コスト範囲    : {s['min_cost']} 〜 {s['max_cost']}")
    print(f"  貪欲法コスト  : {s['greedy_cost']}  (最適比 {s['greedy_approx_ratio']:.2f}x)")
    print(f"\n  サイズ分布 (選択数 → 解の個数):")
    for k, cnt in s["size_profile"]:
        print(f"    {k:2d} 個: {cnt:6d} 解  {'█' * min(cnt, 40)}")
    print(f"\n  各部分集合の選択頻度:")
    for name, freq in s["var_frequency"].items():
        idx = int(name[1:]) - 1
        pct = freq / result.num_feasible * 100
        print(f"    {name} (コスト={inst.costs[idx]}): {freq:5d} 回 ({pct:5.1f}%)  {'█'*int(pct/2)}")


def analyze_cost_budget(result: SetCoverResult):
    f   = result.feasible_zdd
    w   = make_weights(result.instance, f)
    s   = result.stats
    print(f"\n--- 予算別の実行可能解数 ---")
    for b in range(s["min_cost"], s["max_cost"] + 1):
        cnt = int(f.cost_bound_le(w, b).exact_count)
        print(f"  budget <= {b:3d}: {cnt:6d} 解  {'█' * min(cnt, 50)}")


def rank_lightest_solutions(result: SetCoverResult, top_k: int = 5):
    inst = result.instance
    f    = result.feasible_zdd
    w    = make_weights(inst, f)
    print(f"\n--- コスト昇順 Top {top_k} ---")
    rank = 1
    for cost, sol_vars in f.iter_min_weight(w):
        if rank > top_k:
            break
        chosen = [f"S{v}" for v in sol_vars]
        sets   = [inst.subsets[v-1] for v in sol_vars]
        print(f"  #{rank}  コスト={cost}  {chosen}  {sets}")
        rank += 1


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    # ── ケース 1 ──────────────────────────────────────────────
    inst1 = make_instance_manual()
    res1  = solve(inst1, verbose=True)
    print_report(res1)
    analyze_cost_budget(res1)
    rank_lightest_solutions(res1, top_k=5)

    # ── ケース 2 ──────────────────────────────────────────────
    # これ（ solve 関数）を入れると，ケース２自体は動くがケース３が動かなくなる．
    # 原因の予想はこのファイルの冒頭にされている．
    # print("\n\n" + "="*60)
    # inst2 = make_instance_random(n=12, m=8, seed=7)
    # res2  = solve(inst2, verbose=True)
    # print_report(res2)
    # analyze_cost_budget(res2)
    # rank_lightest_solutions(res2, top_k=8)

    # ── ケース 3: inst1 と全く同じ，verbose=True ──────────────
    print("\n\n" + "="*60)
    print("  ケース3: inst1 と同一インスタンスを再実行 (inst2実行後)")
    print("="*60)
    inst3 = make_instance_manual()
    res3  = solve(inst3, verbose=True)
    print_report(res3)

    # ── パイプラインデモ ───────────────────────────────────────
    print("\n\n" + "="*60)
    print("  パイプラインデモ: S1 を含み予算以内の解")
    print("="*60)
    f      = res3.feasible_zdd
    w      = make_weights(inst3, f)
    budget = res3.optimal_cost + 1

    must_S1       = f.onset(1)
    within_budget = f.cost_bound_le(w, budget)
    filtered      = must_S1 & within_budget

    print(f"  全実行可能解数               : {int(f.exact_count)}")
    print(f"  S1 を含む解数                : {int(must_S1.exact_count)}")
    print(f"  予算 <= {budget} の解数       : {int(within_budget.exact_count)}")
    print(f"  S1 を含み予算 <= {budget} の解: {int(filtered.exact_count)}")
    for sol in filtered.enumerate():
        chosen = [f"S{v}" for v in sol]
        cost   = sum(inst3.costs[v-1] for v in sol)
        print(f"    {chosen}  コスト={cost}")