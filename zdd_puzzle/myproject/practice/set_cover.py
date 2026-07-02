"""
集合被覆問題 (Set Cover Problem) — ZDD による完全解法
=======================================================

【問題定義】
  Universe U = {1, ..., m} の全要素を被覆する部分集合族 F ⊆ 2^S を求める．
  各部分集合 S_i にはコスト c_i が付いており，総コストを最小化したい．

【アプローチ】
  1. 全実行可能解（被覆を満たす集合族の集まり）を ZDD で表現
  2. ZDD 上のコスト演算で最小コスト解を特定
  3. 解の列挙・統計・近似比の分析

【ZDD の変数割り当て】
  変数 i (1-indexed) = 「部分集合 S_i を選ぶ」という選択変数

"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from kyotodd import ZDD


# ============================================================
# データ構造
# ============================================================

@dataclass
class SetCoverInstance:
    """集合被覆問題のインスタンス"""
    universe: list[int]           # 被覆すべき要素の集合
    subsets: list[list[int]]      # 選択可能な部分集合のリスト (0-indexed)
    costs: list[int]              # costs[i] = S_i のコスト
    name: str = "unnamed"

    @property
    def n(self) -> int:
        """部分集合の数"""
        return len(self.subsets)

    @property
    def m(self) -> int:
        """Universe のサイズ"""
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
    """求解結果"""
    instance: SetCoverInstance
    feasible_zdd: ZDD                        # 全実行可能解の ZDD
    optimal_solutions: list[list[int]]       # 最適解のリスト (変数番号のリスト)
    optimal_cost: int
    num_feasible: int                        # 実行可能解の総数
    solve_time: float
    stats: dict = field(default_factory=dict)


# ============================================================
# インスタンス生成
# ============================================================

def make_instance_manual() -> SetCoverInstance:
    """
    小規模手作りインスタンス
      U = {1,2,3,4,5}
      S1={1,2,3}  c=3
      S2={2,4}    c=2
      S3={3,4,5}  c=3
      S4={1,5}    c=2
      S5={1,2,4}  c=4
    """
    return SetCoverInstance(
        universe=[1, 2, 3, 4, 5],
        subsets=[
            [1, 2, 3],
            [2, 4],
            [3, 4, 5],
            [1, 5],
            [1, 2, 4],
        ],
        costs=[3, 2, 3, 2, 4],
        name="手作り小規模 (n=5, m=5)",
    )


def make_instance_random(n: int, m: int, seed: int = 0) -> SetCoverInstance:
    """
    ランダムインスタンス生成
      n: 部分集合数
      m: Universe サイズ
    少なくとも各要素が1つ以上の部分集合に含まれるよう保証する．
    """
    rng = random.Random(seed)
    universe = list(range(1, m + 1))

    # 各要素を必ず1つの部分集合に割り当て（実行可能解が存在することを保証）
    mandatory = [[] for _ in range(n)]
    for elem in universe:
        mandatory[rng.randrange(n)].append(elem)

    # ランダムに追加要素を付与
    subsets = []
    for i in range(n):
        base = set(mandatory[i])
        extras = rng.sample(universe, k=rng.randint(0, max(0, m // 3)))
        subsets.append(sorted(base | set(extras)))

    costs = [rng.randint(1, 10) for _ in range(n)]
    return SetCoverInstance(
        universe=universe,
        subsets=subsets,
        costs=costs,
        name=f"ランダム (n={n}, m={m}, seed={seed})",
    )


# ============================================================
# ZDD 構築: 全実行可能解の表現
# ============================================================

def build_feasibility_zdd(inst: SetCoverInstance) -> ZDD:
    """
    全実行可能解を ZDD で表現する．

    変数 v = i+1 (1-indexed) が「部分集合 S_i を選ぶ」を意味する．

    手順:
      全部分集合の選び方 (2^n 通り) のうち，Universe を被覆するものだけを残す．
      「被覆しない」= ある要素 e が，選んだどの部分集合にも含まれない．

      NOT_COVER(e) = { 選択 T | e ∈ U かつ ∀i∈T: e ∉ S_i }
                   = power_set({ i | e ∉ S_i }) の補集合から全体を引く

      feasible = power_set(n) - ⋃_e NOT_COVER(e)
               = power_set(n) ∩ ⋂_e COVER(e)

      COVER(e) = 少なくとも1つの e を含む S_i を選ぶ選択の集合
              = ¬ offset_all_vars_not_containing_e
    """
    n = inst.n
    universe_set = set(inst.universe)

    # 全ての選び方
    all_choices = ZDD.power_set(n)

    # 各要素 e に対して「e を被覆しない選択」を除去
    feasible = all_choices
    for e in inst.universe:
        # e を含む部分集合のインデックス (1-indexed 変数)
        covering_vars = [i + 1 for i, s in enumerate(inst.subsets) if e in s]
        if not covering_vars:
            # e を含む部分集合が存在しない → 被覆不可能
            return ZDD.empty

        # e を含まない変数だけを使う選択 = 被覆しない選択
        non_covering_vars = [i + 1 for i in range(n) if (i + 1) not in covering_vars]

        # 被覆しない選択の集合: non_covering_vars の冪集合
        if non_covering_vars:
            not_cover_e = ZDD.power_set_vars(non_covering_vars)
        else:
            # 全変数が e を含む → 何を選んでも被覆できる
            not_cover_e = ZDD.empty

        # 被覆しない選択を除去
        feasible = feasible - not_cover_e

    return feasible


# ============================================================
# コスト重みベクトルの構築
# ============================================================

def make_weight_vector(inst: SetCoverInstance) -> list[int]:
    """
    ZDD の重みベクトルを作成．
    インデックス v (1-indexed) = S_{v-1} のコスト．
    インデックス 0 は未使用 (0 を入れる)．
    """
    weights = [0] * (inst.n + 1)
    for i, c in enumerate(inst.costs):
        weights[i + 1] = c
    return weights


# ============================================================
# メインソルバー
# ============================================================

def solve(inst: SetCoverInstance, verbose: bool = True) -> SetCoverResult:
    """
    ZDD を使って集合被覆問題を完全に解く．
    """
    if verbose:
        inst.display()

    t0 = time.perf_counter()

    # Step 1: 全実行可能解を ZDD で構築
    if verbose:
        print("  [Step 1] 全実行可能解を ZDD で構築中...")
    feasible = build_feasibility_zdd(inst)
    num_feasible = int(feasible.exact_count)
    if verbose:
        print(f"    実行可能解の総数: {num_feasible}")

    if num_feasible == 0:
        raise ValueError("実行可能解が存在しません (Universe を被覆できません)")

    # Step 2: 重みベクトル作成
    weights = make_weight_vector(inst)

    # Step 3: 最小コストを特定
    if verbose:
        print("  [Step 2] 最小コストを探索中...")
    opt_cost = feasible.min_weight(weights)
    if verbose:
        print(f"    最適コスト: {opt_cost}")

    # Step 4: 最適解 ZDD を抽出 (コスト = opt_cost の解だけ)
    if verbose:
        print("  [Step 3] 最適解 ZDD を抽出中...")
    optimal_zdd = feasible.cost_bound_eq(weights, opt_cost)
    num_optimal = int(optimal_zdd.exact_count)
    if verbose:
        print(f"    最適解の数: {num_optimal}")

    # Step 5: 最適解を列挙 (変数番号リスト → 部分集合インデックス)
    optimal_raw = optimal_zdd.enumerate()   # [[v1, v2, ...], ...]

    t1 = time.perf_counter()

    # Step 6: 統計情報
    stats = _compute_stats(inst, feasible, weights, opt_cost, num_feasible)

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
# 統計分析
# ============================================================

def _compute_stats(inst, feasible, weights, opt_cost, num_feasible) -> dict:
    """解の分布・統計を計算する"""
    stats = {}

    # コスト分布: 最小〜最大コストの範囲で集合数を数える
    min_c = feasible.min_weight(weights)
    max_c = feasible.max_weight(weights)
    stats["min_cost"] = min_c
    stats["max_cost"] = max_c

    # サイズ分布 (何個の部分集合を選ぶか)
    profile = feasible.profile()   # profile[k] = k 個選ぶ解の数
    stats["size_profile"] = [(k, cnt) for k, cnt in enumerate(profile) if cnt > 0]

    # 各変数 (部分集合) の選択頻度
    freq = feasible.element_frequency()
    stats["var_frequency"] = {
        f"S{i+1}": freq[i + 1] for i in range(inst.n)
    }

    # グリーディ近似比 (最大頻度ヒューリスティック) の参考値
    greedy_cost = _greedy_cost(inst)
    stats["greedy_cost"] = greedy_cost
    if opt_cost > 0:
        stats["greedy_approx_ratio"] = greedy_cost / opt_cost
    else:
        stats["greedy_approx_ratio"] = 1.0

    return stats


def _greedy_cost(inst: SetCoverInstance) -> int:
    """
    貪欲法 (コスト/カバー数 比が最小の部分集合を順番に選ぶ) でコストを計算．
    厳密最適解との比較用．
    """
    uncovered = set(inst.universe)
    total_cost = 0
    used = [False] * inst.n
    while uncovered:
        best_i, best_ratio = -1, float("inf")
        for i, (s, c) in enumerate(zip(inst.subsets, inst.costs)):
            if used[i]:
                continue
            gain = len(uncovered & set(s))
            if gain == 0:
                continue
            ratio = c / gain
            if ratio < best_ratio:
                best_ratio, best_i = ratio, i
        if best_i == -1:
            break
        used[best_i] = True
        total_cost += inst.costs[best_i]
        uncovered -= set(inst.subsets[best_i])
    return total_cost


# ============================================================
# 結果レポート
# ============================================================

def print_report(result: SetCoverResult):
    inst = result.instance
    n = inst.n

    print(f"\n{'='*60}")
    print(f"  求解結果レポート: {inst.name}")
    print(f"{'='*60}")
    print(f"  求解時間         : {result.solve_time:.4f} 秒")
    print(f"  実行可能解の総数 : {result.num_feasible}")
    print(f"  最適コスト       : {result.optimal_cost}")
    print(f"  最適解の数       : {len(result.optimal_solutions)}")

    print(f"\n--- 最適解 ---")
    for sol_vars in result.optimal_solutions:
        # 変数番号 → 部分集合インデックス (0-indexed)
        chosen_indices = [v - 1 for v in sol_vars]
        chosen_names = [f"S{v}" for v in sol_vars]
        chosen_sets  = [inst.subsets[i] for i in chosen_indices]
        cost_breakdown = [inst.costs[i] for i in chosen_indices]
        total = sum(cost_breakdown)
        print(f"  選択: {chosen_names}")
        print(f"         集合: {chosen_sets}")
        print(f"         コスト内訳: {cost_breakdown}  合計={total}")

    s = result.stats
    print(f"\n--- 統計情報 ---")
    print(f"  コスト範囲       : {s['min_cost']} 〜 {s['max_cost']}")
    print(f"  貪欲法コスト     : {s['greedy_cost']}")
    print(f"  貪欲近似比       : {s['greedy_approx_ratio']:.2f}x")

    print(f"\n  解のサイズ分布 (選択部分集合数 → 解の個数):")
    for k, cnt in s["size_profile"]:
        bar = "█" * min(cnt, 40)
        print(f"    {k:2d} 個選択: {cnt:6d} 解  {bar}")

    print(f"\n  各部分集合の実行可能解中での選択頻度:")
    for name, freq in s["var_frequency"].items():
        idx = int(name[1:]) - 1
        pct = freq / result.num_feasible * 100
        bar = "█" * int(pct / 2)
        print(f"    {name} (コスト={inst.costs[idx]}): {freq:5d} 回 ({pct:5.1f}%)  {bar}")


# ============================================================
# 追加分析: コスト予算別の解数の変化
# ============================================================

def analyze_cost_budget(result: SetCoverResult):
    """
    予算 b を変化させたとき，実行可能解が何件存在するかを調べる．
    CostBoundMemo を使って繰り返し計算を高速化する．
    """
    from kyotodd import CostBoundMemo   # memo クラス (利用可能な場合)

    inst   = result.instance
    f      = result.feasible_zdd
    w      = make_weight_vector(inst)
    min_c  = result.stats["min_cost"]
    max_c  = result.stats["max_cost"]

    print(f"\n--- 予算別の実行可能解数 ---")
    print(f"  (最小コスト={min_c}, 最大コスト={max_c})")

    try:
        memo = CostBoundMemo()
        for b in range(min_c, max_c + 1):
            sub = f.cost_bound_le_with_memo(w, b, memo)
            cnt = int(sub.exact_count)
            bar = "█" * min(cnt, 50)
            print(f"  budget <= {b:3d}: {cnt:6d} 解  {bar}")
    except Exception:
        # CostBoundMemo が利用できない場合は通常版で
        for b in range(min_c, max_c + 1):
            sub = f.cost_bound_le(w, b)
            cnt = int(sub.exact_count)
            bar = "█" * min(cnt, 50)
            print(f"  budget <= {b:3d}: {cnt:6d} 解  {bar}")


# ============================================================
# 追加分析: 軽量解のランキング
# ============================================================

def rank_lightest_solutions(result: SetCoverResult, top_k: int = 5):
    """コストの低い上位 top_k 解をランキング表示"""
    inst = result.instance
    f    = result.feasible_zdd
    w    = make_weight_vector(inst)

    print(f"\n--- コスト昇順ランキング (Top {top_k}) ---")
    rank = 1
    for cost, sol_vars in f.iter_min_weight(w):
        if rank > top_k:
            break
        chosen = [f"S{v}" for v in sol_vars]
        sets   = [inst.subsets[v - 1] for v in sol_vars]
        print(f"  #{rank}  コスト={cost}  選択={chosen}  集合={sets}")
        rank += 1


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":

    # ── ケース 1: 手作り小規模問題 ──────────────────────────
    inst1  = make_instance_manual()
    res1   = solve(inst1, verbose=True)
    print_report(res1)
    analyze_cost_budget(res1)
    rank_lightest_solutions(res1, top_k=5)

    # ── ケース 2: ランダム中規模問題 ────────────────────────
    print("\n\n" + "=" * 60)
    print("  ランダム中規模問題を求解")
    print("=" * 60)
    inst2  = make_instance_random(n=12, m=8, seed=7)
    res2   = solve(inst2, verbose=True)
    print_report(res2)
    analyze_cost_budget(res2)
    rank_lightest_solutions(res2, top_k=8)

    # ── ケース 3: 直列パイプラインのデモ ────────────────────
    # 「予算 B 以内で，かつ必ず S1 を含む解は何件か」
    print("\n\n" + "=" * 60)
    print("  パイプラインデモ: 条件付きフィルタリング")
    print("=" * 60)
    inst3 = make_instance_manual()
    res3  = solve(inst3, verbose=False)
    f     = res3.feasible_zdd
    w     = make_weight_vector(inst3)

    budget = res3.optimal_cost + 1   # 最適 + 1 の予算
    must_include_S1 = f.onset(1)     # S1 を必ず選ぶ解
    within_budget   = f.cost_bound_le(w, budget)
    filtered        = must_include_S1 & within_budget

    print(f"  全実行可能解数               : {int(f.exact_count)}")
    print(f"  S1 を含む解数                : {int(must_include_S1.exact_count)}")
    print(f"  予算 <= {budget} の解数       : {int(within_budget.exact_count)}")
    print(f"  S1 を含み予算 <= {budget} の解: {int(filtered.exact_count)}")
    print(f"  該当解:")
    for sol in filtered.enumerate():
        chosen = [f"S{v}" for v in sol]
        cost   = sum(inst3.costs[v - 1] for v in sol)
        print(f"    {chosen}  コスト={cost}")