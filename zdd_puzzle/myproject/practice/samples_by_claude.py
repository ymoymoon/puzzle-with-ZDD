"""
kyotodd.ZDD API — サンプルプログラム集
======================================
以下のサンプルはすべて `from kyotodd import ZDD` が通る環境を前提とする．
各セクションは独立して実行できる．
"""

from kyotodd import ZDD


# ============================================================
# 1. 基本的な族の構築
# ============================================================
print("=" * 50)
print("1. 基本的な族の構築")
print("=" * 50)

# 定数
empty  = ZDD.empty   # 空族 (集合を一つも含まない)
single = ZDD.single  # 単位族 {∅}

# シングルトン {{v}} を作る
a = ZDD.singleton(1)   # {{1}}
b = ZDD.singleton(2)   # {{2}}
c = ZDD.singleton(3)   # {{3}}

print("a =", a.to_str())   # {1}
print("b =", b.to_str())   # {2}
print("c =", c.to_str())   # {3}

# リストから直接構築
f = ZDD.from_sets([[1, 2], [2, 3], [1, 3]])
print("f =", f.to_str())   # {1,2},{2,3},{1,3}

# n=3 の冪集合 (2^3 = 8 集合)
ps = ZDD.power_set(3)
print("power_set(3) =", ps.to_str())
print("集合数:", ps.exact_count)   # 8


# ============================================================
# 2. 集合演算 (和・積・差・対称差)
# ============================================================
print("\n" + "=" * 50)
print("2. 集合演算")
print("=" * 50)

F = ZDD.from_sets([[1], [1, 2], [2, 3]])
G = ZDD.from_sets([[1, 2], [3], [2, 3]])

print("F =", F.to_str())
print("G =", G.to_str())
print("F + G (和集合)   =", (F + G).to_str())
print("F & G (共通部分) =", (F & G).to_str())
print("F - G (差)       =", (F - G).to_str())
print("F ^ G (対称差)   =", (F ^ G).to_str())


# ============================================================
# 3. 代数演算 (Join / 商)
# ============================================================
print("\n" + "=" * 50)
print("3. 代数演算")
print("=" * 50)

X = ZDD.from_sets([[1], [2]])
Y = ZDD.from_sets([[3]])

# Join: {{1,3},{2,3}}
joined = X * Y
print("X * Y (join) =", joined.to_str())

# 商と余り
P = ZDD.from_sets([[1, 2], [1, 3], [2, 3], [4]])
D = ZDD.singleton(1)
print("P / {{1}} (quotient) =", (P / D).to_str())
print("P % {{1}} (remainder)=", (P % D).to_str())


# ============================================================
# 4. フィルタリング操作
# ============================================================
print("\n" + "=" * 50)
print("4. フィルタリング")
print("=" * 50)

F = ZDD.combination(4, 2)   # {1..4} の 2-元部分集合，全 C(4,2)=6 個
print("C(4,2) =", F.to_str())

# サイズ制限
print("size == 2:", F.choose(2).to_str())   # 全部 (もともと 2 元)
print("size <= 1:", F.size_le(1).to_str())  # 空
print("size >= 2:", F.size_ge(2).to_str())  # 全部

# 変数 1 を含む集合のみ
print("onset(1) =",  F.onset(1).to_str())
# 変数 1 を含まない集合のみ
print("offset(1) =", F.offset(1).to_str())

# {1,2} を含む集合 (上集合)
print("supersets_of([1,2]) =", F.supersets_of([1, 2]).to_str())


# ============================================================
# 5. 極大・極小・最小ヒッティングセット
# ============================================================
print("\n" + "=" * 50)
print("5. 極大・極小・最小ヒッティングセット")
print("=" * 50)

H = ZDD.from_sets([[1], [1, 2], [2, 3], [3]])
print("H =", H.to_str())
print("maximal =", H.maximal().to_str())   # 極大元
print("minimal =", H.minimal().to_str())   # 極小元

# {{1,2},{2,3},{1,3}} の最小ヒッティングセット
cover = ZDD.from_sets([[1, 2], [2, 3], [1, 3]])
print("minhit({1,2},{2,3},{1,3}) =", cover.minhit().to_str())
# 各集合と交わる最小集合は {1,2},{1,3},{2,3},{2} など


# ============================================================
# 6. 数え上げ・列挙・サンプリング
# ============================================================
print("\n" + "=" * 50)
print("6. 数え上げ・列挙・サンプリング")
print("=" * 50)

F = ZDD.power_set(4)
print("power_set(4) の集合数:", F.exact_count)   # 16

# 列挙
sets = F.enumerate()
print("全集合 (最初の 5 個):", sets[:5])

# 一様乱択
sample = F.uniform_sample(seed=42)
print("一様サンプル:", sample)

# k 個のサンプル
k_samples = F.sample_k(3, seed=0)
print("3 個のサンプル:", k_samples.to_str())


# ============================================================
# 7. 重み付き操作とコスト制約
# ============================================================
print("\n" + "=" * 50)
print("7. 重み付き操作とコスト制約")
print("=" * 50)

# 変数 1..4 にコスト (インデックスは変数番号)
weights = [0, 3, 1, 4, 2]   # w[1]=3, w[2]=1, w[3]=4, w[4]=2

F = ZDD.power_set(4)

# コスト合計 <= 4 の集合
cheap = F.cost_bound_le(weights, 4)
print("cost <= 4:", cheap.to_str())

# 最小コスト集合
min_set = F.min_weight_set(weights)
print("最小コスト集合:", min_set, "コスト:", sum(weights[v] for v in min_set))

# 軽い順に 3 集合
lightest = F.get_k_lightest(3, weights)
print("最軽量 3 集合:")
for w, s in lightest.iter_min_weight(weights):
    print(f"  {s}  コスト={w}")


# ============================================================
# 8. ランク・アンランク (ZDD 構造順インデックス)
# ============================================================
print("\n" + "=" * 50)
print("8. ランク / アンランク")
print("=" * 50)

F = ZDD.combination(4, 2)
print("C(4,2) =", F.to_str())

target = [1, 3]
r = F.rank(target)
print(f"rank({target}) =", r)

retrieved = F.unrank(r)
print(f"unrank({r}) =", retrieved)   # [1, 3] が戻る


# ============================================================
# 9. 変数分析 (含意・対称性)
# ============================================================
print("\n" + "=" * 50)
print("9. 変数分析")
print("=" * 50)

# 変数 1 を含む集合は必ず変数 2 も含む，という族
F = ZDD.from_sets([[2], [1, 2], [1, 2, 3], [2, 3]])
print("F =", F.to_str())

print("imply_chk(1, 2):", F.imply_chk(1, 2))   # 1 -> 2 が成り立つか
print("always():", F.always().to_str())          # 全集合に共通する変数

freq = F.element_frequency()
print("element_frequency:", [(v, freq[v]) for v in range(1, 4)])


# ============================================================
# 10. シリアライズ / デシリアライズ
# ============================================================
print("\n" + "=" * 50)
print("10. シリアライズ / デシリアライズ")
print("=" * 50)

F = ZDD.from_sets([[1, 2], [2, 3], [1, 3]])

# 文字列として保存・復元
s = F.export_str()
F2 = ZDD.import_str(s)
print("文字列経由で復元:", F2.to_str())

# バイナリとして保存・復元
b = F.export_binary_str()
F3 = ZDD.import_binary_str(b)
print("バイナリ経由で復元:", F3.to_str())

# ファイルへ書き出し・読み込み
# F.export_file("/tmp/zdd_sample.zdd")
# F4 = ZDD.import_file("/tmp/zdd_sample.zdd")
# print("ファイル経由で復元:", F4.to_str())