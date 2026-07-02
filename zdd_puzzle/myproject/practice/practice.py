import kyotodd

# the unit family {∅} (contains only the empty set)
base = kyotodd.ZDD.single

x = [0 for _ in range (10)]
s = [base for _ in range (10)]

for i in range(10) :
  x[i] = kyotodd.new_var()
  s[i] = base.change(x[i]) # {{xi}} (toggles x1 into the empty set)
print("create", s[0].to_str(), "...", s[9].to_str())

s12 = s[1] + s[2] # create union set
print(s12.to_str())

s3 = s12.change(3) # add an element [3] to all sets in [s12]
s4 = s12.change(4)
print(s3.to_str()) # ?
print(s4.to_str())

s_intersection = s[5] & s[6] # create intersection set
print(s_intersection.to_str())

s_diff = s12 - s[1] # create diff set
print(s_diff.to_str())

print(s4.plain_size)