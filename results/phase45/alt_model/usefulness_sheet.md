# Usefulness check: are the retrieved examples comparable solutions to the same task?

Rate each side 0 = not comparable, 1 = partly, 2 = clearly comparable. Fill `usefulness_sheet.csv`.


## Query 1  (true: human, problem p02721, found p02721 OK)
```python
n, k, c = list(map(int, input().split()))
s = input()

set1 = set()
set2 = set()
index = 0
cnt = 0
while index < c and cnt < c:
    if s[index] == 'o':
        set1.add(index + 1)
        index += c + 1
        cnt += 1
        continue
    index += 1

index = n - 1
cnt = 0
while index < c and cnt < c:
    if s[index] == 'o':
        set2.add(index + 1)
```

**H1** aigcodeset_002802 sim=0.828 same problem human/human
```python

n, k, c = map(int, input().split())
s = input()
l = [0] * k
r = [0] * k
p = 0
# for i in range(n):
i = 0
while i < n:
    if s[i] == "o":
        l[p] = i
        p += 1
        if (p >= k):
            break
        i += c
    i += 1
p = k-1
# for i in range(n - 1, -1, -1):
i = n - 1
while i >= 0:
    if s[i] == "o":
        r[p] = i
        p -= 1
        if (p < 0):
            break
```

**H2** aigcodeset_002822 sim=0.8235 same problem human/human
```python
N, K, C = map(int,input().split())
S = list(input())

A = list()
B = list()

i = 0
num = 0
while num < K:
  if S[i] == "o":
    A.append(i)
    num += 1
    i += C
  i += 1
i = len(S)-1
num = 0
while num < K:
  if S[i] == "o":
    B.append(i)
    num+=1
    i -= C
  i -= 1
B.sort()

for i,j in zip(A,B):
```

**H3** aigcodeset_003286 sim=0.456 same problem human/human
```python
nkc=input().split()
K=int(nkc[1])
C=int(nkc[2])
s=input()
workday_list=[]
for i,youso in enumerate(s) :
  if youso=='o' :
    workday_list.append(i+1)
    

s=1

current_num=0
current_num1=1
list1=[]
list1.append(workday_list[0])
while(s<K) :
  
  if current_num1>=len(workday_list):
    break
  elif workday_list[current_num1]-workday_list[current_num]>C :
    current_num=current_num1
    list1.append(workday_list[current_num1])
    s+=1
  else :
```

**A1** aigcodeset_006413 sim=0.6349 same problem gemini/fix_runtime_error
```python
from functools import reduce
from itertools import combinations

n, k, c = map(int, input().split())
s = input()

work = range(1, n + 1)
not_working = []
for i, v in enumerate(s, start=1):
    if v == "o":
        continue
    not_working.append(i)

working = set(work) - set(not_working)


def is_consistent(t):
    u = t[1:]
    for i, v in zip(t, u):
        if v - i <= c:
            return False
    return True


l = list(filter(is_consistent, combinations(working, k)))
```

**A2** aigcodeset_005554 sim=0.6047 same problem codestral/fix_runtime_error
```python
from itertools import combinations

n, k, c = map(int, input().split())
s = input()

work = range(1, n + 1)
not_working = []
for i, v in enumerate(s, start=1):
    if v == "o":
        continue
    not_working.append(i)

working = set(work) - set(not_working)

def is_consistent(t):
    u = t[1:]
    for i, v in zip(t, u):
        if v - i <= c:
            return False
    return True

l = list(filter(is_consistent, combinations(working, k)))
if l:
    result = set.intersection(*map(set, l))
    for i in sorted(result):
```

**A3** aigcodeset_006270 sim=0.4837 same problem codestral/from_scratch
```python
N, K, C = map(int, input().split())
S = input()

work_days = []
i = 0
while i < N:
    if S[i] == 'o':
        work_days.append(i + 1)
        i += C + 1
    else:
        i += 1

    if len(work_days) == K:
        break

for day in work_days:
    print(day)
```

## Query 2  (true: human, problem p03274, found p03274 OK)
```python
N, K = map(int, input().split())
X = list(map(int, input().split()))

lis = []
for i in range(N-K+1):
    left = X[i]
    right = X[i+K-1]
    res = right - left
    abs_l, abs_r = abs(left), abs(right)
    if abs_l > abs_r: 
       res += abs_r
    else:
       res += abs_l
    lis.append(res)
print(min(lis))

```

**H1** aigcodeset_000186 sim=0.8938 same problem human/human
```python
N, K = map(int, input().split())
Xn = list(map(int, input().split()))

Ln = [0] * N
for i in range(N-1):
    Ln[i+1] = abs(Xn[i+1] - Xn[i]) + Ln[i]

Rn = [0] * N
for i in range(1, N)[::-1]:
    Rn[i-1] = abs(Xn[i] - Xn[i-1]) + Rn[i]

ans = float("inf")
for i in range(N-K+1):
    ans = min(ans, abs(Xn[i]) + Ln[i+K-1] - Ln[i])

for i in range(K-1, N)[::-1]:
    ans = min(ans, abs(Xn[i]) + Rn[i-K+1] - Rn[i])

print(ans)
```

**H2** aigcodeset_002751 sim=0.8091 same problem human/human
```python
N,K=map(int,input().split())
S=list(map(int,input().split()))
ans=float("inf")
if N==1:
	ans=abs(S[0])
for i in range(N-K+1):
	L=S[i]
	R=S[i+K-1]
	ans=min(abs(L)+abs(R-L),abs(R)+abs(L-R),ans)
print(ans)
```

**H3** aigcodeset_004630 sim=0.644 same problem human/human
```python
MAX = 300000001

N, K = map(int, input().split())
xn = [int(i) for i in input().split()]


def main():
    if 0 <= xn[0]:
        print(xn[K - 1])
        return

    if xn[N - 1] < 0:
        print(abs(xn[N - K]))
        return

    ans = MAX

    for i in range(N - K + 1):
        ans = min(ans,
                  2 * abs(xn[i]) + abs(xn[i + K - 1]),
                  abs(xn[i]) + 2 * abs(xn[i + K - 1]))

    print(ans)
    return

```

**A1** aigcodeset_004902 sim=0.899 same problem codestral/fix_runtime_error
```python
N, K = map(int, input().split())
X = list(map(int, input().split()))

if X[0] < 0:
    fl = 0
    for i in range(N):
        if X[i] >= 0:
            fl = i
            break

    _mi = X[:fl]
    _mi = _mi[::-1]
    for i in range(len(_mi)):
        _mi[i] = abs(_mi[i])

    _pi = X[fl:]

    ans = 10000000000000000

    l_mi = len(_mi)
    l_pi = len(_pi)

    for i in range(l_mi):
        if K - (i+1) <= l_pi and K - (i+1) > -1:
            ans = min(ans, _mi[i]*2+_pi[(K-1)-(i+1)])
```

**A2** aigcodeset_005544 sim=0.7911 same problem gemini/fix_runtime_error
```python
N, K = map(int, input().split())
X = list(map(int, input().split()))

# X.sort()

# print(X)

mi = 0
pl = 0

_mi = []
_pi = []

fl = 0
if X[0] < 0:
    for i in range(N):
        if X[i] >= 0:
            fl = i
            break

    # print(fl)

    _mi = X[:fl]
    _mi = _mi[::-1]

```

**A3** aigcodeset_005769 sim=0.7629 same problem codellama/fix_runtime_error
```python
# input
N, K = map(int, input().split())
x = list(map(int, input().split()))

# check
mov = [
    sum(
        [
            abs(x[j])
            if j == i
            else
            abs(x[j] - x[j - 1])
            for j in range(i, i + K)
        ]
    )
    for i in range(0, N - K)
] + [
    sum(
        [
            abs(x[j])
            if j == i
            else
            abs(x[j + 1] - x[j])
            for j in range(i, i - K, -1)
        ]
```

## Query 3  (true: AI, problem p02777, found p02957 WRONG)
```python
  def solve(S, T, A, B, U):
    if U == S:
        return A - 1, B
    else:
        return A, B - 1
```

**H1** aigcodeset_003492 sim=0.5291 same problem human/human
```python
A, B = list(map(int, input().split()))
K = (A + B) / 2
if K == int(K):
    print(int(K))
else:
    print("IMPOSSIBLE")
```

**H2** aigcodeset_003587 sim=0.495 same problem human/human
```python
A, B = map(int, input().split())

if (A+B)%2 == 0:
    print((A+B)//2)
else:
    print("IMPOSSIBLE")
```

**H3** aigcodeset_001638 sim=0.4856 same problem human/human
```python
A, B = map(int, input().split())
avg = sum([A, B]) // 2
if abs(A - avg) == abs(B - avg):
  print(avg)
else:
  print("IMPOSSIBLE") 
```

**A1** aigcodeset_006672 sim=0.9087 same problem codellama/from_scratch
```python
  def solve(A, B):
    if A == B:
        return "IMPOSSIBLE"
    if A < B:
        smaller = A
        larger = B
    else:
        smaller = B
        larger = A
    for i in range(smaller, larger+1):
        if abs(A-i) == abs(B-i):
            return i
    return "IMPOSSIBLE"
```

**A2** aigcodeset_005819 sim=0.5023 same problem codellama/fix_wrong_answer
```python
A = list(map(int, input().split()))
if (A[0] + A[1]) // 2 == 0:
    print(abs(A[0] - (A[0] + A[1]) // 2))
else:
    print('IMPOSSIBLE')
```

**A3** aigcodeset_005124 sim=0.4948 same problem codestral/fix_wrong_answer
```python
A, B = map(int, input().split())

if (A + B) % 2 == 0:
    K = (A + B) // 2
else:
    K = (A + B) // 2 + 1

print(K)
```

## Query 4  (true: human, problem p02769, found p03291 WRONG)
```python
MAX_NUM = 10**6 + 1
MOD = 10**9+7

fac  = [0 for _ in range(MAX_NUM)]
finv = [0 for _ in range(MAX_NUM)]
inv  = [0 for _ in range(MAX_NUM)]

fac[0]  = fac[1] = 1
finv[0] = finv[1] = 1
inv[1] = 1

for i in range(2,MAX_NUM):
    fac[i] = fac[i-1] * i % MOD
    inv[i] = MOD - inv[MOD%i] * (MOD // i) % MOD
    finv[i] = finv[i-1] * inv[i] % MOD

def combinations(n,k):
    if n < k:
        return 0
    if n < 0 or k < 0:
```

**H1** aigcodeset_001181 sim=0.6307 same problem human/human
```python
S = input()
N = len(S)
mod = 10**9+7

A_cum = [0] * N
C_cum = [0] * N
q_sum_left = [0] * N
q_sum_right = [0] * N
for i in range(N - 1):
    s = S[i]
    if s == 'A':
        A_cum[i + 1] = A_cum[i] + 1
    else:
        A_cum[i + 1] = A_cum[i]
    if s == '?':
        q_sum_left[i + 1] = q_sum_left[i] + 1
    else:
        q_sum_left[i + 1] = q_sum_left[i]
for i in range(N - 1, 0, -1):
    s = S[i]
    if s == 'C':
        C_cum[i - 1] = C_cum[i] + 1
    else:
        C_cum[i - 1] = C_cum[i]
    if s == '?':
```

**H2** aigcodeset_003644 sim=0.3682 same problem human/human
```python
S = input()
def ABC(S):
    N = len(S)
    dp = [[0 for j in range(4)] for i in range(N+1)]
    model = 'ABC'
    p = 10**9 + 7
    for i in range(N, -1, -1):
        for j in range(3,-1,-1):
            if i == N and j == 3:
                dp[i][j] = 1
            elif i == N and j < 3:
                dp[i][j] = 0
            elif i < N and j == 3:
                if S[i] == '?':
                    dp[i][j] = 3 * dp[i+1][j] % p
                else:
                    dp[i][j] = dp[i+1][j] % p
            else:
                if S[i] == '?':
                    m1 = 3
                els
```

**H3** aigcodeset_002241 sim=0.3649 same problem human/human
```python
from sys import exit, setrecursionlimit, stderr
from functools import reduce
from itertools import *
from collections import defaultdict
from bisect import bisect
 
def read():
  return int(input())
 
def reads():
  return [int(x) for x in input().split()]

P = 10**9 + 7

S = input()
N = len(S)

psumA = [0] + list(accumulate(c == 'A' for c in S))
psumC = [0] + list(accumulate(c == 'C' for c in S))
psumX = [0] + list(accumulate(c == '?' for c in S))
Q = psumX[N]

ans = 0
for i in range(1, N-1):
  if S[i] == 'B':
```

**A1** aigcodeset_007312 sim=0.7777 same problem gemini/from_scratch
```python
MOD = 10**9 + 7
s = input()
n = len(s)
a_cnt = [0] * (n + 1)
b_cnt = [0] * (n + 1)
c_cnt = [0] * (n + 1)
for i in range(n):
    if s[i] == 'A':
        a_cnt[i + 1] = a_cnt[i] + 1
    else:
        a_cnt[i + 1] = a_cnt[i]
    if s[i] == 'B':
        b_cnt[i + 1] = b_cnt[i] + 1
    else:
        b_cnt[i + 1] = b_cnt[i]
    if s[i] == 'C':
        c_cnt[i + 1] = c_cnt[i] + 1
    else:
        c_cnt[i + 1] = c_cnt[i]
ans = 0
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n + 1):
            if s[i] == '?' and s[j] == '?' and s[k - 1] == '?':
                ans += 
```

**A2** aigcodeset_005613 sim=0.5795 same problem codestral/fix_runtime_error
```python
S = input()
mod = 10 ** 9 + 7

def DP(i, j, S):
    if i == len(S):
        return 1 if j == 3 else 0

    if S[i] == "?":
        base = 3 * DP(i+1, j, S)
        if j < 3:
            base += DP(i+1, j+1, S)
    elif j < 3 and S[i] == "ABC"[j]:
        base = DP(i+1, j, S) + DP(i+1, j+1, S)
    else:
        base = DP(i+1, j, S)

    return base % mod

print(DP(0, 0, S))
```

**A3** aigcodeset_005143 sim=0.5792 same problem codestral/fix_wrong_answer
```python
a = dict()
a['A'] = []
a['B'] = []
a['C'] = []
a['?'] = []

string = input()

for i in range(0, len(string)):
    temp = string[i]
    a[temp].append(i)

result = 0
dp = [0] * 4
MOD = 10**9 + 7
dp[0] = 1

for c in string:
    if c == '?':
        for i in range(3, 0, -1):
            dp[i] = (dp[i]*3 + dp[i-1]) % MOD
        dp[0] = (dp[0]*3) % MOD
    elif c == 'A':
        dp[1] = (dp[1] + dp[0]) % MOD
    elif c == 'B':
```

## Query 5  (true: AI, problem p02793, found p03700 WRONG)
```python
import sys

def gcd(a, b):
    a, b = max(a, b), min(a, b)
    while a % b > 0:
        a, b = b, a % b
    return b 

def solve():
    input = sys.stdin.readline
    N = int(input())
    A = [int(a) for a in input().split()]
    mod = 10**9 + 7
    gcdA = A[0]
    lcmA = A[0]
    for i in range(1, N):
        gcdA = gcd(lcmA, A[i])
        lcmA *=  (A[i] * pow(gcdA, mod - 2, mod)) % mod
        lcmA %= mod
    Ans = 0
```

**H1** aigcodeset_000969 sim=0.8182 same problem human/human
```python
import sys
rl = sys.stdin.readline


def solve():
    N, A, B = map(int, rl().split())
    h = [int(rl()) for _ in range(N)]
    ad_d = A - B
    
    def calc(t):
        cnt = 0
        for i in range(N):
            d = h[i] - B * t
            if 0 < d:
                cnt += d // ad_d if d % ad_d == 0 else d // ad_d + 1
        return cnt <= t
    
    ok, ng = 10 ** 9, 0
    while 1 < ok - ng:
        mid = (ok + ng) // 2
        if calc(mid):
            ok = mid
        else:
            ng = mid
    print(ok)
```

**H2** aigcodeset_002022 sim=0.5544 same problem human/human
```python
N, A, B = map(int, input().split())
hs = [int(input()) for i in range(N)]

INF = int(1e9)
l, m, r = 0, INF//2, INF + 1

def calc(n):
    global N, A, B
    d = A - B
    remain = n
    for h in hs:
        tmp = h - B * n
        if tmp > 0:
            remain -= (tmp + d - 1) // d
    return remain >= 0

while r - l > 1:
    if calc(m):
        r = m
    else:
        l = m
    m = (r + l + 1) // 2

print(m)
```

**H3** aigcodeset_002251 sim=0.5442 same problem human/human
```python
import sys
input = sys.stdin.readline

n,a,b= map(int, input().split())
h= [int(input()) for i in range(n)]


# めぐる式二部探索、mid=攻撃回数
l=-1
r=pow(10,10)
while r-l>1:
    # rは常にOK、lは常にNG。
    mid=l+(r-l)//2
    # 残りの攻撃回数を記録
    x=mid
    # なんらかの判定
    for i in range(n):
        # 爆風で倒せる敵はスルー
        if mid*b>=h[i]:
            continue
        else:
            # 爆風で削りきれないモンスターを爆弾で削る
            y=-(-(h[i]-(mid*b))//(a-b))
            x-=y
            if x<0:
```

**A1** aigcodeset_006486 sim=0.5796 same problem gemini/fix_runtime_error
```python
import numpy as np

N, A, B = map(int, input().split())
H = np.zeros(shape=N, dtype='int64')
for i in range(N):
    h = int(input())
    H[i] = h


def calc(num, a):
    a -= num * B
    a = (a - 1) // (A - B) + 1
    return sum(a[a > 0]) <= num


start = 0
end = 10 ** 10

while end - start > 1:
    mid = (start + end) // 2
    if calc(mid, H.copy()):
        end = mid
    else:
        start = mid

```

**A2** aigcodeset_006561 sim=0.5189 same problem codestral/fix_wrong_answer
```python
import numpy as np
import math

def search(min_, max_, A, B, arr):
    if min_ == max_:
        return min_
    mid = (max_ + min_) // 2

    tmp = arr - mid * B

    if len(tmp) == 0:
        return search(min_, mid, A, B, arr)

    tmp = np.maximum(tmp, 0)
    tmp = np.ceil(tmp / (A - B))
    sum_ = np.sum(tmp)
    if sum_ <= mid:
        return search(min_, mid, A, B, arr)
    else:
        return search(mid + 1, max_, A, B, arr)

N, A, B = list(map(int, input().split()))
arr = []

for i in range(N):
```

**A3** aigcodeset_006968 sim=0.4763 same problem gemini/from_scratch
```python
n, a, b = map(int, input().split())
h = []
for _ in range(n):
  h.append(int(input()))

h.sort()
ans = 0
i = 0
while i < n:
  if h[i] <= 0:
    i += 1
    continue
  ans += 1
  h[i] -= a
  for j in range(i + 1, n):
    h[j] -= b
    if h[j] <= 0:
      i += 1
      break
print(ans)
```

## Query 6  (true: human, problem p03061, found p03089 WRONG)
```python
N=int(input())
A=list(map(int,input().split()))

def euclid(a,b):
    #print(a,b)
    if b==0:
        return a
    else:
        return euclid(b, a%b)

m=1
for n in range(N):
# 2重のforループ？
# 最大公約数の最大を出力?
    if n==N-1:
        l=euclid(max(A[n],A[0]), min(A[n],A[0]))
    else:
        l=euclid(max(A[n],A[n+1]), min(A[n],A[n+1]))
    if m<l:
        m=l
```

**H1** aigcodeset_004637 sim=0.782 same problem human/human
```python
N = int(input())
b = list(map(int, input().split()))
ans = []
for i in range(N):
    # それぞれの数字の余裕度を計算し、余裕度0で一番後ろのやつをとる
    d = -1
    for j in range(N-i):
        if (j+1)-b[j] == 0:
            d = j
    if d == -1:
        print(-1)
        exit()
    else:
        ans.append(b[d])
        del b[d]
[print(a) for a in ans[::-1]]
```

**H2** aigcodeset_003351 sim=0.6772 same problem human/human
```python
#後ろにある数字を消していく

n = int(input())
B = [int(i) for i in input().split()]

ans = []

while len(B) > 0:
    for i in range(len(B) - 1, -1, -1):
        if B[i] == i + 1:
            ans.append(i + 1)
            del B[i]
            break
    else:
        print(-1)
        exit()

print(*ans[::-1], sep='\n')
```

**H3** aigcodeset_001251 sim=0.518 same problem human/human
```python
N = int(input())
b = [int(num) for num in input().split()]
queue = [[b,[]]]

def bfs(queue):
    c,prev = queue.pop(0)
    cnt = len(c)-1
    while cnt >= 0:
        if c[cnt] == cnt+1:
            del c[cnt]
            prev = [cnt+1] + prev
            queue.append([c,prev])
            break
        cnt -= 1
    return queue

ans = -1
while queue != []:
    if queue[-1][0] == []:
        ans = queue[-1]
        break
    queue = bfs(queue)
        
if ans == -1:
    print(ans)
```

**A1** aigcodeset_007203 sim=0.6257 same problem codestral/fix_runtime_error
```python
N=int(input())
b=list(map(int, input().split()))

a=[]

for i in range(N):
  if i+1 < b[i]:
    a=[0]*1
    a[0]= -1
    break
  a.insert(b[i]-1,b[i])

for j in range(N):
  print(a[j])
```

**A2** aigcodeset_006632 sim=0.6095 same problem gemini/from_scratch
```python
n = int(input())
b = list(map(int, input().split()))

a = []
for i in range(n):
  if len(a) == 0 or b[i] <= a[-1]:
    a.append(b[i])
    print(b[i])
  else:
    print(-1)
    break
```

**A3** aigcodeset_006615 sim=0.6041 same problem gemini/fix_runtime_error
```python
N=int(input())
b=list(map(int, input().split()))

a=[]

for i in range(N):
  if i+1 < b[i]:
    print(-1)
    break
  a.insert(b[i]-1,i+1)

for j in range(N):
  print(a[j])
```

## Query 7  (true: AI, problem p03699, found p02973 WRONG)
```python
N = int(input())
S = []
for _ in range(N):
    s = int(input())
    S.append(s)

ans = sum(S)
S.sort()

if ans % 10!= 0:
    print(ans)
else:
    for i in range(N):
        if S[i] % 10!= 0:
            ans -= S[i]
            break
    print(ans)
```

**H1** aigcodeset_004415 sim=0.753 same problem human/human
```python
N=int(input())
A=[int(input()) for i in range(N)]
from bisect import bisect_left
from collections import deque
l=deque()
for a in A:
    i=bisect_left(l,a)
    if i==0:
        l.appendleft(a)
    else:
        l[i-1]=a
print(len(l))
```

**H2** aigcodeset_000994 sim=0.6868 same problem human/human
```python
from bisect import bisect
n = int(input())
a = [int(input()) for _ in range(n)]
dp = [1 for _ in range(n+2)]
for i in range(n):
	b = bisect(dp, -a[i])
	dp[b] = -a[i]
ans = 1
while dp[ans] <= 0:
	ans += 1
print(ans)
```

**H3** aigcodeset_000719 sim=0.5392 same problem human/human
```python
from bisect import bisect_left
from collections import deque
N = int(input())
a = int(input())
ls = deque([a])
for i in range(N-1):
  c = int(input())
  ind = bisect_left(ls,c)
  if ind==0:
    ls.appendleft(c)
    continue
  ls[ind-1] = c
print(len(ls))
```

**A1** aigcodeset_006746 sim=0.9328 same problem gemini/fix_wrong_answer
```python
N = int(input())
A = []
for i in range(N):
    A.append(int(input()))

A.sort()
ans = 1
prev = A[0]
for i in range(1, N):
    if A[i] > prev:
        ans += 1
        prev = A[i]

print(ans)
```

**A2** aigcodeset_005057 sim=0.9278 same problem codestral/fix_wrong_answer
```python
N = int(input())
A = []
for i in range(N):
    A.append(int(input()))

A.sort()
ans = 1
first = A[0]
for num in A:
    if num > first:
        ans += 1
        first = num

print(ans)
```

**A3** aigcodeset_005460 sim=0.9145 same problem gemini/from_scratch
```python
n = int(input())
a = []
for _ in range(n):
  a.append(int(input()))
a.sort()
ans = 1
prev = a[0]
for i in range(1, n):
  if a[i] > prev:
    ans += 1
    prev = a[i]
print(ans)
```

## Query 8  (true: AI, problem p02829, found p02829 OK)
```python
A = int(input())
B = int(input())
C = [1, 2, 3]
C.remove(A)
C.remove(B)
print(C[0])
```

**H1** aigcodeset_002165 sim=0.9789 same problem human/human
```python
A = int(input())
B = int(input())
T = [1,2,3]
T.remove(A)
T.remove(B)
print(T[0])
```

**H2** aigcodeset_002879 sim=0.9631 same problem human/human
```python
A = int(input())
B = int(input())

X = [1, 2, 3]

X.remove(A)
X.remove(B)

print(X[0])
```

**H3** aigcodeset_001130 sim=0.8975 same problem human/human
```python
a, b = [int(input()) for _ in range(2)]
c = [1, 2, 3]
c.remove(a)
c.remove(b)
print(*c)
```

**A1** aigcodeset_006297 sim=0.732 same problem codellama/fix_runtime_error
```python
A = input()
B = input()

if A == 1 and B == 2:
    print(3)
elif A == 2 and B == 1:
    print(3)
elif A == 1 and B == 3:
    print(2)
elif A == 3 and B == 1:
    print(2)
else:
    print(1)
```

**A2** aigcodeset_005163 sim=0.7131 same problem gemini/from_scratch
```python
a = int(input())
b = int(input())
for i in range(1, 4):
  if i != a and i != b:
    print(i)
    break
```

**A3** aigcodeset_006916 sim=0.6933 same problem gemini/fix_runtime_error
```python
a,b=map(int,input().split())
if a==1 and b==2:
	print(3)
elif a==1 and b==3:
	print(2)
elif a==2 and b==1:
	print(3)
elif a==2 and b==3:
	print(1)
elif a==3 and b==1:
	print(2)
else:
	print(1)
```

## Query 9  (true: human, problem p03475, found p02916 WRONG)
```python
N = int(input())
C = [0]*(N-1)
S = [0]*(N-1)
F = [0]*(N-1)
T = [0]*N

for i in range(N-1):
  C[i], S[i], F[i] = map(int, input().split())

for j in range(N-1):
	T[j] = C[j] + S[j]
	for i in range(N-2):
  		if T[j] > S[i+1]:
    		T[j] = T[j] + C[i+1]
  		else:
    		T[j] = S[i+1]

T[N-1] = 0
for i in range(N):
  print(T[i])
```

**H1** aigcodeset_002328 sim=0.9241 same problem human/human
```python
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

cnt = 0
for i in range(N-1):
    cnt += B[A[i]-1]
    if A[i] + 1 == A[i+1]:
        cnt += C[A[i]-1]
    
cnt += B[A[N-1]-1]
print(cnt)
```

**H2** aigcodeset_001852 sim=0.6798 same problem human/human
```python
N = int(input())
food = list(map(int, input().split(" ")))
one = list(map(int, input().split(" ")))
sec = list(map(int, input().split(" ")))
cnt = 0
for i in range(N-1):
  cnt += one[food[i]-1]
  if food[i+1] == food[i]+1:
    cnt += sec[food[i]-1]
cnt += one[food[N-1]-1]
print(cnt)
```

**H3** aigcodeset_004142 sim=0.6289 same problem human/human
```python
n = int(input())

a = list(map(int,input().split()))
b = list(map(int,input().split()))
c = list(map(int,input().split()))

a = [d-1 for d in a]
#print(a)
manzoku = 0
jotai = a[0]

for val in a:
    manzoku = manzoku + b[val]
    if jotai +1 == val:
        manzoku += c[val-1]
    jotai = val

print(manzoku)
```

**A1** aigcodeset_006210 sim=0.9103 same problem codellama/fix_runtime_error
```python
N = int(input())
A = []
B = []
C = []
for i in range(1,N+1):
  A += [int(input())]
for i in range(1,N+1):
  B += [int(input())]
for i in range(1,N+1):
  C += [int(input())]
d = 0
for i in range(1,N):
  if A[i] == A[i-1]+1:
    d += C[i-1]
for i in range(1,N+1):
  d += B[i-1]
print(d)
```

**A2** aigcodeset_005969 sim=0.8339 same problem codellama/fix_wrong_answer
```python
n = int(input())

a = list(map(int,input().split()))
b = list(map(int,input().split()))
c = list(map(int,input().split()))
sum = 0

for i in range(len(a)):
    sum += b[a[i] - 1]
    if(i != len(a) - 1 and a[i] + 1 == a[i + 1]):
        sum += c[a[i] - 1]

print(sum)
```

**A3** aigcodeset_006056 sim=0.8183 same problem codestral/fix_runtime_error
```python
N = int(input())
A_list = list(map(int,input().split()))
B_list = list(map(int,input().split()))
C_list = list(map(int,input().split()))
ans = sum(B_list)

for i in range(1, N):
    if A_list[i] == A_list[i-1] + 1:
        ans += C_list[A_list[i] - 2]
print(ans)
```

## Query 10  (true: AI, problem p02829, found p02777 WRONG)
```python
A = input()
B = input()

if A == 1 and B == 2:
    print(3)
elif A == 2 and B == 1:
    print(3)
elif A == 1 and B == 3:
    print(2)
elif A == 3 and B == 1:
    print(2)
else:
    print(1)
```

**H1** aigcodeset_000748 sim=0.8793 same problem human/human
```python
S,T = input().split()
A,B = map(int,input().split())
u = input()
if S == u:
    print(A-1, B)
elif T == u:
    print(A, B-1)
```

**H2** aigcodeset_000955 sim=0.8368 same problem human/human
```python
s,t = input().split()
a, b = map(int, input().split())
u = input()

if s==u:
    print(a-1,b,sep=' ')
else:
    print(a,b-1,sep=' ')
```

**H3** aigcodeset_003044 sim=0.8284 same problem human/human
```python
S,T = map(str,input().split())
A,B = map(int,input().split())
U = str(input())
if S == U:
    print(A-1,B)
else:
    print(A,B-1)
```

**A1** aigcodeset_005456 sim=0.8494 same problem gemini/from_scratch
```python
S, T = input().split()
A, B = map(int, input().split())
U = input()

if S == U:
  A -= 1
elif T == U:
  B -= 1

print(A, B)
```

**A2** aigcodeset_007171 sim=0.8486 same problem codestral/fix_runtime_error
```python
s, t = input().split()
a, b = map(int, input().split())
u = input()

if u == s:
    print(a - 1, b)
elif u == t:
    print(a, b - 1)
```

**A3** aigcodeset_005366 sim=0.8469 same problem gemini/fix_runtime_error
```python
s, t = input().split()
a, b = map(int, input().split())
u = input()

if s == u:
    print(a - 1, b)
else:
    print(a, b - 1)
```

## Query 11  (true: AI, problem p03829, found p03829 OK)
```python
N, A, B = map(int, input().split())
X = list(map(int, input().split()))
res = 0
for i in range(N - 1):
  res += min(A * (X[i + 1] - X[i]), B)
print(res)
```

**H1** aigcodeset_004248 sim=0.9137 same problem human/human
```python
N,A,B = map(int,input().split())
X = list(map(int,input().split()))

ans = 0
t = 1
for i in range(len(X)-1):
    ans += min((X[i+1] - X[i])*A,B)
print(ans)
```

**H2** aigcodeset_001580 sim=0.8899 same problem human/human
```python
n,a,b=map(int,input().split())
X=list(map(int,input().split()))
ans=0
for i in range(n-1):
    ans+=min(a*(X[i+1]-X[i]),b)
print(ans)
```

**H3** aigcodeset_000987 sim=0.8861 same problem human/human
```python
N,A,B = map(int,input().split())
X = list(map(int,input().split()))

ans = 0
for i in range(1,N):
    if (X[i]-X[i-1])*A > B:
        ans += B
    else:
        ans += (X[i]-X[i-1])*A
print(ans)
```

**A1** aigcodeset_005223 sim=0.9535 same problem gemini/fix_runtime_error
```python
N, A, B = map(int, input().split())
X = list(map(int, input().split()))
res = 0
for i in range(N - 1):
  if X[i + 1] - X[i] > B:
    res += B
  else:
    res += A * (X[i + 1] - X[i])
print(res)
```

**A2** aigcodeset_004921 sim=0.8925 same problem codellama/fix_runtime_error
```python
n, a, b = map(int, input().split())
ans = 0
for i in range(n - 1):
    ans += max(a * (x[i + 1] - x[i]), b)
print(ans)
```

**A3** aigcodeset_007358 sim=0.7315 same problem codestral/fix_wrong_answer
```python
n, a, b = map(int, input().split())
x = list(map(int, input().split()))
ans = 0
for i in range(n - 1):
    dist = x[i + 1] - x[i]
    walk = dist * a
    teleport = b
    ans += min(walk, teleport)
print(ans)
```

## Query 12  (true: human, problem p02959, found p02959 OK)
```python
from collections import deque

N = int(input())
Monsters = deque(map(int, input().split()))
Fighters = deque(map(int, input().split()))

Knockouts = 0
prev_ftr = 0

while len(Monsters) > 0:
  mon = Monsters.popleft()
  ko1 = min(mon, prev_ftr)
  #print(mon, prev_ftr, ko1)
  Knockouts += ko1
  mon -= ko1

  if len(Fighters) > 0:
    ftr = Fighters.popleft()
    ko2 = min(mon, ftr)
    #print(mon, ftr, ko2)
```

**H1** aigcodeset_000305 sim=0.4558 same problem human/human
```python
n = int(input())
a, b = [list(map(int, input().split())) for i in range(2)]

ans = 0
for i in range(n):
  ans += min(a[i], b[i])
  b[i] = max(0, b[i]-a[i])
  ans += min(a[i+1], b[i])
  a[i+1] = max(0, a[i+1]-b[i])
print(ans)
```

**H2** aigcodeset_003053 sim=0.4508 same problem human/human
```python
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

c = [0]*(n+1)
ans = 0
for i in range(n):
  if b[i]+c[i] > a[i]:
    if c[i]>=a[i]:
      ans = ans + a[i]
      c[i+1] = b[i]
    else:
      ans = ans + a[i]
      c[i+1] = b[i]+c[i]-a[i]
  else:
    ans = ans + b[i] + c[i]
if c[n]>a[n]:
  ans = ans + a[n]
else:
  ans = ans + c[n]
print(ans)
```

**H3** aigcodeset_003839 sim=0.4071 same problem human/human
```python
n = int(input())
a = list(map(int,input().split()))
b = list(map(int,input().split()))
ans = 0
for i in range(n):
    if b[i] <= a[i]:
        ans += b[i]
    elif b[i] <= a[i] + a[i+1]:
        a[i+1] = a[i+1] + a[i] - b[i]
        ans += b[i]
    else:
        ans += a[i] + a[i+1]
        a[i+1] = 0
print(ans)
```

**A1** aigcodeset_006262 sim=0.6884 same problem codestral/from_scratch
```python
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

total_defeated = 0
for i in range(n):
    # Calculate the minimum number of monsters that can be defeated by the current hero and the next one
    min_defeated = min(a[i], b[i])
    total_defeated += min_defeated

    # Subtract the defeated monsters from the current town and the next one
    a[i] -= min_defeated
    a[i + 1] -= max(0, min_defeated - b[i])

print(total_defeated)
```

**A2** aigcodeset_005034 sim=0.6521 same problem gemini/from_scratch
```python
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

total_monsters = 0
for i in range(n):
    total_monsters += min(a[i], b[i])
    a[i + 1] -= max(0, b[i] - a[i])

total_monsters += min(a[n], b[n - 1])

print(total_monsters)
```

**A3** aigcodeset_005727 sim=0.4997 same problem gemini/fix_runtime_error
```python
N = int(input())
A = [int(i) for i in input().split()]
B = [int(i) for i in input().split()]
ret = [0] * N
mon = 0
for i in range(0, N):
  if(int(B[i]) - int(A[i]) >= 0 ):
    mon = mon + int(A[i])
    B[i+1] = int(B[i+1]) - ( int(B[i]) - int(A[i]) )
  else:
    mon = mon + int(B[i])

if(int(A[N]) - int(B[N-1]) > 0):
  mon = mon + (int(A[N]) - int(B[N-1]))

else:
  mon = mon + int(B[N-1])

print(mon)
```

## Query 13  (true: human, problem p03061, found p03061 OK)
```python
import numpy as np
N = int(input())
A = np.array([int(i) for i in input().split()])

cm = np.prod(A)
max_gcd = 0
for i in range(len(A)):
    Am = A.copy()
    Am[i] = cm
    gcd = np.gcd.reduce(Am)
    if gcd > max_gcd:
        max_gcd = gcd
print(max_gcd)
```

**H1** aigcodeset_003882 sim=0.6857 same problem human/human
```python
N = int(input())
A = list(map(int,input().split()))
L = [0]
R = [0]
import fractions
def GCD(a,b):
    if a*b != 0:
        return fractions.gcd(a,b)
    else:
        return max(a,b)
for i in range(N-1):
    x = A[i]
    y = A[-i-1]
    L.append(GCD(L[i],x))
    R.append(GCD(R[i],y))
ans = 0
for i in range(N):
    x = L[i]
    y = R[-i-1]
    ans = max(GCD(x,y),ans)
print(ans)
```

**H2** aigcodeset_003255 sim=0.6397 same problem human/human
```python
from fractions import gcd
N = int(input())
A = list(map(int, input().split()))
L = [0]
R = [0]
M = []
maximum = 0
for n in range(N):
    L.append(gcd(A[n], L[-1]))
    R.append(gcd(A[-n-1], R[-1]))
R.reverse()
for n in range(N):
    M.append(gcd(L[n], R[n+1]))
for n in range(N):
    maximum = max(maximum, M[n])
print(maximum)
```

**H3** aigcodeset_000404 sim=0.5906 same problem human/human
```python
import os, sys, re, math, fractions

N = int(input())
A = list(map(int,input().split(' ')))

memo_L = [1] * N
memo_R = [1] * N

for i in range(N):
    if i == 0:
        memo_L[i] = A[i]
        memo_R[i] = A[N-1]
    elif i == N - 1:
        memo_L[i] = fractions.gcd(A[i], memo_L[i-1])
        memo_R[i] = fractions.gcd(A[0], memo_R[i-1])
    else:
        memo_L[i] = fractions.gcd(A[i], memo_L[i-1])
        memo_R[i] = fractions.gcd(A[N-1-i], memo_R[i-1])

memo_R.reverse()

ret = 1
for j in range(N):
    if j == 0:
        ret = max(ret, memo_R[1])
```

**A1** aigcodeset_005679 sim=0.9639 same problem codestral/fix_runtime_error
```python
import numpy as np

N = int(input())
A = np.array([int(i) for i in input().split()])

max_gcd = 0
for i in range(1, 10**9 + 1):
    Am = A.copy()
    Am[0] = i
    gcd = np.gcd.reduce(Am)
    if gcd > max_gcd:
        max_gcd = gcd
print(max_gcd)
```

**A2** aigcodeset_006196 sim=0.6777 same problem codellama/fix_runtime_error
```python
from math import gcd

n = int(input())
A = list(map(int, input().split()))
left = [0] * (n + 1)
right = [0] * (n + 2)

for a, aa, i in zip(A, A[::-1], range(1, n + 1)):
    left[i] = gcd(a, left[i - 1])
    right[n + 1 - i] = gcd(aa, right[n - i + 2])

ans = 1
for i in range(1, n + 1):
    if i == 1:
        if ans < right[i + 1]:
            ans = right[i + 1]
    elif i == n:
        if ans < left[i - 1]:
            ans = left[i - 1]
    else:
        if ans < gcd(left[i - 1], right[i + 1]):
            ans = gcd(left[i - 1], right[i + 1])

print(ans)
```

**A3** aigcodeset_006818 sim=0.6683 same problem gemini/fix_wrong_answer
```python
N = int(input())
A_arr=input().split()

for i in range(N):
    A_arr[i] = int(A_arr[i])
    
A_arr.sort(reverse=True)
_A_arr = [A_arr[i] for i in range(N)]


# import math
# from functools import reduce

# def gcd_list(numbers):
    # return reduce(math.gcd, numbers)

def gcd(a,b):
    if b==0:return a
    return gcd(b,a%b)


def gcd_l(x):
    max_element = 1
    if len(x)==1:
        return x[0]
```

## Query 14  (true: AI, problem p03274, found p03829 WRONG)
```python
n, k = map(int, input().split())
x = list(map(int, input().split()))

# Calculate the distances between candles
distances = [abs(x[i] - x[i - 1]) for i in range(1, n)]

# Sort the distances in ascending order
distances.sort()

# Calculate the total distance to travel
total_distance = sum(distances[:k - 1])

# Add the distance to the first candle
total_distance += abs(x[0])

# Print the total distance
print(total_distance)

```

**H1** aigcodeset_004382 sim=0.6201 same problem human/human
```python
# D - Walk and Teleport
n,a,b=map(int,input().split())
x=list(map(int,input().split()))
dist=[]

for i in range(n-1):
    dist.append(x[i+1]-x[i])
ans=0
for i in range(len(dist)):
    if dist[i]*a<b:
        ans+=dist[i]*a
    else:
        ans+=b
print(ans)
```

**H2** aigcodeset_004721 sim=0.5368 same problem human/human
```python
n, a, b = map(int, input().split())

x = list(map(int, input().split())) 

cost = 0
for i in range(n-1):
    length = x[i+1] - x[i]
    if length*a < b:
        cost += length*a
    else:
        cost += b
print(cost)
```

**H3** aigcodeset_004248 sim=0.4859 same problem human/human
```python
N,A,B = map(int,input().split())
X = list(map(int,input().split()))

ans = 0
t = 1
for i in range(len(X)-1):
    ans += min((X[i+1] - X[i])*A,B)
print(ans)
```

**A1** aigcodeset_005418 sim=0.7282 same problem gemini/from_scratch
```python
n, a, b = map(int, input().split())
x = list(map(int, input().split()))

# Calculate the distances between towns
distances = [x[i] - x[i - 1] for i in range(1, n)]

# Calculate the minimum fatigue level using dynamic programming
dp = [0] * n
dp[0] = 0
for i in range(1, n):
    dp[i] = min(dp[i - 1] + a * distances[i - 1], dp[i - 1] + b)

# Print the minimum fatigue level
print(dp[n - 1])
```

**A2** aigcodeset_007358 sim=0.6287 same problem codestral/fix_wrong_answer
```python
n, a, b = map(int, input().split())
x = list(map(int, input().split()))
ans = 0
for i in range(n - 1):
    dist = x[i + 1] - x[i]
    walk = dist * a
    teleport = b
    ans += min(walk, teleport)
print(ans)
```

**A3** aigcodeset_004951 sim=0.5515 same problem codestral/fix_runtime_error
```python
N, A, B = map(int, input().split())
X = list(map(int, input().split()))

dist = [0] * (N - 1)
for i in range(N - 1):
    dist[i] = X[i + 1] - X[i]

fatigue = 0
for i in dist:
    if i * A < B:
        fatigue += i * A
    else:
        fatigue += B

print(fatigue)
```

## Query 15  (true: AI, problem p03003, found p03003 OK)
```python
n, m = map(int, input().split())
S = list(map(int, input().split()))
T = list(map(int, input().split()))

p = 10**9 + 7

def LCS(S, T, n, m):
    # Initialization
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 1

    # DP
    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0:
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % p
            if j > 0:
                dp[i][j] = (dp[i][j] + dp[i][j - 1]) % p
            if i > 0 and j > 0 and S[i - 1] == T[j - 1]:
                dp[i][j] = (dp[i][j] - dp[i - 1][j - 1] + p) % p
```

**H1** aigcodeset_001959 sim=0.6536 same problem human/human
```python
N, M = map(int, input().split())
s = [int(c) for c in input().split()]
t = [int(c) for c in input().split()]
MOD = 10**9+7

dp = [[0]*(M+1) for i in range(N+1)]
dp_cum = [[0]*(M+1) for i in range(N+1)]

dp[0][0] = 1
for i in range(N+1):
  dp_cum[i][0] = 1
for j in range(M+1):
  dp_cum[0][j] = 1
  
for i in range(N):
  for j in range(M):
    if s[i]!=t[j]:
      dp[i+1][j+1] = 0
    else:
      dp[i+1][j+1] = dp_cum[i][j]
    dp_cum[i+1][j+1] = dp_cum[i][j+1] + dp_cum[i+1][j] - dp_cum[i][j] + dp[i+1][j+1]
    dp_cum[i+1][j+1] %= MOD

print(dp_cum[-1][-1])
```

**H2** aigcodeset_002782 sim=0.5165 same problem human/human
```python
def main():
    MOD = 10 ** 9 + 7
    N, M = list(map(int, input().split(' ')))
    S = list(map(int, input().split(' ')))
    T = list(map(int, input().split(' ')))
    dp = [[0 for _ in range(M + 1)] for _ in range(N + 1)]
    sdp = [[0 for _ in range(M + 1)] for _ in range(N + 1)]
    for n, s in enumerate(S):
        for m, t in enumerate(T):
            sdp_val = sdp[n][m]
            new_dp_val = (sdp_val + 1) if s == t else 0
            dp[n + 1][m + 1] = new_dp_val % MOD
            sdp[n + 1][m + 1] = (sdp[n][m + 1] + sdp[n + 1][m] - sdp_val + new_dp_val) % MOD
    print((sdp[N][M] +
```

**H3** aigcodeset_004536 sim=0.3949 same problem human/human
```python
from collections import defaultdict


class Bit:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def sum(self, i):
        s = 0
        while i > 0:
            s = (s + self.tree[i]) % 1000000007
            i -= i & -i
        return s

    def add(self, i, x):
        while i <= self.size:
            self.tree[i] = (self.tree[i] + x) % 1000000007
            i += i & -i


n, m = list(map(int, input().split()))
sss = list(map(int, input().split()))
ttt = list(map(int, input().split()))

```

**A1** aigcodeset_005312 sim=0.8386 same problem gemini/fix_runtime_error
```python
n, m = map(int, input().split())
S = list(map(int, input().split()))
T = list(map(int, input().split()))

p = 10**9 + 7

def LCS(S,T,n,m):
  #初期化
  dp=[[0 for i in range(m+1)] for j in range(n+1)]
  for i in range(m+1): dp[i][0] = 1
  for j in range(n+1): dp[0][j] = 1
    
  #DP
  for i in range(n):
    for j in range(m):
      dp[i+1][j+1] = (dp[i][j+1] + dp[i+1][j])%p
      if S[i] == T[j]:
        dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j])%p
  return dp[n][m]%p

print(LCS(S,T,n,m))
```

**A2** aigcodeset_005132 sim=0.6721 same problem codellama/fix_runtime_error
```python
N, M = map(int, input().split())
S = input().split()
T = input().split()

dp = [[0] * (N + 2) for _ in range(M + 2)]
for i in range(N+1):
    dp[i][0] = 1
for j in range(M+1):
    dp[0][j] = 1

for i, s in enumerate(S, 1):
    for j, t in enumerate(T, 1):
        dp[i][j] += dp[i - 1][j]
        dp[i][j] += dp[i][j - 1]
        if s == t:
            dp[i][j] += dp[i - 1][j - 1]
        dp[i][j] -= dp[i - 1][j - 1]
        dp[i][j] %= mod

print(dp[N][M])
```

**A3** aigcodeset_006373 sim=0.5169 same problem codestral/from_scratch
```python

def solve(n, m, s, t):
    mod = 10**9 + 7
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0:
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % mod
            if j > 0:
                dp[i][j] = (dp[i][j] + dp[i][j - 1]) % mod
            if i > 0 and j > 0 and s[i - 1] == t[j - 1]:
                dp[i][j] = (dp[i][j] - dp[i - 1][j - 1] + mod) % mod
    return dp[n][m]

n, m = map(int, input().split())
s = list(map(int, input().split()))
t = list(map(int, input().split()))
print(solve(n, m, s,
```