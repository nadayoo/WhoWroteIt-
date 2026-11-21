# Usefulness check: are the retrieved examples comparable solutions to the same task?

Rate each side 0 = not comparable, 1 = partly, 2 = clearly comparable. Fill `usefulness_sheet.csv`.


## Query 1  (true: human, problem p02721, found p03721 WRONG)
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

**H1** aigcodeset_003646 sim=0.7848 same problem human/human
```python
n, k = map(int,input().split())
arr = []
for _ in range(n):
    a,b = map(int,input().split())
    arr.append([a,b])
arr.sort()
for i in arr:
    k -= i[1]
    if k <= 0:
        print(i[0])
        break
```

**H2** aigcodeset_003408 sim=0.7723 same problem human/human
```python
# C問題
N, K = map(int, input().split())
ab = []
for i in range(N):
    ab.append(list(map(int, input().split())))
ab.sort()
tmp_k = 0
tmp_i = 0
for i in range(N):
    tmp_k += ab[i][1]
    if tmp_k >= K:
        tmp_i = i
        break
print(ab[tmp_i][0])        
```

**H3** aigcodeset_000376 sim=0.7618 same problem human/human
```python
N, K = map(int, input().split())
ab = [[int(_) for _ in input().split()] for _ in range(N)]
ab = sorted(ab, key=lambda x: x[0])
tmp = 0

for i in ab:
    tmp += i[1]
    if tmp >= K:
        ans = i[0]
        break
print(ans)
```

**A1** aigcodeset_006698 sim=0.7858 same problem gemini/fix_runtime_error
```python
n,k=map(int,input().split())

a=[]
b=[]
d={}
for _ in range(n):
    a_i,b_i=map(int,input().split())
    d[a_i]=d.get(a_i,0)+b_i

d=sorted(d.items(),key=lambda x: x[0])


prev=0
for key,value in d:
    a.append(key)
    b.append(prev+value)

    prev+=value


for j in range(len(b)):
    if k<=b[j]:
        break

print(a[j])
```

**A2** aigcodeset_006416 sim=0.7473 same problem gemini/from_scratch
```python
n, k = map(int, input().split())
a = []
for _ in range(n):
    ai, bi = map(int, input().split())
    a.extend([ai] * bi)
a.sort()
print(a[k - 1])
```

**A3** aigcodeset_006708 sim=0.6913 same problem codestral/fix_wrong_answer
```python
from collections import defaultdict
import sys
sdin = sys.stdin.readline

N, K = map(int, sdin().split())
AB = []
for _ in range(N):
    AB.append(tuple(map(int, sdin().split())))

dic = defaultdict(int)

# Add elements to the dictionary
for a, b in AB:
    dic[a] += b

# Create a list of keys in ascending order
key_list = sorted(dic.keys())

# Subtract from res until it becomes 0
res = K
i = 0
while res > 0:
    res -= dic[key_list[i]]
    ans = key_list[i]
    i += 1
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

**H1** aigcodeset_002751 sim=0.8708 same problem human/human
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

**H2** aigcodeset_000186 sim=0.7919 same problem human/human
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

**H3** aigcodeset_004630 sim=0.765 same problem human/human
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

**A1** aigcodeset_005769 sim=0.8416 same problem codellama/fix_runtime_error
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

**A2** aigcodeset_004902 sim=0.7587 same problem codestral/fix_runtime_error
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

**A3** aigcodeset_007252 sim=0.7462 same problem codestral/fix_wrong_answer
```python
import sys

input=sys.stdin.readline
n,k=map(int,input().split())
x=list(map(int,input().split()))
s=0
for i in range(n):
    if x[i]>=0:
        s=i
        break
res=10**9
if s+k-1<n:
  res=x[s+k-1]
if s-k>=0 and res>abs(x[s-k]):
  res=abs(x[s-k])
for i in range(1,k):
    if s+k-i-1>n-1 or s-i<0:
        continue
    if x[s+k-i-1]+2*abs(x[s-i])<res:
        res=x[s+k-i-1]+2*abs(x[s-i])
for i in range(1,k):
    if s-k+i<0 or s+i-1>n-1:
        continue
    if abs(x[s-k+i])+2*x[s+i-1]<res:
        res=abs(x[s-k+i])+2*x[s+i-1]
```

## Query 3  (true: AI, problem p02777, found p02777 OK)
```python
  def solve(S, T, A, B, U):
    if U == S:
        return A - 1, B
    else:
        return A, B - 1
```

**H1** aigcodeset_000748 sim=0.6234 same problem human/human
```python
S,T = input().split()
A,B = map(int,input().split())
u = input()
if S == u:
    print(A-1, B)
elif T == u:
    print(A, B-1)
```

**H2** aigcodeset_003044 sim=0.5972 same problem human/human
```python
S,T = map(str,input().split())
A,B = map(int,input().split())
U = str(input())
if S == U:
    print(A-1,B)
else:
    print(A,B-1)
```

**H3** aigcodeset_000535 sim=0.5789 same problem human/human
```python
s,t = map(str,input().split())
a,b = map(int,input().split())
u = str(input())
if(u==s):
    a-=1
else:
    b-=1
print(a,b)
```

**A1** aigcodeset_007171 sim=0.6125 same problem codestral/fix_runtime_error
```python
s, t = input().split()
a, b = map(int, input().split())
u = input()

if u == s:
    print(a - 1, b)
elif u == t:
    print(a, b - 1)
```

**A2** aigcodeset_005456 sim=0.6011 same problem gemini/from_scratch
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

**A3** aigcodeset_005366 sim=0.5968 same problem gemini/fix_runtime_error
```python
s, t = input().split()
a, b = map(int, input().split())
u = input()

if s == u:
    print(a - 1, b)
else:
    print(a, b - 1)
```

## Query 4  (true: human, problem p02769, found p02769 OK)
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

**H1** aigcodeset_001036 sim=0.8067 same problem human/human
```python
n,k = map(int,input().split())

mod = 10**9 + 7
l = 10**7
fac = [1]*l
facr = [1]*l

for i in range(l-1):
    fac[i+1] = fac[i]*(i+1)%mod
facr[l-1] = pow(fac[l-1],mod - 2,mod)
for i in range(1,l)[::-1]:
    facr[i-1] = facr[i]*i%mod

def combi(N,K):
    return fac[N]*facr[N-K]%mod*facr[K]%mod

ans = 1
for i in range(1,min(n,k+1)):
    ans += combi(n,i)*combi(n-1,i)%mod
    ans %= mod
print(ans)
```

**H2** aigcodeset_003540 sim=0.7799 same problem human/human
```python
n,k=map(int,input().split())
MOD=10**9+7
def comb(n,m,p=10**9+7):
    buf=perm[n]*inv[m]
    buf%=MOD
    buf*=inv[n-m]
    buf%=MOD
    return buf
def inv_mod(a):
  return pow(a,MOD-2,MOD)

perm=[1]*(2*n+1)
for i in range(1,2*n+1):
    perm[i]=perm[i-1]*(i)
    perm[i]%=MOD
inv=[1]*(2*n+1)
inv[-1]=inv_mod(perm[-1])
for i in range(2*n-1,-1,-1):
    inv[i]=inv[i+1]*(i+1)
    inv[i]%=MOD
#print(comb(5,2))
buf=comb(2*n-1,n)
for j in range(k+1,n):
    x=comb(n,j)*comb(n-1,j)
    x%=MOD
```

**H3** aigcodeset_002396 sim=0.7352 same problem human/human
```python
import sys
read = sys.stdin.read
readline = sys.stdin.readline
readlines = sys.stdin.readlines

MOD = 10**9+7

fac = [1, 1]
f_inv = [1, 1]
inv = [0, 1]

def prepare(n, mod):
    for i in range(2, n+1):
        fac.append((fac[-1] * i) % mod)
        inv.append((-inv[mod % i] * (mod//i)) % mod)
        f_inv.append((f_inv[-1] * inv[-1]) % mod)


def modcmb(n, r, mod):
    if n < 0 or r < 0 or r > n:
        return 0

    return fac[n] * f_inv[r] * f_inv[n-r] % mod


```

**A1** aigcodeset_005995 sim=0.7775 same problem gemini/fix_wrong_answer
```python
n, k = map(int, input().split())
MOD = 10**9 + 7

fact = [0] * (n+1)
inv_fact = [0] * (n+1)

fact[0] = 1
for i in range(1, n+1):
    fact[i] = fact[i-1] * i % MOD

inv_fact[n] = pow(fact[n], MOD-2, MOD)
for i in range(n, 0, -1):
    inv_fact[i-1] = inv_fact[i] * i % MOD


def nCk(n, k):
    return fact[n] * inv_fact[k] * inv_fact[n-k] % MOD
  
ans = 0
for i in range(k+1):
    ans += nCk(n, i) * nCk(n-1, n-i-1) % MOD
print(ans % MOD)
```

**A2** aigcodeset_007272 sim=0.7663 same problem codestral/from_scratch
```python
import sys

MOD = 10**9 + 7

def factorial(n):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    return fact

def power(x, n):
    result = 1
    while n > 0:
        if n % 2 == 1:
            result = (result * x) % MOD
        x = (x * x) % MOD
        n //= 2
    return result

def combinations(n, k):
    fact = factorial(n)
    return (fact[n] * power(fact[k], MOD - 2) * power(fact[n - k], MOD - 2)) % MOD

def count_combinations(n, k):
    if k == 0:
```

**A3** aigcodeset_006357 sim=0.7479 same problem gemini/fix_runtime_error
```python
#input
n, k = map(int, input().split())

#output
mod = pow(10, 9) + 7

def cmb(n, r):
    res = 1
    fac = 1
    for i in range(r):
        res *= (n-i)
        res %= mod
        fac *= (i+1)
        fac %= mod
    return res*pow(fac, mod-2, mod) % mod

if n <= k-1:
    print(cmb(2*n-1, n))
else:
    answer = 0
    for m in range(k+1):
        answer += cmb(n, m)*cmb(n-1, k-m) % mod

print(answer % mod)
```

## Query 5  (true: AI, problem p02793, found p02793 OK)
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

**H1** aigcodeset_002662 sim=0.8013 same problem human/human
```python
from fractions import gcd


def lcm(a, b):
    return a * b // gcd(a, b)


MOD = 10 ** 9 + 7

N = int(input())
A = list(map(int, input().split()))


_lcm = 1
for a in A:
    _lcm = lcm(_lcm, a)

ans = _lcm * sum([pow(s, MOD - 2, MOD) for s in A]) % MOD

print(int(ans % MOD))
```

**H2** aigcodeset_000954 sim=0.7753 same problem human/human
```python
from fractions import gcd
N=int(input())
A=list(map(int,input().split(' ')))
ans = 0
l_1 = 1
l_2 = 1
mod = 10**9+7
def lcm(x,y):
    return x*y//gcd(x,y)
for i in A[:N//2]:
    l_1 = lcm(l_1,i)
for j in A[N//2:]:
    l_2 = lcm(l_2,j)
l = lcm(l_1,l_2)
out1 = sum([l_1//i for i in A[:N//2]])
out2 = sum([l_2//i for i in A[N//2:]])
out1 *= l//l_1
out2 *= l//l_2
print((out1+out2)%mod)
```

**H3** aigcodeset_004634 sim=0.7634 same problem human/human
```python
def main():
    from fractions import gcd
    a = int(input())
    l = list(map(int,input().split()))



    def lcm(a,b):
        return a*b//gcd(a,b)
    A= 1
    MOD = 10**9+7
    ans = 0
    t = 1
    for i in l:
        t = t* i //gcd(t,i)
    for i in l:
        ans +=t//i
    ans %=MOD
    print(ans)
main()
```

**A1** aigcodeset_006072 sim=0.8084 same problem codestral/from_scratch
```python
from collections import defaultdict
from math import gcd

MOD = 10**9 + 7

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    # Calculate the gcd of all elements in a
    g = a[0]
    for i in range(1, n):
        g = gcd(g, a[i])

    # Divide all elements by g
    for i in range(n):
        a[i] //= g

    # Count the frequency of each element in a
    freq = defaultdict(int)
    for i in range(n):
        freq[a[i]] += 1

    # Calculate the minimum possible value of B_1 +... + B_N
    ans = 0
```

**A2** aigcodeset_006053 sim=0.7891 same problem gemini/from_scratch
```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

n = int(input())
a = list(map(int, input().split()))

g = a[0]
for i in range(1, n):
    g = gcd(g, a[i])

ans = 0
for i in range(n):
    ans += a[i] // g
ans %= (10**9 + 7)
print(ans)
```

**A3** aigcodeset_006779 sim=0.7144 same problem codestral/fix_runtime_error
```python
import fractions
mod = 1000000007

def add(a, b):
    return (a + b) % mod

n = int(input())
a = list(map(int,input().split()))
ans = a[0]
ans_ans = 0
for i in range(1, n):
    ans = ans * a[i] // fractions.gcd(ans, a[i])

for j in range(0, n):
    ans_ans = add(ans_ans, ans // a[j])

print(int(ans_ans))
```

## Query 6  (true: human, problem p03061, found p03496 WRONG)
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

**H1** aigcodeset_000322 sim=0.7421 same problem human/human
```python
n=int(input())
a=list(map(int,input().split()))
print(2*n-1)
if abs(max(a))>=abs(min(a)):
    b=a.index(max(a))
    for i in range(n):
        print(b+1,i+1)
    for j in range(n-1):
        print(j+1,j+2)
    
else:
    b=a.index(min(a))
    for i in range(n):
        print(b+1,i+1)
    for j in range(n-1,0,-1):
        print(j+1,j)
```

**H2** aigcodeset_002134 sim=0.7219 same problem human/human
```python
N = int(input())
A = list(map(int, input().split()))

M = max(A)
m = min(A)
print(2*N-2)

if M+m >= 0:
  a = A.index(M)
  for i in range(N):
    if i != a:
      #A[i] += M
      print(a+1, i+1)
  for i in range(N-1):
    #A[i+1] += A[i]
    print(i+1, i+2)
else:
  a = A.index(m)
  for i in range(N):
    if i != a:
      print(a+1, i+1)
  for i in range(N, 1, -1):
    print(i, i-1)
```

**H3** aigcodeset_004707 sim=0.6022 same problem human/human
```python
def main():
    n, *a = map(int, open(0).read().split())
    p, q = min(a), max(a)

    if p == q:
        print(0)
        exit()

    x = []
    f, g = a.index(p) + 1, a.index(q) + 1
    if abs(p) > abs(q):
        for i in range(n):
            if a[i] > 0:
                x.append((f, i + 1))
        for i in range(n - 1):
            x.append((n - i, n - i - 1))

    else:
        for i in range(n):
            if a[i] < 0:
                x.append((g, i + 1))
        for i in range(n - 1):
            x.append((i + 1, i + 2))

    print(len(x))
```

**A1** aigcodeset_007010 sim=0.6269 same problem codestral/fix_wrong_answer
```python
N, *A = map(int, open(0).read().split())

mi = min(A)

ans = []
for i in range(1, N):
    ans.append((0, i))

for i in range(1, N):
    ans.append((i, i - 1))

for i in range(1, N):
    ans.append((i, 0))

print(len(ans))
for x, y in ans:
    print(x + 1, y + 1)
```

**A2** aigcodeset_005758 sim=0.6138 same problem gemini/fix_wrong_answer
```python
N = int(input())
A = list(map(int, input().split()))

ans = []
for i in range(N - 1, 0, -1):
    if A[i - 1] > A[i]:
        ans.append((i, i + 1))
        A[i] += A[i - 1]
print(len(ans))
for x, y in ans:
    print(x, y)
```

**A3** aigcodeset_006278 sim=0.6137 same problem gemini/fix_runtime_error
```python
n = int(input())
m = [int(i) for i in input().split()]

ins = []
l = sorted(m)
if abs(l[-1]) > abs(l[0]):
    # Focus on the largest element
    highest = m.index(l[-1])
    for i in range(len(m)-1):
        if m[i] > m[i+1]:
            m[i+1] += ((abs(m[i+1]-m[i])//abs(l[-1]))+1)
            ins.extend([str(highest+1)+" "+str(i+2)] * ((abs(m[i+1]-m[i])//abs(l[-1]))+1))
else:
    lowest = m.index(l[0])
    for i in range(len(m)-1,0,-1):
        if m[i] < m[i-1]:
            m[i] += ((abs(m[i-1]-m[i])//abs(l[0]))+1)
            ins.extend([str(lowest+1)+" "+str(i)] * ((abs(m[i-1]-m[i])//abs(l[
```

## Query 7  (true: AI, problem p03699, found p03699 OK)
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

**H1** aigcodeset_003555 sim=0.9603 same problem human/human
```python
N = int(input())
s = []
for i in range(N):
    s.append(int(input()))
ans = sum(s)
if ans%10 == 0:
    flag = True
    s.sort()
    for x in s:
        if x%10!=0:
            print(ans - x)
            flag = False
            break
    if flag:
        print(0)
else:
    print(ans)
```

**H2** aigcodeset_000138 sim=0.9262 same problem human/human
```python
N = int(input())
s = []

for i in range(N):
	s.append(int(input()))
s.sort()

max = sum(s)
for i in range(N):
	if(max % 10 != 0):
		print(max)
		exit(0)
	elif((max-s[i]) % 10 != 0):
		print(max-s[i])
		exit(0)
print("0")
```

**H3** aigcodeset_002676 sim=0.9008 same problem human/human
```python
n = int(input())
s = [int(input()) for i in range(n)]
s = sorted(s)
if sum(s)%10!=0:
    print(sum(s))
else:
    for i in range(n):
        if (sum(s)-s[i])%10!=0:
            print(sum(s)-s[i])
            exit()
    else:
        print("0")
```

**A1** aigcodeset_004787 sim=0.8546 same problem gemini/fix_wrong_answer
```python
n = int(input())

s=[int(input()) for i in range(n)]

ans=sum(s)
if ans % 10 != 0:
    print(ans)
else:
    for i in range(len(s)-1, -1, -1):
        if s[i] % 10 != 0:
            print(ans-s[i])
            break
    else:
        print(0)
```

**A2** aigcodeset_005311 sim=0.8226 same problem gemini/fix_runtime_error
```python
a=int(input())
b=[int(input()) for i in range(a)]
b.sort()
c=sum(b)
if c%10!=0:
  print(c)
else:
  for i in range(a-1,-1,-1):
    if b[i]%10!=0:
      d=c-b[i]
      print(d)
      break
  if d==c:
    print(0)
```

**A3** aigcodeset_004876 sim=0.7245 same problem codestral/from_scratch
```python
n = int(input())
scores = [int(input()) for _ in range(n)]
scores.sort(reverse=True)
max_score = 0
current_score = 0
for score in scores:
    current_score += score
    if current_score % 10!= 0:
        max_score = max(max_score, current_score)
print(max_score)
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

**H1** aigcodeset_002879 sim=0.9564 same problem human/human
```python
A = int(input())
B = int(input())

X = [1, 2, 3]

X.remove(A)
X.remove(B)

print(X[0])
```

**H2** aigcodeset_001130 sim=0.9446 same problem human/human
```python
a, b = [int(input()) for _ in range(2)]
c = [1, 2, 3]
c.remove(a)
c.remove(b)
print(*c)
```

**H3** aigcodeset_002165 sim=0.9393 same problem human/human
```python
A = int(input())
B = int(input())
T = [1,2,3]
T.remove(A)
T.remove(B)
print(T[0])
```

**A1** aigcodeset_005163 sim=0.6054 same problem gemini/from_scratch
```python
a = int(input())
b = int(input())
for i in range(1, 4):
  if i != a and i != b:
    print(i)
    break
```

**A2** aigcodeset_006297 sim=0.5777 same problem codellama/fix_runtime_error
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

**A3** aigcodeset_006916 sim=0.5774 same problem gemini/fix_runtime_error
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

## Query 9  (true: human, problem p03475, found p02714 WRONG)
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

**H1** aigcodeset_004682 sim=0.7038 same problem human/human
```python
n=int(input())
s=input()
a = [0,0,0]
for c in s:
    if c=='R':
        a[0]+=1
    elif c=='G':
        a[1]+=1
    else:
        a[2]+=1
total = a[0]*a[1]*a[2]
for d in range(n):
    for i in range(n):
        j = i+d
        k = i+2*d
        if j>=n or k>=n:
            break
        if s[i]!=s[j] and s[j]!=s[k] and s[i]!=s[k]:
            total -= 1
print(total)
```

**H2** aigcodeset_003742 sim=0.5261 same problem human/human
```python
def main():
    n=int(input())
    s=list(input())
    rgb=[[0]*3 for _ in range(n)]
    rgb[0][0],rgb[0][1],rgb[0][2]=s.count('R'),s.count('G'),s.count('B')
    ans=0

    def num(x):
        if x=='R':
            return 0
        elif x=='G':
            return 1
        else:
            return 2
    for i in range(n-1):
        rgb[i+1][0],rgb[i+1][1],rgb[i+1][2]=rgb[i][0],rgb[i][1],rgb[i][2]
        rgb[i+1][num(s[i])]-=1

    for i in range(n-2):
        for j in range(i+1,n-1):
            if s[i]==s[j]:
                continue
            else:
                ans+=rgb[j+1][3-num(s[i
```

**H3** aigcodeset_000670 sim=0.4402 same problem human/human
```python
import sys
sys.setrecursionlimit(1 << 25)
read = sys.stdin.readline
ra = range
enu = enumerate


def read_ints():
    return list(map(int, read().split()))


def read_a_int():
    return int(read())


def read_tuple(H):
    '''
    H is number of rows
    '''
    ret = []
    for _ in range(H):
        ret.append(tuple(map(int, read().split())))
    return ret


```

**A1** aigcodeset_007148 sim=0.7578 same problem codestral/fix_wrong_answer
```python
N = int(input())
S = list(input())
cnt = 0

for i in range(N-2):
  for j in range(i+1, N-1):
    if S[i] == S[j]:
      continue
    for k in range(j+1, N):
      if S[i]!= S[k] and S[j]!= S[k] and j-i!= k-j:
        cnt += 1

print(cnt)
```

**A2** aigcodeset_005547 sim=0.7528 same problem codellama/fix_runtime_error
```python
import sys
input = sys.stdin.readline
N = int(input())
S = [int(input()) for _ in range(N)]
t = 0
for i in range(1, N - 1):
    for j in range(i + 1, N):
        for k in range(j + 1, N + 1):
            if S[i - 1] != S[j - 1] and S[j - 1] != S[k - 1] and S[k - 1] != S[i - 1] and j - i != k - j:
                t += 1
print(t)
```

**A3** aigcodeset_005264 sim=0.7493 same problem gemini/fix_wrong_answer
```python
N = int(input())
S = list(input())
cnt = 0

for i in range(N-2):
  for j in range(i+1, N-1):
    for k in range(j+1, N):
      if S[i] != S[j] and S[i] != S[k] and S[j] != S[k] and j-i != k-j:
        cnt += 1
        
print(cnt)
```

## Query 10  (true: AI, problem p02829, found p03573 WRONG)
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

**H1** aigcodeset_002765 sim=0.7762 same problem human/human
```python
a,b,c=map(int,input().split())
if a==b:
    print(c)
elif b==c:
  print(a)
else:
  print(b)
```

**H2** aigcodeset_000555 sim=0.7727 same problem human/human
```python
a, b, c = map(int, input().split())
if a == b:
    print(c)
elif a == c:
    print(b)
else:
    print(a)
```

**H3** aigcodeset_003741 sim=0.7727 same problem human/human
```python
A,B,C = map(int,input().split())
if A==B:
  print(C)
elif A==C:
  print(B)
else:
  print(A)
```

**A1** aigcodeset_007153 sim=0.8353 same problem gemini/from_scratch
```python
if A == B:
  print(C)
elif A == C:
  print(B)
else:
  print(A)
```

**A2** aigcodeset_007262 sim=0.6933 same problem codestral/fix_wrong_answer
```python
 def solve(A: int, B: int, C: int) -> int:
        if A == B:
            return C
        elif A == C:
            return B
        else:
            return A

    A, B, C = map(int, input().split())
    print(solve(A, B, C))
```

**A3** aigcodeset_005394 sim=0.4964 same problem codellama/from_scratch
```python
def find_different(a, b, c):
    if a == b:
        return c
    elif b == c:
        return a
    else:
        return b
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

**H1** aigcodeset_004248 sim=0.9591 same problem human/human
```python
N,A,B = map(int,input().split())
X = list(map(int,input().split()))

ans = 0
t = 1
for i in range(len(X)-1):
    ans += min((X[i+1] - X[i])*A,B)
print(ans)
```

**H2** aigcodeset_000987 sim=0.8764 same problem human/human
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

**H3** aigcodeset_004721 sim=0.7524 same problem human/human
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

**A1** aigcodeset_005223 sim=0.9176 same problem gemini/fix_runtime_error
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

**A2** aigcodeset_004921 sim=0.8821 same problem codellama/fix_runtime_error
```python
n, a, b = map(int, input().split())
ans = 0
for i in range(n - 1):
    ans += max(a * (x[i + 1] - x[i]), b)
print(ans)
```

**A3** aigcodeset_004951 sim=0.8018 same problem codestral/fix_runtime_error
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

**H1** aigcodeset_003839 sim=0.3274 same problem human/human
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

**H2** aigcodeset_001012 sim=0.3195 same problem human/human
```python
def LI():
  return [int(s) for s in input().split()]

N = int(input())

A_s = LI()
B_s = LI()

counter = 0

for i in range(N):
  if A_s[i] > B_s[i]:
    counter+=B_s[i]
  elif B_s[i] < A_s[i] + A_s[i+1]:
    counter+=B_s[i]
    A_s[i+1] = A_s[i+1] - (B_s[i]-A_s[i])
  else:
    counter+=A_s[i]+A_s[i+1]
    A_s[i+1] = 0
    
print(counter)
```

**H3** aigcodeset_000305 sim=0.3077 same problem human/human
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

**A1** aigcodeset_006262 sim=0.6016 same problem codestral/from_scratch
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

**A2** aigcodeset_005034 sim=0.51 same problem gemini/from_scratch
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

**A3** aigcodeset_005174 sim=0.4375 same problem gemini/fix_wrong_answer
```python
def main():
    n = int(input())
    a_s = list(map(int, input().split(' ')))
    b_s = list(map(int, input().split(' ')))
    count = 0
    _a = a_s[n]
    for i, a in enumerate(reversed(a_s[:n])):
        s = _a + a
        b = b_s[n-(i+1)]
        count += min(s, b)
        # print(count)
        if b <= s:
            _a = a
        _a = max(0, s-_a)
    print(count)

if __name__ == '__main__':
    main()
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

**H1** aigcodeset_003255 sim=0.7389 same problem human/human
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

**H2** aigcodeset_003882 sim=0.7386 same problem human/human
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

**H3** aigcodeset_000548 sim=0.6999 same problem human/human
```python
n = int(input())
li = list(map(int,input().split()))

def gcd(a,b):
  while b!=0:
    a,b=b,a%b
  return a

ans = [0]*n
lileft = [0]*(n+1)
liright = [0]*(n+1)
for i in range(n):
  lileft[i+1] = gcd(lileft[i],li[i])
for i in range(n):
  liright[i+1] = gcd(liright[i],li[n-i-1])
for i in range(n):
  ans[i] = gcd(lileft[i],liright[n-i-1])
print(max(ans))
```

**A1** aigcodeset_005679 sim=0.8943 same problem codestral/fix_runtime_error
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

**A2** aigcodeset_006457 sim=0.7999 same problem codellama/from_scratch
```python
  def get_max_gcd(numbers):
    max_gcd = 0
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            gcd = math.gcd(numbers[i], numbers[j])
            if gcd > max_gcd:
                max_gcd = gcd
    return max_gcd

def main():
    n = int(input())
    numbers = list(map(int, input().split()))
    max_gcd = get_max_gcd(numbers)
    print(max_gcd)

if __name__ == "__main__":
    main()
```

**A3** aigcodeset_007152 sim=0.7983 same problem codestral/from_scratch
```python

from math import gcd
from functools import reduce

# Get the number of elements
n = int(input())

# Get the list of elements
elements = list(map(int, input().split()))

# Calculate the current gcd of the elements
current_gcd = reduce(gcd, elements)

# Calculate the maximum possible gcd
max_possible_gcd = max(current_gcd, max(elements) + current_gcd)

# Print the maximum possible gcd
print(max_possible_gcd)
```

## Query 14  (true: AI, problem p03274, found p03274 OK)
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

**H1** aigcodeset_001639 sim=0.6828 same problem human/human
```python
# ABC107C - Candles (ARC101C)
def main():
    N, K, *X = map(int, open(0).read().split())
    ans = min(r - l + min(abs(l), abs(r)) for l, r in zip(X, X[K - 1 :]))
    print(ans)


if __name__ == "__main__":
    main()
```

**H2** aigcodeset_003074 sim=0.534 same problem human/human
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

**H3** aigcodeset_002751 sim=0.5161 same problem human/human
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

**A1** aigcodeset_004918 sim=0.7657 same problem codestral/from_scratch
```python
# Read input
n, k = map(int, input().split())
x = list(map(int, input().split()))

# Sort the coordinates
x.sort()

# Initialize left and right pointers
left, right = 0, n - 1

# Initialize minimum time
min_time = 0

# Light k candles
for _ in range(k):
    # If the distance to the left candle is less than or equal to the distance to the right candle
    if abs(x[left]) <= abs(x[right]):
        # Add the distance to the left candle to the minimum time
        min_time += abs(x[left])
        # Move the left pointer to the right
        left += 1
    # If the distance to the right candle is le
```

**A2** aigcodeset_006472 sim=0.7537 same problem codellama/from_scratch
```python
N = int(input())
K = int(input())
x = list(map(int, input().split()))

def light_candles(x, K):
    time = 0
    for i in range(K):
        time += abs(x[i] - x[i-1])
    return time

print(light_candles(x, K))
```

**A3** aigcodeset_005769 sim=0.5352 same problem codellama/fix_runtime_error
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

**H1** aigcodeset_000712 sim=0.8286 same problem human/human
```python
import sys
input = sys.stdin.readline

mod = 10**9+7
n,m= map(int, input().split())
s= list(map(int, input().split()))
t= list(map(int, input().split()))

# dp[i][j]:nからi番目までとtからj番目まで取ったときの共通部分整数列の組み合わせ
dp=[[1]*(m+1) for i in range(n+1)]

for i in range(n):
    for j in range(m):
        if s[i]==t[j]:
            dp[i+1][j+1]=dp[i+1][j]+dp[i][j+1]
        else:
            dp[i+1][j+1]=dp[i+1][j]+dp[i][j+1]-dp[i][j]
        dp[i+1][j+1]%=mod

print(dp[n][m])
```

**H2** aigcodeset_001959 sim=0.8168 same problem human/human
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

**H3** aigcodeset_002782 sim=0.7525 same problem human/human
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

**A1** aigcodeset_005312 sim=0.9575 same problem gemini/fix_runtime_error
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

**A2** aigcodeset_005132 sim=0.8385 same problem codellama/fix_runtime_error
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

**A3** aigcodeset_007018 sim=0.8077 same problem codestral/fix_wrong_answer
```python
import sys

def main():
    MOD = 10**9 + 7
    LI = lambda : [int(x) for x in sys.stdin.readline().split()]

    N,M = LI()
    S = LI()
    T = LI()

    dp = [[0]*(M+1) for _ in range(N+1)]
    dp[0][0] = 1

    for i in range(1,N+1):
        for j in range(1,M+1):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
            if S[i-1] == T[j-1]:
                dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % MOD
    print(dp[-1][-1])

if __name__ == '__main__':
    main()
```