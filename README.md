
Generating unlabelled necklaces in CAT
======================================

This short document will describe an algorithm that can generate a 2 types of combinatorial objects in CAT time that currently have no know CAT, or even polynomial time algorithm [1][2]. These 2 objects are necklaces equivalent under modular addition, and most relevant unlabelled necklaces. Let's start by describing the algorithm for generating regular necklaces.

Necklaces
---------

A necklace is an equivalence class of strings under rotation. That is, all these strings are part of the same equivalent class (they are the same necklace):

> abacb, bacba, acbab, cbaba, babac

The amount of symbols in the alphabet is K, the length of the string is called N and the representivate of the equivalance class is the lexiographically lowest string. Our goal is to write an algorithm that generates all representative K-ary necklaces of length N. Ideally in CAT (Constant amortized time), that means that the average time to generate the next necklace is constant for any value for N or K.
Naïve implementation

A trivial way to find all necklaces is to recursively generate all K-ary strings of length N and compare each with its rotation. A back of the envelope calculation suggests that there are roughly N times as many strings as necklaces (for N rotations, except periodic ones) and we have to do ~N/2 checks on each of those. Even if the check is constant that gives us ~O(n²), terrible but a start.

```python
N = 4
K = 3

def is_representative(s):
	# Check all other rotations
	for r in range(1, N):
		# Rotate and check if lexiographically smaller than input
		if s[r:] + s[:r] &amp;lt; s:
			return False
	# If none was smaller s is representative of its eq. class
	return True

def necklaces(s = []):
	# Terminating condition, when N symbols are placed
	if len(s) == N:
		if is_representative(s):
			print(s)
		return

	# Otherwise place each possible symbol at this depth
	for i in range(0, K):
		s.append(i)
		necklaces(s)
		s.pop()

necklaces()
```

Recursive functions are prune-able trees
----------------------------------------

The first thing to realize is that recursive functions like these can be looked at as trees, and a fast algorithm prunes at the earliest moment possible using little resources to do so. The first step, pruning earlier is something we can do easily, a string might already be lexiographically smaller when rotated even when it’s incomplete. The string “10????” can never be representative, because its rotation by 1, “0????1”,  is already smaller.

```python
def potentially_representative(s):
	for r in range(1, len(s)):
		# shift and fill up with worst case
		if s[r:] + [K]*(len(s)-r) &amp;lt; s:
			return False

	return True

def necklaces(s = []):
	# Terminating condition, when N symbols are placed
	if len(s) == N:
		if is_representative(s):
			print(s)
		return

	# Prune small strings that can not become necklaces early
	if not potentially_representative(s):
		return

	# Otherwise place each possible at this depth
	for i in range(0, K):
		s.append(i)
		necklaces(s)
		s.pop()

necklaces()
```

In general you always have to think about what the earliest possible point of exit is, where in the tree can you be sure that none of your children will be relevant. This is much faster, but our prune-check is still very expensive.

Equality under rotation means periodicity
-----------------------------------------

We now prune a string as soon as a rotation is smaller. That means that up until that point the rotation had been equal, had it been bigger, it can’t possible become smaller by appending more symbols.

When a string is equal to its rotation it has a period equal to that rotation. Since a string has one lowest relevant periodicity, we only have to check that rotation, the periodicity of the string. It then quickly follows that if we keep track of the periodicity of the string, and only place symbols that either are bigger than the periodicity suggest (creating a new period the length of the string) or follow the period we can completely remove potentially_representative.

```python
def necklaces(s, period = 1):
	# Terminating condition, when N symbols are placed
	if len(s) == N:
		if is_representative(s):
			print(s)
		return

	# Follow period
	s.append(s[-period])
	necklaces(s, period)
	s.pop()

	# Create larger symbol that the periodicity suggest
	# This way we can never create a string that is smaller under rotation
	for i in range(s[-period]+1, K):
		s.append(i)
		necklaces(s, len(s))
		s.pop()

for i in range(0, K):
	necklaces([i])
```
 
Final result for generating necklaces
-------------------------------------

When checking is_representative we now know the periodicity and we can make it much faster, since we only have to check the cases in which the period is not a divisor of the length of the string:

```python
N = 10
K = 3

def necklaces(s, period = 1):
	# Terminating condition, when N symbols are placed
	if len(s) == N:
		# When the period divides the string length we are done, otherwise check
		if len(s) % period == 0 or s &lt;= s[period:] + s[:period]:
			print(s)
		return

	# Follow period
	s.append(s[-period])
	necklaces(s, period)
	s.pop()

	# Create larger symbol that the periodicity suggest
	# This way we can never create a string that is smaller under rotation
	for i in range(s[-period]+1, K):
		s.append(i)
		# This new string breaks the old period and therefor has a period the length of the string
		necklaces(s, len(s))
		s.pop()

for i in range(0, K):
	necklaces([i])
```


Alphabet rotation
=================

The motivation (and an example) for this symmetry is found in music. A melody can be expressed as a string of notes, in some cases a melody (and such a string) has 2 important symmetries, transposition and rotation. Taking care of the symmetry of rotation was already tackled. But transposition (rotation of the alphabet) has not. Let's start with a naïve solution for clarity

We can just generate all necklaces using the algorithm of the previous post and check which ones remain lexiographical minimal under alphabet and string rotation

```python	
def is_representative(s):
    for j in range(0, N):
        rotated = s[j:] + s[:j]
        # The best transposition is the one that makes the first symbol 0 at the current rotation
        # also note what alphabet rotation means, it is the modular addition of any value to each symbol of the string
        transposed = [(x-s[0]+K)%K for x in rotated]
        if transposed &lt; s:
            return False
    return True
```

Solving for a single alphabet rotation
--------------------------------------

The normal necklace algorithm could be seen as an algorithm that generates necklaces under alphabet rotation with only one rotation allowed (no rotation). At that point you can wonder about still allowing one rotation, but a different one, say the alphabet rotated by one (0 -> k-1, 1 -> 0, 2 -> 1, …, k-1->k-2). The only thing that we have to change about our algorithm is adding the alphabet rotation when we are placing a new symbol that follows the period. And including alphabet rotation when comparing a string with its transformation:

```python
N = 10
K = 3
 
# The alphabet rotation applied to our necklaces
KR = 1
 
# Given a string 'a' and a rotation r, alphabet rotation kr,
# returns true if a &lt;= t(a)
def less_eq_than(a, r, kr):
    for i in range(0, len(a)):
        transformed = (a[(i+r)%len(a)] + kr) % K
        if a[i] &lt; transformed: return True         if a[i] &gt; transformed: return False
    return True
 
def necklaces(s, period = 1):
    # Terminating condition, when N symbols are placed
    if len(s) == N:
        # When the period divides the string length we are done, otherwise check
        if less_eq_than(s, period, KR):
            print(s)
        return
 
    # Follow period
    s.append((s[-period] + KR) % K)
    necklaces(s, period)
    s.pop()
 
    # Create larger symbol that the periodicity suggest
    # This way we can never create a string that is smaller under rotation
    for i in range(s[-period]+1, K):
        s.append((i+KR) % K)
        # This new string breaks the old period and therefor has a period the length of the string
        necklaces(s, len(s))
        s.pop()
 
for i in range(0, K):
    necklaces([i])
```

This gives some intuition about the how the string’s period works including alphabet rotation.

Generalizing to all rotations
-----------------------------

With alphabet rotations included we have a 2D transformation and therefor we can have multiple symmetries in our necklace (different co-prime periods). That means we need to keep track of any active (s = T(s)) period and before adding a symbol we find only the ones that do not place a lower valued symbol according to any active period.

```python	
# Try all possible symbols
for i in range(0, K):
	# If any of the periods would result in a lower lex string with this symbol, skip it.
    if any((i - s[p])%K &lt; s[-p] for p in periods):
        continue
	# Check which of the current periods are still active in the new string (which might break some periodicity)
    active_periods = [len(s)] + [p for p in periods if (i - s[p]) % K == s[-p]]
    s.append(i)
    necklaces(s, active_periods)
    s.pop()
```

It follows that when the string is of the desired length we only have to worry about the active periods and verify those, the complete code:

```python
N = 10
K = 3
 
# This can be optimized by checking symbol for symbol and early exiting, but this is slightly clearer
def is_representative(s, periods):
    for j in periods:
        rotated = s[j:] + s[:j]
        # The best transposition is the one that makes the first symbol 0 at the current rotation
        transposed = [(x-rotated[0]+K)%K for x in rotated]
        if transposed &lt; s:
            return False
    return True
 
def necklaces(s, periods = []):
    # Terminating condition, when N symbols are placed
    if len(s) == N:
        # When the period divides the string length we are done, otherwise check
        if is_representative(s, periods):
            print(s)
        return
 
    # Create larger symbol that any of the primitive periods suggest
    # This way we can never create a string that is smaller under string rotation and alphabet rotation
    for i in range(0, K):
        if any((i - s[p])%K &lt; s[-p] for p in periods):
            continue
 
        active_periods = [len(s)] + [p for p in periods if (i - s[p]) % K == s[-p]]
        s.append(i)
        necklaces(s, active_periods)
        s.pop()
 
necklaces([0])
```

Note that we currently effectively copy the periods list, but it can also be implemented as a stack.


Unlabelled necklaces
====================

And now I will finally describe an algorithm that generates unlabeled necklaces, that is, any permutation of the alphabet is allowed. This is in fact an open problem[1][2] with no reasonable algorithm known. My proof of CAT is incomplete but I do have some pretty graphs and it is certainly polynomial amortized time.

The basic goal is to keep track of the lexiographically lowest mapping of the alphabet for each primitive period of the string being build. In O(1) amortized.

That’s a whole mouthful, let’s start with figuring out the permutation of the alphabet that creates the lexiographically lowest string given an input string. The only thing you have to is go over each character of your input string while mapping each character’s first occurance to the lowest available value. Python example:

```python
# Generates the lowest string possible from s by permutating the alphabet
def optimal_mapping(s):
	mapping = {}
	output = []
	for c in s:
		if not c in mapping:
			mapping[c] = len(mapping)

		output.append(mapping[c])
	return ouput
```

You can see how we can keep track of the optimal permutation of a string as we are building it, now just as with the alphabet rotations in part 2 we simply keep track of this for each rotation under which the string does not change by applying this function. And we don’t allow placing any symbols that would make any rotation lower lexiographically.

First here is a data structure that allows pushing/popping characters while keeping track of the optimal alphabet permutation. It also allows comparison of the last placed symbol (in the optimal mapping) with the matching symbol in the original string assuming some rotation r:

```python
# Constants for ordinality of strings
EQUAL = 0
TRANSFORMED_LESS = 1
ORIGINAL_LESS = 2

class Mapping:
	UNUSED = -1
	def __init__(self, K, r):
		# The rotation that this instance is mapping, only used for comparison
		self.r = r
		# mapping contains the permutation of the alphabet, s -> t(s)
		self.mapping = [self.UNUSED] * K
		# how often does each characters (from s, not t(s)) occur
		self.references = [0] * K
		# What is the lowest available symbol we can map to
		self.least_open_symbol = 0

	def push(self, x):
		# Map to lowest possible new symbol if we never saw x before
		if self.mapping[x] == self.UNUSED:
			self.mapping[x] = self.least_open_symbol
			self.least_open_symbol += 1

		self.references[x] += 1

	def pop(self, x):
		self.references[x] -= 1
		# If the occurance is 0 free up this symbol in the mapping
		if self.references[x] == 0:
			self.mapping[x] = self.UNUSED
			self.least_open_symbol -= 1

	# Assuming the characters s[r:] were pushed.
	# What is lexiographical ordering of the remapped rotated string versus 's'.
	def compare(self, s):
		if s[-(self.r+1)] > self.mapping[s[-1]]: return TRANSFORMED_LESS
		if s[-(self.r+1)] < self.mapping[s[-1]]: return ORIGINAL_LESS
		return EQUAL
```

Next we need a way to keep track of which rotations of the string are still equal to it. Only those Mapping instances need to receive new symbols. Since we might remove multiple rotations per pushed symbol we need a data structure that allows adding rotations, removing them, and reverting to a marked state. That is reasonably simple:

```python
# Keep track of a set of unique numbers
# Remembers operations and allows reverting to a previously marked state
class ActivePeriods:
	def __init__(self):
		self.values = set()
		# List of changes to values
		self.changes = []
		# List of places we should revert to
		self.markings = []

	def begin(self):
		self.markings.append(len(self.changes))

	def add(self, x):
		self.values.add(x)
		self.changes.append(('A', x))

	# Remove any rotations that became invalid
	def remove(self, x):
		self.values.remove(x)
		self.changes.append(('R', x))

	# Undo all removals/additions since last begin call
	def end(self):
		for i in range(0, len(self.changes) - self.markings.pop()):
			x = self.changes.pop()
			if x[0] == 'A':
				self.values.remove(x[1])
			else:
				self.values.add(x[1])
```

Now that we have some convenient data structures we get to the actual complexity, we want a third data structure that allows pushing/popping symbols and that tells us whether there is any r for which s[r:] < s[:-r], in which case we should prune that branch. It does this by creating a new mapping for each new symbol that gets added (because there is a new value for r). and not updating a mapping when s[r:] > s[:-r].

```python
# For each rotation of the pushed symbols, keep track of optimal mapping (see above)
# And when a mapping is worse that the original, remove that rotation as relevant
# When pushing a symbol tells you if it will result in lower lex. in any rotation
class Periodicity:
	def __init__(self, N, K):
		self.mappings = [Mapping(K, i) for i in range(0, N)]
		self.active_periods = ActivePeriods()
		self.s = []

	def push(self, x):
		# Add symbol to string
		self.s.append(x)
		# Add new period(eg: rotation)
		self.active_periods.add(len(self.s)-1)
		# Apply symbol to all mappings of all open periods
		for p in self.active_periods.values:
			self.mappings[p].push(x)

		# Before removing all periods that have become invalid, mark this state so that we can move back
		self.active_periods.begin()

		# Check if any active period is lex. lower than the string, if so we stop recursing
		if any(self.mappings[p].compare(self.s) == TRANSFORMED_LESS for p in self.active_periods.values):
			self.pop()
			return False

		# otherwise remove any periods that have become higher lex
		for p in set(self.active_periods.values):
			if self.mappings[p].compare(self.s) == ORIGINAL_LESS:
				self.active_periods.remove(p)

		return True

	def pop(self):
		self.active_periods.end()
		for p in self.active_periods.values:
			self.mappings[p].pop(self.s[-1])

		self.active_periods.remove(len(self.s)-1)
		x = self.s.pop()
```

At this point we can make a simple recursive function. But there is one problem, the above data structure only tells us that there are now some rotations ‘r’ for which s[r:] == s[:-r], but it tells us nothing about s[:r] vs s[-r:]. It doesn’t “wrap around”. So we need to complete the wrap around when we have placed N symbols:

```python
# Example values
K = 3
N = 7
periods = Periodicity(N, K)
# If we know what rotated (but truncated) strings are equal lex
# figure out if s is truly representative
def is_representative(s):
	for p in periods.active_periods.values:
		for i in range(0, p):
			periods.mappings[p].push(s[i])
			a = periods.mappings[p].mapping[s[i]]

			if a != s[i-p]:
				for j in range(i, -1, -1):
					periods.mappings[p].pop(s[j])

			if a < s[i-p]: return False 			if a > s[i-p]: break
		else:
			for j in range(p-1, -1, -1):
				periods.mappings[p].pop(s[j])

			# Is this string periodic?
			# if not p == 0:
			# 	return False

	return True

# This is a simple DFS that prunes as Periodicity commands
def necklaces():
	if len(periods.s) == N:
		if is_representative(periods.s):
			print("".join([str(x) for x in periods.s]))
	else:
		for i in range(0, K):
			if periods.push(i):
				necklaces()
				periods.pop()

necklaces()
```

And that will work! For performance characteristics I can only say that the operations per handled period is O(1). And I strongly suspect that the amount of active periods does not grow as N or K grows. Which makes the algorithm O(1) amortized. I have no proof unfortunately, only a pretty graphs of runtime with respect to N & K:
![](/graph.png "")

This is from a C implementation, it was ran once on a 2.4GHZ Nehalem C architecture CPU.

C implementation:
```c
// Tested with:
// gcc -march=native -O3 -o unlabelled unlabelled.cpp 
// Run like (eg: N=10, K=4):
// ./unlabelled 10 4
#include <time.h>
#include <stdio.h>		
#include <stdint.h>
#include <stdlib.h>
//Enable this to execute the slow naive algorithms as well and see whether they have the same results
//#define CHECKSUM
//Print all the necklaces, don't enable when testing running time.
//#define PRINT

//Max value of N
#define MAX 64

//Contains the current optimal mapping and required symbols for each rotation of a necklace
typedef struct
{
	char mapping[MAX][MAX];
	char maxk[MAX];
} Mapping;

int N;
int K;

int t;
int k;

//Meta info for performance and verification checks:
//Contains the "operations" count, incremented at every loop iteration and recursion
uint64_t opCount = 0;
//Counts the amount of generated necklaces
uint64_t counter = 0;
//A simple XOR of all necklaces for verification
uint64_t checksum = 0;

//Contains result
char a[MAX];
uint64_t periods;
Mapping m;

void Print(int t)
{
	char buffer[N+2];
	for(int i = 1; i <= t; i++)
		buffer[i] = a[i] + '0';

	buffer[t+1] = '\0';
	printf("%s\n", &buffer[1]);
}

inline void addToChecksum()
{
	uint64_t simplified = 0;
	for(int i = 1; i <= N; i++)
		simplified |= a[i] << ((i*4) % 64); 

	checksum ^= simplified;
}

void representative()
{
	#ifdef PRINT
	Print(N);
	#endif
	#ifdef CHECKSUM
	addToChecksum();
	#endif
	counter++;
}

//0 = equal
//1 = original wins
//2 = rotation wins
//Keeps track of the optimal mapping under each rotation and returns wether a or a rotated by tp is lower lex at t
int stillValid(int t, int tp, Mapping* m)
{
	int original = a[(t-tp - 1 + N) % N + 1];
	int rotation = a[t];

	if(m->mapping[tp][rotation] == -1)
	{
		if(original > m->maxk[tp]) return 2;
		if(original < m->maxk[tp]) return 1;
		m->mapping[tp][rotation] = m->maxk[tp]++;
		return -1;
	}

	rotation = m->mapping[tp][rotation];
	if(original > rotation) return 2;
	if(original < rotation) return 1;
	return 0;
}

//Generates all representatives of the equivalance class of unlabelled k-ary necklaces of length N 
void GenU()
{
	++opCount;
	if(t > N)
	{
		//All the rotations that are still competing are just checked from N-rotation up till N, wrap around and see if any win
		int tp;
		for(uint64_t i = periods; i; i &= 0xffffffffffffffff ^ (1LL << tp) )
		{
			++opCount;

			tp = __builtin_ctzl(i);
			int j;

			uint64_t edited_mappings = 0;
			int lowerlex = 0;
			for(j = 1; j <= tp; j++)
			{
				++opCount;
				int result = stillValid(j, tp, &m);
				if(result == -1)
				{
					edited_mappings |= 1LL << j;
					continue;
				}
				if(result == 0) continue;
				if(result == 1) break;
				if(result == 2)
				{
					lowerlex = 1;
					break;
				}
			}
			
			int _t;
			for(uint64_t i = edited_mappings; i; i &= 0xffffffffffffffff ^ (1LL << _t) )
			{
				++opCount;
				_t = __builtin_ctzl(i);
				m.mapping[tp][a[_t]] = -1;
				m.maxk[tp] -= 1;
			}

			// Periodic?
			if(lowerlex)
				return;
			// use this for unlabelled lyndon words, a little bonus
			//if( (j == tp+1 && tp != 0) || lowerlex )
			//	return;
		}

		representative();
	}
	else
	{
		//Already add the extra rotation to keep track of
		periods |= (1LL << (t-1));
		//Consider each symbol at most one bigger than the biggest one placed so far
		for(a[t] = 0; a[t] <= k+1 && a[t] < K; a[t]++)
		{
			++opCount;
			int tp;
		
			uint64_t removed_periods = 0;
			uint64_t edited_mappings = 0;
			//Consider each rotation that is still equal under optimal mapping
			for(uint64_t i = periods; i; i &= 0xffffffffffffffff ^ (1LL << tp) )
			{
				++opCount;
				tp = __builtin_ctzl(i);
				int result = stillValid(t, tp, &m);
				
				if(result == -1)
				{
					edited_mappings |= 1LL << tp;
				}

				if(result == 1) 
				{
					periods -= (1LL << tp);
					removed_periods += (1LL << tp);
				}

				//Skip this symbol as an option because rotation tp is lower lex.
				if(result == 2) goto SKIP_J;
			}

			int swap = k;
			k += (a[t] == k + 1);
			t += 1; GenU(); t -= 1;
			k = swap;

			SKIP_J:;
			for(uint64_t i = removed_periods; i; i &= 0xffffffffffffffff ^ (1LL << tp) )
			{
				++opCount;
				tp = __builtin_ctzl(i);
				periods += 1LL << tp;
			}
			for(uint64_t i = edited_mappings; i; i &= 0xffffffffffffffff ^ (1LL << tp) )
			{
				++opCount;
				tp = __builtin_ctzl(i);
				m.mapping[tp][a[t]] = -1;
				--m.maxk[tp];
			}
		}
		periods -= periods & (1LL << (t-1));
	}
}

//Naive unlabelled implementation
int isUnlabelled(int n)
{
	int mapping[K];
	for(int k = 0; k < K; k++) mapping[k] = -1;
	int maxk = 0;

	for(int i = 1; i <= N; i++)
	{
		int X = a[i];
		int Y = a[(i-1+n) % N + 1];
		if(mapping[Y] == -1)
			mapping[Y] = maxk++;

		Y = mapping[Y];

		if(X > Y)
			return -1;

		if(X < Y)
			return 1;
	}

	return 0;
}
void GenUNaive(int t, int p)
{
	if( t > N && N % p == 0)
	{
		if(isUnlabelled(0) < 0)
			return;

		for(int i = N-1; i > 0; i--)
			if(isUnlabelled(i) <= 0)
				return;

		representative();
	}

	if(t <= N)
	{
		a[t] = a[t-p];
		GenUNaive(t+1, p);

		if(t > 1)
		{
			for(int j= a[t-p] + 1; j <= K-1; j++)
			{
				a[t] = j;
				GenUNaive(t+1, t);
			}
		}
	}
}

void init()
{
	checksum = 0;
	counter = 0;
	opCount = 0;
	
	periods = 1;
	t = 2;
	k = 0;

	//Initializing mapping to know values simplifies the code above
	//Mapping m;
	for(int i = 0; i < MAX; i++)
		a[i] = 0;

	for(int i = 0; i < MAX; i++)
		for(int j = 0; j < MAX; j++)
			m.mapping[i][j] = -1;

	for(int j = 0; j < MAX; j++)
		m.maxk[j] = 0;

	m.mapping[0][0] = 0;
	m.maxk[0] = 1;
}

void computeRange(int mink, int maxk, int minn, int maxn)
{
	if(mink < 2) mink = 2;
	for(K = mink; K <= maxk; ++K)
	{
		printf("#K=%d\n", K);
		for(N = minn; N <= maxn; ++N)
		{
			init();
			clock_t start, end;
			start = clock();
			GenU(2,0);
			end = clock();
			printf("%d %f %f %llu\n", N, ((double)(opCount))/counter, ((double)(end-start))/counter, counter);
		}
	}
}

int main(int argc, char* argv[])
{
	setbuf(stdout, NULL);

	if(argv[1][0] == 'r')
	{
		computeRange(atoi(argv[4]), atoi(argv[5]), atoi(argv[2]), atoi(argv[3]));
		return 0;
	}

	N = atoi(argv[1]);
	K = atoi(argv[2]);

	#ifdef CHECKSUM
	init();
	GenUNaive(2,1);
	printf("N=%d K=%d necklaces=%llu checksum=%llu\n", N, K, counter, checksum);
	#endif

	init();
	clock_t start, end;
	start = clock();
	GenU();
	end = clock();

	#ifdef CHECKSUM
	printf("N=%d K=%d counter=%llu time=%llu checksum=%d time/counter=%f OPS/counter=%d\n", N, K, counter, (end-start),  checksum, ((double)(end-start))/counter, opCount/counter);
	#else
	printf("N=%d K=%d counter=%llu time=%llu time/counter=%f OPS/counter=%d\n", N, K, counter, (end-start), ((double)(end-start))/counter, opCount/counter );
	#endif
}
```


[1] P. Flener and J. Pearson / Solving necklace constraint problems (2008)
Future work includes the quest for a constant-amortized-time enumeration-algorithm for unlabelled k-ary necklaces.

[2] K. Cattell / Fast Algorithms to Generate Necklaces, Unlabeled Necklaces, and Irreducible Polynomials over GF 2 (1998)
It remains an interesting challenge to extend these ideas to generate
unlabeled necklaces over nonbinary alphabets
