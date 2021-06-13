// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file.

package floats

import (
	"errors"
	"math"
	"sort"

	"gonum.org/v1/gonum/floats/scalar"
	"gonum.org/v1/gonum/internal/asm/f64"
)

const (
	zeroLength   = "floats: zero length slice"
	shortSpan    = "floats: slice length less than 2"
	badLength    = "floats: slice lengths do not match"
	badDstLength = "floats: destination slice length does not match input"
)

// Add adds, element-wise, the elements of s and dst, and stores the result in dst.
// It panics if the argument lengths do not match.
func Add(dst, s []float64) {
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	f64.AxpyUnitaryTo(dst, 1, s, dst)
}

// AddTo adds, element-wise, the elements of s and t and
// stores the result in dst.
// It panics if the argument lengths do not match.
func AddTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic(badLength)
	}
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	f64.AxpyUnitaryTo(dst, 1, s, t)
	return dst
}

// AddConst adds the scalar c to all of the values in dst.
func AddConst(c float64, dst []float64) {
	f64.AddConst(c, dst)
}

// AddScaled performs dst = dst + alpha * s.
// It panics if the slice argument lengths do not match.
func AddScaled(dst []float64, alpha float64, s []float64) {
	if len(dst) != len(s) {
		panic(badLength)
	}
	f64.AxpyUnitaryTo(dst, alpha, s, dst)
}

// AddScaledTo performs dst = y + alpha * s, where alpha is a scalar,
// and dst, y and s are all slices.
// It panics if the slice argument lengths do not match.
//
// At the return of the function, dst[i] = y[i] + alpha * s[i]
func AddScaledTo(dst, y []float64, alpha float64, s []float64) []float64 {
	if len(s) != len(y) {
		panic(badLength)
	}
	if len(dst) != len(y) {
		panic(badDstLength)
	}
	f64.AxpyUnitaryTo(dst, alpha, s, y)
	return dst
}

// argsort is a helper that implements sort.Interface, as used by
// Argsort.
type argsort struct {
	s    []float64
	inds []int
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Less(i, j int) bool {
	return a.s[i] < a.s[j]
}

func (a argsort) Swap(i, j int) {
	a.s[i], a.s[j] = a.s[j], a.s[i]
	a.inds[i], a.inds[j] = a.inds[j], a.inds[i]
}

// Argsort sorts the elements of dst while tracking their original order.
// At the conclusion of Argsort, dst will contain the original elements of dst
// but sorted in increasing order, and inds will contain the original position
// of the elements in the slice such that dst[i] = origDst[inds[i]].
// It panics if the argument lengths do not match.
func Argsort(dst []float64, inds []int) {
	if len(dst) != len(inds) {
		panic(badDstLength)
	}
	for i := range dst {
		inds[i] = i
	}

	a := argsort{s: dst, inds: inds}
	sort.Sort(a)
}

// Count applies the function f to every element of s and returns the number
// of times the function returned true.
func Count(f func(float64) bool, s []float64) int {
	var n int
	for _, val := range s {
		if f(val) {
			n++
		}
	}
	return n
}

// CumProd finds the cumulative product of the first i elements in
// s and puts them in place into the ith element of the
// destination dst.
// It panics if the argument lengths do not match.
//
// At the return of the function, dst[i] = s[i] * s[i-1] * s[i-2] * ...
func CumProd(dst, s []float64) []float64 {
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	if len(dst) == 0 {
		return dst
	}
	return f64.CumProd(dst, s)
}

// CumSum finds the cumulative sum of the first i elements in
// s and puts them in place into the ith element of the
// destination dst.
// It panics if the argument lengths do not match.
//
// At the return of the function, dst[i] = s[i] + s[i-1] + s[i-2] + ...
func CumSum(dst, s []float64) []float64 {
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	if len(dst) == 0 {
		return dst
	}
	return f64.CumSum(dst, s)
}

// Distance computes the L-norm of s - t. See Norm for special cases.
// It panics if the slice argument lengths do not match.
func Distance(s, t []float64, L float64) float64 {
	if len(s) != len(t) {
		panic(badLength)
	}
	if len(s) == 0 {
		return 0
	}
	if L == 2 {
		return f64.L2DistanceUnitary(s, t)
	}
	var norm float64
	if L == 1 {
		for i, v := range s {
			norm += math.Abs(t[i] - v)
		}
		return norm
	}
	if math.IsInf(L, 1) {
		for i, v := range s {
			absDiff := math.Abs(t[i] - v)
			if absDiff > norm {
				norm = absDiff
			}
		}
		return norm
	}
	for i, v := range s {
		norm += math.Pow(math.Abs(t[i]-v), L)
	}
	return math.Pow(norm, 1/L)
}

// Div performs element-wise division dst / s
// and stores the value in dst.
// It panics if the argument lengths do not match.
func Div(dst, s []float64) {
	if len(dst) != len(s) {
		panic(badLength)
	}
	f64.Div(dst, s)
}

// DivTo performs element-wise division s / t
// and stores the value in dst.
// It panics if the argument lengths do not match.
func DivTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic(badLength)
	}
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	return f64.DivTo(dst, s, t)
}

// Dot computes the dot product of s1 and s2, i.e.
// sum_{i = 1}^N s1[i]*s2[i].
// It panics if the argument lengths do not match.
func Dot(s1, s2 []float64) float64 {
	if len(s1) != len(s2) {
		panic(badLength)
	}
	return f64.DotUnitary(s1, s2)
}

// Equal returns true when the slices have equal lengths and
// all elements are numerically identical.
func Equal(s1, s2 []float64) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, val := range s1 {
		if s2[i] != val {
			return false
		}
	}
	return true
}

// EqualApprox returns true when the slices have equal lengths and
// all element pairs have an absolute tolerance less than tol or a
// relative tolerance less than tol.
func EqualApprox(s1, s2 []float64, tol float64) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, a := range s1 {
		if !scalar.EqualWithinAbsOrRel(a, s2[i], tol, tol) {
			return false
		}
	}
	return true
}

// EqualFunc returns true when the slices have the same lengths
// and the function returns true for all element pairs.
func EqualFunc(s1, s2 []float64, f func(float64, float64) bool) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, val := range s1 {
		if !f(val, s2[i]) {
			return false
		}
	}
	return true
}

// EqualLengths returns true when all of the slices have equal length,
// and false otherwise. It also returns true when there are no input slices.
func EqualLengths(slices ...[]float64) bool {
	// This length check is needed: http://play.golang.org/p/sdty6YiLhM
	if len(slices) == 0 {
		return true
	}
	l := len(slices[0])
	for i := 1; i < len(slices); i++ {
		if len(slices[i]) != l {
			return false
		}
	}
	return true
}

// Find applies f to every element of s and returns the indices of the first
// k elements for which the f returns true, or all such elements
// if k < 0.
// Find will reslice inds to have 0 length, and will append
// found indices to inds.
// If k > 0 and there are fewer than k elements in s satisfying f,
// all of the found elements will be returned along with an error.
// At the return of the function, the input inds will be in an undetermined state.
func Find(inds []int, f func(float64) bool, s []float64, k int) ([]int, error) {
	// inds is also returned to allow for calling with nil.

	// Reslice inds to have zero length.
	inds = inds[:0]

	// If zero elements requested, can just return.
	if k == 0 {
		return inds, nil
	}

	// If k < 0, return all of the found indices.
	if k < 0 {
		for i, val := range s {
			if f(val) {
				inds = append(inds, i)
			}
		}
		return inds, nil
	}

	// Otherwise, find the first k elements.
	nFound := 0
	for i, val := range s {
		if f(val) {
			inds = append(inds, i)
			nFound++
			if nFound == k {
				return inds, nil
			}
		}
	}
	// Finished iterating over the loop, which means k elements were not found.
	return inds, errors.New("floats: insufficient elements found")
}

// HasNaN returns true when the slice s has any values that are NaN and false
// otherwise.
func HasNaN(s []float64) bool {
	for _, v := range s {
		if math.IsNaN(v) {
			return true
		}
	}
	return false
}

// LogSpan returns a set of n equally spaced points in log space between,
// l and u where N is equal to len(dst). The first element of the
// resulting dst will be l and the final element of dst will be u.
// It panics if the length of dst is less than 2.
// Note that this call will return NaNs if either l or u are negative, and
// will return all zeros if l or u is zero.
// Also returns the mutated slice dst, so that it can be used in range, like:
//
//     for i, x := range LogSpan(dst, l, u) { ... }
func LogSpan(dst []float64, l, u float64) []float64 {
	Span(dst, math.Log(l), math.Log(u))
	for i := range dst {
		dst[i] = math.Exp(dst[i])
	}
	return dst
}

// LogSumExp returns the log of the sum of the exponentials of the values in s.
// Panics if s is an empty slice.
func LogSumExp(s []float64) float64 {
	// Want to do this in a numerically stable way which avoids
	// overflow and underflow
	// First, find the maximum value in the slice.
	maxval := Max(s)
	if math.IsInf(maxval, 0) {
		// If it's infinity either way, the logsumexp will be infinity as well
		// returning now avoids NaNs
		return maxval
	}
	var lse float64
	// Compute the sumexp part
	for _, val := range s {
		lse += math.Exp(val - maxval)
	}
	// Take the log and add back on the constant taken out
	return math.Log(lse) + maxval
}

// Max returns the maximum value in the input slice. If the slice is empty, Max will panic.
func Max(s []float64) float64 {
	return s[MaxIdx(s)]
}

// MaxIdx returns the index of the maximum value in the input slice. If several
// entries have the maximum value, the first such index is returned.
// It panics if s is zero length.
func MaxIdx(s []float64) int {
	if len(s) == 0 {
		panic(zeroLength)
	}
	max := math.NaN()
	var ind int
	for i, v := range s {
		if math.IsNaN(v) {
			continue
		}
		if v > max || math.IsNaN(max) {
			max = v
			ind = i
		}
	}
	return ind
}

// Min returns the minimum value in the input slice.
// It panics if s is zero length.
func Min(s []float64) float64 {
	return s[MinIdx(s)]
}

// MinIdx returns the index of the minimum value in the input slice. If several
// entries have the minimum value, the first such index is returned.
// It panics if s is zero length.
func MinIdx(s []float64) int {
	if len(s) == 0 {
		panic(zeroLength)
	}
	min := math.NaN()
	var ind int
	for i, v := range s {
		if math.IsNaN(v) {
			continue
		}
		if v < min || math.IsNaN(min) {
			min = v
			ind = i
		}
	}
	return ind
}

// Mul performs element-wise multiplication between dst
// and s and stores the value in dst.
// It panics if the argument lengths do not match.
func Mul(dst, s []float64) {
	if len(dst) != len(s) {
		panic(badLength)
	}
	for i, val := range s {
		dst[i] *= val
	}
}

// MulTo performs element-wise multiplication between s
// and t and stores the value in dst.
// It panics if the argument lengths do not match.
func MulTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic(badLength)
	}
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	for i, val := range t {
		dst[i] = val * s[i]
	}
	return dst
}

// NearestIdx returns the index of the element in s
// whose value is nearest to v. If several such
// elements exist, the lowest index is returned.
// It panics if s is zero length.
func NearestIdx(s []float64, v float64) int {
	if len(s) == 0 {
		panic(zeroLength)
	}
	switch {
	case math.IsNaN(v):
		return 0
	case math.IsInf(v, 1):
		return MaxIdx(s)
	case math.IsInf(v, -1):
		return MinIdx(s)
	}
	var ind int
	dist := math.NaN()
	for i, val := range s {
		newDist := math.Abs(v - val)
		// A NaN distance will not be closer.
		if math.IsNaN(newDist) {
			continue
		}
		if newDist < dist || math.IsNaN(dist) {
			dist = newDist
			ind = i
		}
	}
	return ind
}

// NearestIdxForSpan return the index of a hypothetical vector created
// by Span with length n and bounds l and u whose value is closest
// to v. That is, NearestIdxForSpan(n, l, u, v) is equivalent to
// Nearest(Span(make([]float64, n),l,u),v) without an allocation.
// It panics if n is less than two.
func NearestIdxForSpan(n int, l, u float64, v float64) int {
	if n < 2 {
		panic(shortSpan)
	}
	if math.IsNaN(v) {
		return 0
	}

	// Special cases for Inf and NaN.
	switch {
	case math.IsNaN(l) && !math.IsNaN(u):
		return n - 1
	case math.IsNaN(u):
		return 0
	case math.IsInf(l, 0) && math.IsInf(u, 0):
		if l == u {
			return 0
		}
		if n%2 == 1 {
			if !math.IsInf(v, 0) {
				return n / 2
			}
			if math.Copysign(1, v) == math.Copysign(1, l) {
				return 0
			}
			return n/2 + 1
		}
		if math.Copysign(1, v) == math.Copysign(1, l) {
			return 0
		}
		return n / 2
	case math.IsInf(l, 0):
		if v == l {
			return 0
		}
		return n - 1
	case math.IsInf(u, 0):
		if v == u {
			return n - 1
		}
		return 0
	case math.IsInf(v, -1):
		if l <= u {
			return 0
		}
		return n - 1
	case math.IsInf(v, 1):
		if u <= l {
			return 0
		}
		return n - 1
	}

	// Special cases for v outside (l, u) and (u, l).
	switch {
	case l < u:
		if v <= l {
			return 0
		}
		if v >= u {
			return n - 1
		}
	case l > u:
		if v >= l {
			return 0
		}
		if v <= u {
			return n - 1
		}
	default:
		return 0
	}

	// Can't guarantee anything about exactly halfway between
	// because of floating point weirdness.
	return int((float64(n)-1)/(u-l)*(v-l) + 0.5)
}

// Norm returns the L norm of the slice S, defined as
// (sum_{i=1}^N s[i]^L)^{1/L}
// Special cases:
// L = math.Inf(1) gives the maximum absolute value.
// Does not correctly compute the zero norm (use Count).
func Norm(s []float64, L float64) float64 {
	// Should this complain if L is not positive?
	// Should this be done in log space for better numerical stability?
	//	would be more cost
	//	maybe only if L is high?
	if len(s) == 0 {
		return 0
	}
	if L == 2 {
		return f64.L2NormUnitary(s)
	}
	var norm float64
	if L == 1 {
		for _, val := range s {
			norm += math.Abs(val)
		}
		return norm
	}
	if math.IsInf(L, 1) {
		for _, val := range s {
			norm = math.Max(norm, math.Abs(val))
		}
		return norm
	}
	for _, val := range s {
		norm += math.Pow(math.Abs(val), L)
	}
	return math.Pow(norm, 1/L)
}

// Prod returns the product of the elements of the slice.
// Returns 1 if len(s) = 0.
func Prod(s []float64) float64 {
	prod := 1.0
	for _, val := range s {
		prod *= val
	}
	return prod
}

// Reverse reverses the order of elements in the slice.
func Reverse(s []float64) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

// Same returns true when the input slices have the same length and all
// elements have the same value with NaN treated as the same.
func Same(s, t []float64) bool {
	if len(s) != len(t) {
		return false
	}
	for i, v := range s {
		w := t[i]
		if v != w && !(math.IsNaN(v) && math.IsNaN(w)) {
			return false
		}
	}
	return true
}

// Scale multiplies every element in dst by the scalar c.
func Scale(c float64, dst []float64) {
	if len(dst) > 0 {
		f64.ScalUnitary(c, dst)
	}
}

// ScaleTo multiplies the elements in s by c and stores the result in dst.
// It panics if the slice argument lengths do not match.
func ScaleTo(dst []float64, c float64, s []float64) []float64 {
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	if len(dst) > 0 {
		f64.ScalUnitaryTo(dst, c, s)
	}
	return dst
}

// Span returns a set of N equally spaced points between l and u, where N
// is equal to the length of the destination. The first element of the destination
// is l, the final element of the destination is u.
// It panics if the length of dst is less than 2.
//
// Span also returns the mutated slice dst, so that it can be used in range expressions,
// like:
//
//     for i, x := range Span(dst, l, u) { ... }
func Span(dst []float64, l, u float64) []float64 {
	n := len(dst)
	if n < 2 {
		panic(shortSpan)
	}

	// Special cases for Inf and NaN.
	switch {
	case math.IsNaN(l):
		for i := range dst[:len(dst)-1] {
			dst[i] = math.NaN()
		}
		dst[len(dst)-1] = u
		return dst
	case math.IsNaN(u):
		for i := range dst[1:] {
			dst[i+1] = math.NaN()
		}
		dst[0] = l
		return dst
	case math.IsInf(l, 0) && math.IsInf(u, 0):
		for i := range dst[:len(dst)/2] {
			dst[i] = l
			dst[len(dst)-i-1] = u
		}
		if len(dst)%2 == 1 {
			if l != u {
				dst[len(dst)/2] = 0
			} else {
				dst[len(dst)/2] = l
			}
		}
		return dst
	case math.IsInf(l, 0):
		for i := range dst[:len(dst)-1] {
			dst[i] = l
		}
		dst[len(dst)-1] = u
		return dst
	case math.IsInf(u, 0):
		for i := range dst[1:] {
			dst[i+1] = u
		}
		dst[0] = l
		return dst
	}

	step := (u - l) / float64(n-1)
	for i := range dst {
		dst[i] = l + step*float64(i)
	}
	return dst
}

// Sub subtracts, element-wise, the elements of s from dst.
// It panics if the argument lengths do not match.
func Sub(dst, s []float64) {
	if len(dst) != len(s) {
		panic(badLength)
	}
	f64.AxpyUnitaryTo(dst, -1, s, dst)
}

// SubTo subtracts, element-wise, the elements of t from s and
// stores the result in dst.
// It panics if the argument lengths do not match.
func SubTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic(badLength)
	}
	if len(dst) != len(s) {
		panic(badDstLength)
	}
	f64.AxpyUnitaryTo(dst, -1, t, s)
	return dst
}

// Sum returns the sum of the elements of the slice.
func Sum(s []float64) float64 {
	return f64.Sum(s)
}

// Within returns the first index i where s[i] <= v < s[i+1]. Within panics if:
//  - len(s) < 2
//  - s is not sorted
func Within(s []float64, v float64) int {
	if len(s) < 2 {
		panic(shortSpan)
	}
	if !sort.Float64sAreSorted(s) {
		panic("floats: input slice not sorted")
	}
	if v < s[0] || v >= s[len(s)-1] || math.IsNaN(v) {
		return -1
	}
	for i, f := range s[1:] {
		if v < f {
			return i
		}
	}
	return -1
}

// SumCompensated returns the sum of the elements of the slice calculated with greater
// accuracy than Sum at the expense of additional computation.
func SumCompensated(s []float64) float64 {
	// SumCompensated uses an improved version of Kahan's compensated
	// summation algorithm proposed by Neumaier.
	// See https://en.wikipedia.org/wiki/Kahan_summation_algorithm for details.
	var sum, c float64
	for _, x := range s {
		// This type conversion is here to prevent a sufficiently smart compiler
		// from optimising away these operations.
		t := float64(sum + x)
		if math.Abs(sum) >= math.Abs(x) {
			c += (sum - t) + x
		} else {
			c += (x - t) + sum
		}
		sum = t
	}
	return sum + c
}
