// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package floats

import (
	"errors"
	"math"
	"sort"
	"strconv"

	"gonum.org/v1/gonum/internal/asm/f64"
)

// Add adds, element-wise, the elements of s and dst, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Add(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	f64.AxpyUnitaryTo(dst, 1, s, dst)
}

// AddTo adds, element-wise, the elements of s and t and
// stores the result in dst. Panics if the lengths of s, t and dst do not match.
func AddTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic("floats: length of adders do not match")
	}
	if len(dst) != len(s) {
		panic("floats: length of destination does not match length of adder")
	}
	f64.AxpyUnitaryTo(dst, 1, s, t)
	return dst
}

// AddConst adds the scalar c to all of the values in dst.
func AddConst(c float64, dst []float64) {
	f64.AddConst(c, dst)
}

// AddScaled performs dst = dst + alpha * s.
// It panics if the lengths of dst and s are not equal.
func AddScaled(dst []float64, alpha float64, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of destination and source to not match")
	}
	f64.AxpyUnitaryTo(dst, alpha, s, dst)
}

// AddScaledTo performs dst = y + alpha * s, where alpha is a scalar,
// and dst, y and s are all slices.
// It panics if the lengths of dst, y, and s are not equal.
//
// At the return of the function, dst[i] = y[i] + alpha * s[i]
func AddScaledTo(dst, y []float64, alpha float64, s []float64) []float64 {
	if len(dst) != len(s) || len(dst) != len(y) {
		panic("floats: lengths of slices do not match")
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
// It panics if the lengths of dst and inds do not match.
func Argsort(dst []float64, inds []int) {
	if len(dst) != len(inds) {
		panic("floats: length of inds does not match length of slice")
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
// destination dst. A panic will occur if the lengths of arguments
// do not match.
//
// At the return of the function, dst[i] = s[i] * s[i-1] * s[i-2] * ...
func CumProd(dst, s []float64) []float64 {
	if len(dst) != len(s) {
		panic("floats: length of destination does not match length of the source")
	}
	if len(dst) == 0 {
		return dst
	}
	return f64.CumProd(dst, s)
}

// CumSum finds the cumulative sum of the first i elements in
// s and puts them in place into the ith element of the
// destination dst. A panic will occur if the lengths of arguments
// do not match.
//
// At the return of the function, dst[i] = s[i] + s[i-1] + s[i-2] + ...
func CumSum(dst, s []float64) []float64 {
	if len(dst) != len(s) {
		panic("floats: length of destination does not match length of the source")
	}
	if len(dst) == 0 {
		return dst
	}
	return f64.CumSum(dst, s)
}

// Distance computes the L-norm of s - t. See Norm for special cases.
// A panic will occur if the lengths of s and t do not match.
func Distance(s, t []float64, L float64) float64 {
	if len(s) != len(t) {
		panic("floats: slice lengths do not match")
	}
	if len(s) == 0 {
		return 0
	}
	var norm float64
	if L == 2 {
		for i, v := range s {
			diff := t[i] - v
			norm = math.Hypot(norm, diff)
		}
		return norm
	}
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
// and stores the value in dst. It panics if the
// lengths of s and t are not equal.
func Div(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: slice lengths do not match")
	}
	f64.Div(dst, s)
}

// DivTo performs element-wise division s / t
// and stores the value in dst. It panics if the
// lengths of s, t, and dst are not equal.
func DivTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) || len(dst) != len(t) {
		panic("floats: slice lengths do not match")
	}
	return f64.DivTo(dst, s, t)
}

// Dot computes the dot product of s1 and s2, i.e.
// sum_{i = 1}^N s1[i]*s2[i].
// A panic will occur if lengths of arguments do not match.
func Dot(s1, s2 []float64) float64 {
	if len(s1) != len(s2) {
		panic("floats: lengths of the slices do not match")
	}
	return f64.DotUnitary(s1, s2)
}

// Equal returns true if the slices have equal lengths and
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

// EqualApprox returns true if the slices have equal lengths and
// all element pairs have an absolute tolerance less than tol or a
// relative tolerance less than tol.
func EqualApprox(s1, s2 []float64, tol float64) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, a := range s1 {
		if !EqualWithinAbsOrRel(a, s2[i], tol, tol) {
			return false
		}
	}
	return true
}

// EqualFunc returns true if the slices have the same lengths
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

// EqualWithinAbs returns true if a and b have an absolute
// difference of less than tol.
func EqualWithinAbs(a, b, tol float64) bool {
	return a == b || math.Abs(a-b) <= tol
}

const minNormalFloat64 = 2.2250738585072014e-308

// EqualWithinRel returns true if the difference between a and b
// is not greater than tol times the greater value.
func EqualWithinRel(a, b, tol float64) bool {
	if a == b {
		return true
	}
	delta := math.Abs(a - b)
	if delta <= minNormalFloat64 {
		return delta <= tol*minNormalFloat64
	}
	// We depend on the division in this relationship to identify
	// infinities (we rely on the NaN to fail the test) otherwise
	// we compare Infs of the same sign and evaluate Infs as equal
	// independent of sign.
	return delta/math.Max(math.Abs(a), math.Abs(b)) <= tol
}

// EqualWithinAbsOrRel returns true if a and b are equal to within
// the absolute tolerance.
func EqualWithinAbsOrRel(a, b, absTol, relTol float64) bool {
	if EqualWithinAbs(a, b, absTol) {
		return true
	}
	return EqualWithinRel(a, b, relTol)
}

// EqualWithinULP returns true if a and b are equal to within
// the specified number of floating point units in the last place.
func EqualWithinULP(a, b float64, ulp uint) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	if math.Signbit(a) != math.Signbit(b) {
		return math.Float64bits(math.Abs(a))+math.Float64bits(math.Abs(b)) <= uint64(ulp)
	}
	return ulpDiff(math.Float64bits(a), math.Float64bits(b)) <= uint64(ulp)
}

func ulpDiff(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return b - a
}

// EqualLengths returns true if all of the slices have equal length,
// and false otherwise. Returns true if there are no input slices.
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
	// inds is also returned to allow for calling with nil

	// Reslice inds to have zero length
	inds = inds[:0]

	// If zero elements requested, can just return
	if k == 0 {
		return inds, nil
	}

	// If k < 0, return all of the found indices
	if k < 0 {
		for i, val := range s {
			if f(val) {
				inds = append(inds, i)
			}
		}
		return inds, nil
	}

	// Otherwise, find the first k elements
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
	// Finished iterating over the loop, which means k elements were not found
	return inds, errors.New("floats: insufficient elements found")
}

// HasNaN returns true if the slice s has any values that are NaN and false
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
// Panics if len(dst) < 2
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
// entries have the maximum value, the first such index is returned. If the slice
// is empty, MaxIdx will panic.
func MaxIdx(s []float64) int {
	if len(s) == 0 {
		panic("floats: zero slice length")
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

// Min returns the maximum value in the input slice. If the slice is empty, Min will panic.
func Min(s []float64) float64 {
	return s[MinIdx(s)]
}

// MinIdx returns the index of the minimum value in the input slice. If several
// entries have the maximum value, the first such index is returned. If the slice
// is empty, MinIdx will panic.
func MinIdx(s []float64) int {
	if len(s) == 0 {
		panic("floats: zero slice length")
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
// and s and stores the value in dst. Panics if the
// lengths of s and t are not equal.
func Mul(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: slice lengths do not match")
	}
	for i, val := range s {
		dst[i] *= val
	}
}

// MulTo performs element-wise multiplication between s
// and t and stores the value in dst. Panics if the
// lengths of s, t, and dst are not equal.
func MulTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) || len(dst) != len(t) {
		panic("floats: slice lengths do not match")
	}
	for i, val := range t {
		dst[i] = val * s[i]
	}
	return dst
}

const (
	nanBits = 0x7ff8000000000000
	nanMask = 0xfff8000000000000
)

// NaNWith returns an IEEE 754 "quiet not-a-number" value with the
// payload specified in the low 51 bits of payload.
// The NaN returned by math.NaN has a bit pattern equal to NaNWith(1).
func NaNWith(payload uint64) float64 {
	return math.Float64frombits(nanBits | (payload &^ nanMask))
}

// NaNPayload returns the lowest 51 bits payload of an IEEE 754 "quiet
// not-a-number". For values of f other than quiet-NaN, NaNPayload
// returns zero and false.
func NaNPayload(f float64) (payload uint64, ok bool) {
	b := math.Float64bits(f)
	if b&nanBits != nanBits {
		return 0, false
	}
	return b &^ nanMask, true
}

// NearestIdx returns the index of the element in s
// whose value is nearest to v.  If several such
// elements exist, the lowest index is returned.
// NearestIdx panics if len(s) == 0.
func NearestIdx(s []float64, v float64) int {
	if len(s) == 0 {
		panic("floats: zero length slice")
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
// NearestIdxForSpan panics if n is less than two.
func NearestIdxForSpan(n int, l, u float64, v float64) int {
	if n <= 1 {
		panic("floats: span must have length >1")
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
		twoNorm := math.Abs(s[0])
		for i := 1; i < len(s); i++ {
			twoNorm = math.Hypot(twoNorm, s[i])
		}
		return twoNorm
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

// ParseWithNA converts the string s to a float64 in v.
// If s equals missing, w is returned as 0, otherwise 1.
func ParseWithNA(s, missing string) (v, w float64, err error) {
	if s == missing {
		return 0, 0, nil
	}
	v, err = strconv.ParseFloat(s, 64)
	if err == nil {
		w = 1
	}
	return v, w, err
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

// Round returns the half away from zero rounded value of x with prec precision.
//
// Special cases are:
// 	Round(±0) = +0
// 	Round(±Inf) = ±Inf
// 	Round(NaN) = NaN
func Round(x float64, prec int) float64 {
	if x == 0 {
		// Make sure zero is returned
		// without the negative bit set.
		return 0
	}
	// Fast path for positive precision on integers.
	if prec >= 0 && x == math.Trunc(x) {
		return x
	}
	pow := math.Pow10(prec)
	intermed := x * pow
	if math.IsInf(intermed, 0) {
		return x
	}
	if x < 0 {
		x = math.Ceil(intermed - 0.5)
	} else {
		x = math.Floor(intermed + 0.5)
	}

	if x == 0 {
		return 0
	}

	return x / pow
}

// RoundEven returns the half even rounded value of x with prec precision.
//
// Special cases are:
// 	RoundEven(±0) = +0
// 	RoundEven(±Inf) = ±Inf
// 	RoundEven(NaN) = NaN
func RoundEven(x float64, prec int) float64 {
	if x == 0 {
		// Make sure zero is returned
		// without the negative bit set.
		return 0
	}
	// Fast path for positive precision on integers.
	if prec >= 0 && x == math.Trunc(x) {
		return x
	}
	pow := math.Pow10(prec)
	intermed := x * pow
	if math.IsInf(intermed, 0) {
		return x
	}
	if isHalfway(intermed) {
		correction, _ := math.Modf(math.Mod(intermed, 2))
		intermed += correction
		if intermed > 0 {
			x = math.Floor(intermed)
		} else {
			x = math.Ceil(intermed)
		}
	} else {
		if x < 0 {
			x = math.Ceil(intermed - 0.5)
		} else {
			x = math.Floor(intermed + 0.5)
		}
	}

	if x == 0 {
		return 0
	}

	return x / pow
}

func isHalfway(x float64) bool {
	_, frac := math.Modf(x)
	frac = math.Abs(frac)
	return frac == 0.5 || (math.Nextafter(frac, math.Inf(-1)) < 0.5 && math.Nextafter(frac, math.Inf(1)) > 0.5)
}

// Same returns true if the input slices have the same length and the all elements
// have the same value with NaN treated as the same.
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
func ScaleTo(dst []float64, c float64, s []float64) []float64 {
	if len(dst) != len(s) {
		panic("floats: lengths of slices do not match")
	}
	if len(dst) > 0 {
		f64.ScalUnitaryTo(dst, c, s)
	}
	return dst
}

// Span returns a set of N equally spaced points between l and u, where N
// is equal to the length of the destination. The first element of the destination
// is l, the final element of the destination is u.
//
// Panics if len(dst) < 2.
//
// Span also returns the mutated slice dst, so that it can be used in range expressions,
// like:
//
//     for i, x := range Span(dst, l, u) { ... }
func Span(dst []float64, l, u float64) []float64 {
	n := len(dst)
	if n < 2 {
		panic("floats: destination must have length >1")
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

// Sub subtracts, element-wise, the elements of s from dst. Panics if
// the lengths of dst and s do not match.
func Sub(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	f64.AxpyUnitaryTo(dst, -1, s, dst)
}

// SubTo subtracts, element-wise, the elements of t from s and
// stores the result in dst. Panics if the lengths of s, t and dst do not match.
func SubTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic("floats: length of subtractor and subtractee do not match")
	}
	if len(dst) != len(s) {
		panic("floats: length of destination does not match length of subtractor")
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
		panic("floats: slice length less than 2")
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
