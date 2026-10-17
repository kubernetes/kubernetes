/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package exact is an arbitrary-precision model of resource.Quantity: the
// value a Quantity denotes, the documented grammar it is parsed from, and the
// canonical form it is serialized to.
//
// It exists to check resource.Quantity against, so it favours being obviously
// correct over being fast, and its method set mirrors resource.Quantity's so
// that in principle it could stand in for it. It must not import
// k8s.io/apimachinery/pkg/api/resource.
package exact

import (
	"fmt"
	"math"
	"math/big"
	"strings"
)

// The formats a Quantity is serialized in; the values equal resource.Format's.
const (
	DecimalExponent = "DecimalExponent"
	BinarySI        = "BinarySI"
	DecimalSI       = "DecimalSI"
)

// RatLimit bounds the decimal exponent of the values New and Parse build as a
// big.Rat. big.Rat.SetString puts the same bound on the exponent it reads,
// after taking off the fraction digits, and 10^RatLimit takes about 10 ms to
// build.
const RatLimit = 1_000_000

// MaxExponent bounds the decimal exponent of every value: Parse returns
// ErrOutOfDomain for a non-zero value above it and rounds one below it to 1n,
// and New panics with Unjudgeable outside it. The int32 edges of
// resource.Quantity's scale are far inside.
const MaxExponent = int64(1) << 40

// Unjudgeable is the panic value of an operation that package exact cannot
// compute, so that a caller checking a library against it can tell "the
// reference cannot judge this" from a panic in the code under test.
type Unjudgeable string

func (u Unjudgeable) Error() string { return "exact: cannot judge: " + string(u) }

var bigTen = big.NewInt(10)

// Quantity is an exact rational value, together with the format a parsed
// quantity keeps for serialization.
//
// A value whose decimal exponent is within ±RatLimit is a big.Rat, and all
// arithmetic on it is big.Rat arithmetic. Beyond that the value is kept as a
// mantissa and a decimal exponent, since the cost of materializing it grows
// without bound: 10^(4e8) takes minutes and a gigabyte, and
// resource.Quantity's int32 scale reaches ±2.1e9. Which of the two holds a
// value depends only on the value. Only what those values need is supported:
// Sign, Cmp, the canonical string, and rounding and the accessors, which work
// on the mantissa once a comparison with a bound has settled the rest. Add and
// Sub report that they cannot take such an operand.
//
// The zero value is 0 with no format, which serializes like
// resource.Quantity's zero value.
type Quantity struct {
	// rat is the value, unless edge is set.
	rat *big.Rat
	// edge is the value when its exponent is beyond ±RatLimit.
	edge   *decimal
	format string
}

// decimal is mantissa * 10^exponent with a non-zero mantissa that has no
// trailing decimal zeros.
type decimal struct {
	mantissa *big.Int
	exponent int64
}

// New returns mantissa * 10^exponent in the given format; a nil mantissa is
// zero. mantissa is copied.
// It panics with Unjudgeable when the value's exponent is outside
// ±MaxExponent.
func New(mantissa *big.Int, exponent int64, format string) Quantity {
	if mantissa == nil || mantissa.Sign() == 0 {
		return Quantity{format: format}
	}
	m, z := stripZeros(mantissa)
	if exponent > MaxExponent-z || exponent < -MaxExponent-z {
		panic(Unjudgeable(fmt.Sprintf("%se%d is outside the exponent domain ±2^40", mantissa, exponent)))
	}
	return fromDecimal(decimal{mantissa: m, exponent: exponent + z}, format)
}

// fromDecimal holds d as a big.Rat or as a decimal, by its exponent.
func fromDecimal(d decimal, format string) Quantity {
	if d.exponent > RatLimit || d.exponent < -RatLimit {
		return Quantity{edge: &d, format: format}
	}
	return Quantity{rat: decimalRat(d.mantissa, d.exponent), format: format}
}

// NewInt64 returns v * 10^exponent in DecimalSI format.
func NewInt64(v int64, exponent int64) Quantity {
	return New(big.NewInt(v), exponent, DecimalSI)
}

// newRat returns r in the representation its value calls for.
func newRat(r *big.Rat, format string) Quantity {
	if r.Sign() == 0 {
		return Quantity{format: format}
	}
	if d := ratDecimal(r); d.exponent > RatLimit || d.exponent < -RatLimit {
		return Quantity{edge: &d, format: format}
	}
	return Quantity{rat: r, format: format}
}

// stripZeros returns a copy of the non-zero n without its trailing decimal
// zeros, and how many there were. It goes through the decimal string, which
// math/big converts in subquadratic time, rather than dividing by ten once per
// zero.
func stripZeros(n *big.Int) (*big.Int, int64) {
	s := n.String()
	t := strings.TrimRight(s, "0")
	m, _ := new(big.Int).SetString(t, 10)
	return m, int64(len(s) - len(t))
}

func pow10(n int64) *big.Int {
	return new(big.Int).Exp(bigTen, big.NewInt(n), nil)
}

// decimalRat returns m * 10^e as a big.Rat.
func decimalRat(m *big.Int, e int64) *big.Rat {
	if e >= 0 {
		return new(big.Rat).SetInt(new(big.Int).Mul(m, pow10(e)))
	}
	return new(big.Rat).SetFrac(m, pow10(-e))
}

func (q Quantity) r() *big.Rat {
	if q.rat == nil {
		return new(big.Rat)
	}
	return q.rat
}

// decimal returns q as mantissa * 10^exponent, the form both the edges and
// the canonical string work in. For a big.Rat the denominator divides a power
// of ten, since every value is built from decimal digits and powers of two.
// It must not be called on zero.
func (q Quantity) decimal() decimal {
	if q.edge != nil {
		return *q.edge
	}
	return ratDecimal(q.r())
}

// ratDecimal returns the non-zero r as mantissa * 10^exponent.
func ratDecimal(r *big.Rat) decimal {
	den := r.Denom()
	twos := int64(den.TrailingZeroBits())
	fives := log5(new(big.Int).Rsh(den, uint(twos)))
	k := max(twos, fives)
	m := new(big.Int).Mul(r.Num(), pow10(k))
	m.Quo(m, den)
	if k == 0 {
		// An integer may end in zeros; a proper fraction in lowest terms
		// scaled by its smallest power of ten cannot.
		var z int64
		m, z = stripZeros(m)
		return decimal{mantissa: m, exponent: z}
	}
	return decimal{mantissa: m, exponent: -k}
}

// log5 returns k such that n == 5^k, and panics if n is not a power of five.
func log5(n *big.Int) int64 {
	if n.Cmp(big.NewInt(1)) == 0 {
		return 0
	}
	// 5^k has floor(k*log2(5))+1 bits.
	guess := int64(float64(n.BitLen()-1) / math.Log2(5))
	for k := max(guess-1, 1); k <= guess+1; k++ {
		if new(big.Int).Exp(big.NewInt(5), big.NewInt(k), nil).Cmp(n) == 0 {
			return k
		}
	}
	panic(Unjudgeable(fmt.Sprintf("%s does not divide a power of ten", n)))
}

// Mantissa returns a copy of the mantissa: the value's digits without
// trailing zeros, or 0.
func (q Quantity) Mantissa() *big.Int {
	if q.IsZero() {
		return new(big.Int)
	}
	return new(big.Int).Set(q.decimal().mantissa)
}

// Exponent returns the decimal exponent of the value, or 0 for zero.
func (q Quantity) Exponent() int64 {
	if q.IsZero() {
		return 0
	}
	return q.decimal().exponent
}

// Format returns the format the quantity is serialized in; like
// resource.Quantity's, it is empty for the zero value.
func (q Quantity) Format() string { return q.format }

// Sign returns -1, 0 or 1.
func (q Quantity) Sign() int {
	if q.edge != nil {
		return q.edge.mantissa.Sign()
	}
	return q.r().Sign()
}

// IsZero reports whether the value is zero.
func (q Quantity) IsZero() bool { return q.Sign() == 0 }

// Neg returns -q.
func (q Quantity) Neg() Quantity {
	if q.edge != nil {
		return Quantity{edge: &decimal{mantissa: new(big.Int).Neg(q.edge.mantissa), exponent: q.edge.exponent}, format: q.format}
	}
	return newRat(new(big.Rat).Neg(q.r()), q.format)
}

// ExactString returns the value as <mantissa>e<exponent>, or "0", whatever
// the format. It is the form used to report values, not a serialization.
func (q Quantity) ExactString() string {
	if q.IsZero() {
		return "0"
	}
	d := q.decimal()
	return fmt.Sprintf("%se%d", d.mantissa, d.exponent)
}

func decimalDigits(n *big.Int) int64 {
	l := int64(len(n.String()))
	if n.Sign() < 0 {
		l--
	}
	return l
}

// cmpMag compares |a| and |b| from their decimal forms, without
// materializing either value.
func cmpMag(a, b Quantity) int {
	as, bs := a.Sign(), b.Sign()
	if as == 0 || bs == 0 {
		switch {
		case as == 0 && bs == 0:
			return 0
		case as == 0:
			return -1
		default:
			return 1
		}
	}
	da, db := a.decimal(), b.decimal()
	ma, mb := new(big.Int).Abs(da.mantissa), new(big.Int).Abs(db.mantissa)
	la, lb := decimalDigits(ma), decimalDigits(mb)
	// 10^(exponent+digits-1) <= |x| < 10^(exponent+digits).
	if da.exponent+la-1 >= db.exponent+lb {
		return 1
	}
	if db.exponent+lb-1 >= da.exponent+la {
		return -1
	}
	// The magnitudes overlap, so the exponents differ by at most the longer
	// mantissa's digit count and the alignment is cheap.
	base := min(da.exponent, db.exponent)
	ma.Mul(ma, pow10(da.exponent-base))
	mb.Mul(mb, pow10(db.exponent-base))
	return ma.Cmp(mb)
}

// Cmp returns -1, 0 or 1 as q is less than, equal to or greater than y.
func (q Quantity) Cmp(y Quantity) int {
	if q.edge == nil && y.edge == nil {
		return q.r().Cmp(y.r())
	}
	qs, ys := q.Sign(), y.Sign()
	switch {
	case qs < ys:
		return -1
	case qs > ys:
		return 1
	case qs == 0:
		return 0
	}
	c := cmpMag(q, y)
	if qs < 0 {
		return -c
	}
	return c
}

// CmpInt64 compares q with y.
func (q Quantity) CmpInt64(y int64) int { return q.Cmp(NewInt64(y, 0)) }

// Equal reports whether q and y denote the same value, whatever their formats.
func (q Quantity) Equal(y Quantity) bool { return q.Cmp(y) == 0 }

// Add returns q + y in q's format, or in y's when q is zero, as
// resource.Quantity.Add does. ok is false, and the result meaningless, when
// both operands are non-zero and one is kept beyond ±RatLimit.
func (q Quantity) Add(y Quantity) (sum Quantity, ok bool) {
	switch {
	case q.IsZero():
		return y, true
	case y.IsZero():
		return q, true
	case q.edge != nil || y.edge != nil:
		return Quantity{}, false
	}
	return newRat(new(big.Rat).Add(q.r(), y.r()), q.format), true
}

// Sub returns q - y; see Add.
func (q Quantity) Sub(y Quantity) (Quantity, bool) { return q.Add(y.Neg()) }

// Mul returns q * y in q's format. Unlike Add it needs no alignment, so it
// works on every value.
func (q Quantity) Mul(y int64) Quantity {
	if q.edge != nil && y != 0 {
		m := new(big.Int).Mul(q.edge.mantissa, big.NewInt(y))
		return New(m, q.edge.exponent, q.format)
	}
	return newRat(new(big.Rat).Mul(q.r(), new(big.Rat).SetInt64(y)), q.format)
}

func rail(sign int) int64 {
	if sign < 0 {
		return math.MinInt64
	}
	return math.MaxInt64
}

// quoAway returns q / 10^scale rounded away from zero, and whether that lost
// nothing. q must be a big.Rat and scale within ±RatLimit.
func (q Quantity) quoAway(scale int32) (*big.Int, bool) {
	x := new(big.Rat).Quo(q.r(), decimalRat(big.NewInt(1), int64(scale)))
	n, rem := new(big.Int).QuoRem(x.Num(), x.Denom(), new(big.Int))
	if rem.Sign() == 0 {
		return n, true
	}
	return n.Add(n, big.NewInt(int64(q.Sign()))), false
}

func ratScale(scale int32) bool { return scale >= -RatLimit && scale <= RatLimit }

// int64Units is the most units of 10^scale an int64 holds with q's sign:
// 2^63 for a negative q, 2^63-1 otherwise.
func int64Units(sign int) *big.Int {
	if sign < 0 {
		return new(big.Int).Lsh(big.NewInt(1), 63)
	}
	return big.NewInt(math.MaxInt64)
}

// edgeQuoAway is quoAway for a value kept as a decimal, or a scale beyond
// ±RatLimit. The caller has ruled out a quotient beyond what fits the
// mantissa: either q.exponent >= scale and |q| is at most 2^63 units, or
// q.exponent < scale, where a quotient below one unit rounds to one.
func (q Quantity) edgeQuoAway(scale int32) (*big.Int, bool) {
	d := q.decimal()
	shift := d.exponent - int64(scale)
	if shift >= 0 {
		return new(big.Int).Mul(d.mantissa, pow10(shift)), true
	}
	if -shift > decimalDigits(d.mantissa) {
		return big.NewInt(int64(q.Sign())), false
	}
	n, rem := new(big.Int).QuoRem(d.mantissa, pow10(-shift), new(big.Int))
	if rem.Sign() == 0 {
		return n, true
	}
	return n.Add(n, big.NewInt(int64(q.Sign()))), false
}

// AsScaledInt64 returns q / 10^scale rounded away from zero, and whether that
// fits in int64; when it does not, the result is the int64 limit of q's sign.
func (q Quantity) AsScaledInt64(scale int32) (int64, bool) {
	if q.IsZero() {
		return 0, true
	}
	var n *big.Int
	switch {
	case q.edge == nil && ratScale(scale):
		n, _ = q.quoAway(scale)
	case cmpMag(q, New(int64Units(q.Sign()), int64(scale), DecimalSI)) > 0:
		return rail(q.Sign()), false
	default:
		n, _ = q.edgeQuoAway(scale)
	}
	if n.IsInt64() {
		return n.Int64(), true
	}
	return rail(q.Sign()), false
}

// AsInt64 returns q as an int64 when q is an integer that fits.
func (q Quantity) AsInt64() (int64, bool) {
	if q.IsZero() {
		return 0, true
	}
	if q.Exponent() < 0 {
		return 0, false
	}
	return q.AsScaledInt64(0)
}

// RoundToScale rounds q away from zero to a multiple of 10^scale and reports
// whether that lost nothing.
func (q Quantity) RoundToScale(scale int32) (Quantity, bool) {
	switch {
	case q.IsZero():
		return q, true
	case q.edge == nil && ratScale(scale):
		n, exact := q.quoAway(scale)
		return newRat(decimalRat(n, int64(scale)), q.format), exact
	case q.Exponent() >= int64(scale):
		return q, true
	}
	n, exact := q.edgeQuoAway(scale)
	return New(n, int64(scale), q.format), exact
}

// AsApproximateFloat64 returns the float64 nearest to q, or the infinity or
// zero of q's sign outside the float64 range.
func (q Quantity) AsApproximateFloat64() float64 {
	switch {
	case q.edge == nil:
	case cmpMag(q, NewInt64(1, 309)) >= 0:
		return math.Inf(q.Sign())
	case cmpMag(q, NewInt64(1, -400)) < 0:
		return math.Copysign(0, float64(q.Sign()))
	default:
		// Within the float64 range an exponent beyond ±RatLimit takes about
		// as many mantissa digits, so the big.Rat costs what the mantissa does.
		d := q.decimal()
		f, _ := decimalRat(d.mantissa, d.exponent).Float64()
		return f
	}
	f, _ := q.r().Float64()
	return f
}

// Rat returns a copy of q as a big.Rat, unless q is kept beyond ±RatLimit.
func (q Quantity) Rat() (*big.Rat, bool) {
	if q.edge != nil {
		return nil, false
	}
	return new(big.Rat).Set(q.r()), true
}
