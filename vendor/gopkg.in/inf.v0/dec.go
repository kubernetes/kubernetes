// Package inf (type inf.Dec) implements "infinite-precision" decimal
// arithmetic.
// "Infinite precision" describes two characteristics: practically unlimited
// precision for decimal number representation and no support for calculating
// with any specific fixed precision.
// (Although there is no practical limit on precision, inf.Dec can only
// represent finite decimals.)
//
// This package is currently in experimental stage and the API may change.
//
// This package does NOT support:
//  - rounding to specific precisions (as opposed to specific decimal positions)
//  - the notion of context (each rounding must be explicit)
//  - NaN and Inf values, and distinguishing between positive and negative zero
//  - conversions to and from float32/64 types
//
// Features considered for possible addition:
//  + formatting options
//  + Exp method
//  + combined operations such as AddRound/MulAdd etc
//  + exchanging data in decimal32/64/128 formats
//
package inf // import "gopkg.in/inf.v0"

// TODO:
//  - avoid excessive deep copying (quo and rounders)

import (
	"fmt"
	"io"
	"math/big"
	"strings"
)

// A Dec represents a signed arbitrary-precision decimal.
// It is a combination of a sign, an arbitrary-precision integer coefficient
// value, and a signed fixed-precision exponent value.
// The sign and the coefficient value are handled together as a signed value
// and referred to as the unscaled value.
// (Positive and negative zero values are not distinguished.)
// Since the exponent is most commonly non-positive, it is handled in negated
// form and referred to as scale.
//
// The mathematical value of a Dec equals:
//
//  unscaled * 10**(-scale)
//
// Note that different Dec representations may have equal mathematical values.
//
//  unscaled  scale  String()
//  -------------------------
//         0      0    "0"
//         0      2    "0.00"
//         0     -2    "0"
//         1      0    "1"
//       100      2    "1.00"
//        10      0   "10"
//         1     -1   "10"
//
// The zero value for a Dec represents the value 0 with scale 0.
//
// Operations are typically performed through the *Dec type.
// The semantics of the assignment operation "=" for "bare" Dec values is
// undefined and should not be relied on.
//
// Methods are typically of the form:
//
//	func (z *Dec) Op(x, y *Dec) *Dec
//
// and implement operations z = x Op y with the result as receiver; if it
// is one of the operands it may be overwritten (and its memory reused).
// To enable chaining of operations, the result is also returned. Methods
// returning a result other than *Dec take one of the operands as the receiver.
//
// A "bare" Quo method (quotient / division operation) is not provided, as the
// result is not always a finite decimal and thus in general cannot be
// represented as a Dec.
// Instead, in the common case when rounding is (potentially) necessary,
// QuoRound should be used with a Scale and a Rounder.
// QuoExact or QuoRound with RoundExact can be used in the special cases when it
// is known that the result is always a finite decimal.
//
type Dec struct {
	unscaled big.Int
	scale    Scale
}

// Scale represents the type used for the scale of a Dec.
type Scale int32

const scaleSize = 4 // bytes in a Scale value

// Scaler represents a method for obtaining the scale to use for the result of
// an operation on x and y.
type scaler interface {
	Scale(x *Dec, y *Dec) Scale
}

var bigInt = [...]*big.Int{
	big.NewInt(0), big.NewInt(1), big.NewInt(2), big.NewInt(3), big.NewInt(4),
	big.NewInt(5), big.NewInt(6), big.NewInt(7), big.NewInt(8), big.NewInt(9),
	big.NewInt(10),
}

var exp10cache [64]big.Int = func() [64]big.Int {
	e10, e10i := [64]big.Int{}, bigInt[1]
	for i := range e10 {
		e10[i].Set(e10i)
		e10i = new(big.Int).Mul(e10i, bigInt[10])
	}
	return e10
}()

// NewDec allocates and returns a new Dec set to the given int64 unscaled value
// and scale.
func NewDec(unscaled int64, scale Scale) *Dec {
	return new(Dec).SetUnscaled(unscaled).SetScale(scale)
}

// NewDecBig allocates and returns a new Dec set to the given *big.Int unscaled
// value and scale.
func NewDecBig(unscaled *big.Int, scale Scale) *Dec {
	return new(Dec).SetUnscaledBig(unscaled).SetScale(scale)
}

// Scale returns the scale of x.
func (x *Dec) Scale() Scale {
	return x.scale
}

// Unscaled returns the unscaled value of x for u and true for ok when the
// unscaled value can be represented as int64; otherwise it returns an undefined
// int64 value for u and false for ok. Use x.UnscaledBig().Int64() to avoid
// checking the validity of the value when the check is known to be redundant.
func (x *Dec) Unscaled() (u int64, ok bool) {
	u = x.unscaled.Int64()
	var i big.Int
	ok = i.SetInt64(u).Cmp(&x.unscaled) == 0
	return
}

// UnscaledBig returns the unscaled value of x as *big.Int.
func (x *Dec) UnscaledBig() *big.Int {
	return &x.unscaled
}

// SetScale sets the scale of z, with the unscaled value unchanged, and returns
// z.
// The mathematical value of the Dec changes as if it was multiplied by
// 10**(oldscale-scale).
func (z *Dec) SetScale(scale Scale) *Dec {
	z.scale = scale
	return z
}

// SetUnscaled sets the unscaled value of z, with the scale unchanged, and
// returns z.
func (z *Dec) SetUnscaled(unscaled int64) *Dec {
	z.unscaled.SetInt64(unscaled)
	return z
}

// SetUnscaledBig sets the unscaled value of z, with the scale unchanged, and
// returns z.
func (z *Dec) SetUnscaledBig(unscaled *big.Int) *Dec {
	z.unscaled.Set(unscaled)
	return z
}

// Set sets z to the value of x and returns z.
// It does nothing if z == x.
func (z *Dec) Set(x *Dec) *Dec {
	if z != x {
		z.SetUnscaledBig(x.UnscaledBig())
		z.SetScale(x.Scale())
	}
	return z
}

// Sign returns:
//
//	-1 if x <  0
//	 0 if x == 0
//	+1 if x >  0
//
func (x *Dec) Sign() int {
	return x.UnscaledBig().Sign()
}

// Neg sets z to -x and returns z.
func (z *Dec) Neg(x *Dec) *Dec {
	z.SetScale(x.Scale())
	z.UnscaledBig().Neg(x.UnscaledBig())
	return z
}

// Cmp compares x and y and returns:
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func (x *Dec) Cmp(y *Dec) int {
	xx, yy := upscale(x, y)
	return xx.UnscaledBig().Cmp(yy.UnscaledBig())
}

// Abs sets z to |x| (the absolute value of x) and returns z.
func (z *Dec) Abs(x *Dec) *Dec {
	z.SetScale(x.Scale())
	z.UnscaledBig().Abs(x.UnscaledBig())
	return z
}

// Add sets z to the sum x+y and returns z.
// The scale of z is the greater of the scales of x and y.
func (z *Dec) Add(x, y *Dec) *Dec {
	xx, yy := upscale(x, y)
	z.SetScale(xx.Scale())
	z.UnscaledBig().Add(xx.UnscaledBig(), yy.UnscaledBig())
	return z
}

// Sub sets z to the difference x-y and returns z.
// The scale of z is the greater of the scales of x and y.
func (z *Dec) Sub(x, y *Dec) *Dec {
	xx, yy := upscale(x, y)
	z.SetScale(xx.Scale())
	z.UnscaledBig().Sub(xx.UnscaledBig(), yy.UnscaledBig())
	return z
}

// Mul sets z to the product x*y and returns z.
// The scale of z is the sum of the scales of x and y.
func (z *Dec) Mul(x, y *Dec) *Dec {
	z.SetScale(x.Scale() + y.Scale())
	z.UnscaledBig().Mul(x.UnscaledBig(), y.UnscaledBig())
	return z
}

// Round sets z to the value of x rounded to Scale s using Rounder r, and
// returns z.
func (z *Dec) Round(x *Dec, s Scale, r Rounder) *Dec {
	return z.QuoRound(x, NewDec(1, 0), s, r)
}

// QuoRound sets z to the quotient x/y, rounded using the given Rounder to the
// specified scale.
//
// If the rounder is RoundExact but the result can not be expressed exactly at
// the specified scale, QuoRound returns nil, and the value of z is undefined.
//
// There is no corresponding Div method; the equivalent can be achieved through
// the choice of Rounder used.
//
func (z *Dec) QuoRound(x, y *Dec, s Scale, r Rounder) *Dec {
	return z.quo(x, y, sclr{s}, r)
}

func (z *Dec) quo(x, y *Dec, s scaler, r Rounder) *Dec {
	scl := s.Scale(x, y)
	var zzz *Dec
	if r.UseRemainder() {
		zz, rA, rB := new(Dec).quoRem(x, y, scl, true, new(big.Int), new(big.Int))
		zzz = r.Round(new(Dec), zz, rA, rB)
	} else {
		zz, _, _ := new(Dec).quoRem(x, y, scl, false, nil, nil)
		zzz = r.Round(new(Dec), zz, nil, nil)
	}
	if zzz == nil {
		return nil
	}
	return z.Set(zzz)
}

// QuoExact sets z to the quotient x/y and returns z when x/y is a finite
// decimal. Otherwise it returns nil and the value of z is undefined.
//
// The scale of a non-nil result is "x.Scale() - y.Scale()" or greater; it is
// calculated so that the remainder will be zero whenever x/y is a finite
// decimal.
func (z *Dec) QuoExact(x, y *Dec) *Dec {
	return z.quo(x, y, scaleQuoExact{}, RoundExact)
}

// quoRem sets z to the quotient x/y with the scale s, and if useRem is true,
// it sets remNum and remDen to the numerator and denominator of the remainder.
// It returns z, remNum and remDen.
//
// The remainder is normalized to the range -1 < r < 1 to simplify rounding;
// that is, the results satisfy the following equation:
//
//  x / y = z + (remNum/remDen) * 10**(-z.Scale())
//
// See Rounder for more details about rounding.
//
func (z *Dec) quoRem(x, y *Dec, s Scale, useRem bool,
	remNum, remDen *big.Int) (*Dec, *big.Int, *big.Int) {
	// difference (required adjustment) compared to "canonical" result scale
	shift := s - (x.Scale() - y.Scale())
	// pointers to adjusted unscaled dividend and divisor
	var ix, iy *big.Int
	switch {
	case shift > 0:
		// increased scale: decimal-shift dividend left
		ix = new(big.Int).Mul(x.UnscaledBig(), exp10(shift))
		iy = y.UnscaledBig()
	case shift < 0:
		// decreased scale: decimal-shift divisor left
		ix = x.UnscaledBig()
		iy = new(big.Int).Mul(y.UnscaledBig(), exp10(-shift))
	default:
		ix = x.UnscaledBig()
		iy = y.UnscaledBig()
	}
	// save a copy of iy in case it to be overwritten with the result
	iy2 := iy
	if iy == z.UnscaledBig() {
		iy2 = new(big.Int).Set(iy)
	}
	// set scale
	z.SetScale(s)
	// set unscaled
	if useRem {
		// Int division
		_, intr := z.UnscaledBig().QuoRem(ix, iy, new(big.Int))
		// set remainder
		remNum.Set(intr)
		remDen.Set(iy2)
	} else {
		z.UnscaledBig().Quo(ix, iy)
	}
	return z, remNum, remDen
}

type sclr struct{ s Scale }

func (s sclr) Scale(x, y *Dec) Scale {
	return s.s
}

type scaleQuoExact struct{}

func (sqe scaleQuoExact) Scale(x, y *Dec) Scale {
	rem := new(big.Rat).SetFrac(x.UnscaledBig(), y.UnscaledBig())
	f2, f5 := factor2(rem.Denom()), factor(rem.Denom(), bigInt[5])
	var f10 Scale
	if f2 > f5 {
		f10 = Scale(f2)
	} else {
		f10 = Scale(f5)
	}
	return x.Scale() - y.Scale() + f10
}

func factor(n *big.Int, p *big.Int) int {
	// could be improved for large factors
	d, f := n, 0
	for {
		dd, dm := new(big.Int).DivMod(d, p, new(big.Int))
		if dm.Sign() == 0 {
			f++
			d = dd
		} else {
			break
		}
	}
	return f
}

func factor2(n *big.Int) int {
	// could be improved for large factors
	f := 0
	for ; n.Bit(f) == 0; f++ {
	}
	return f
}

func upscale(a, b *Dec) (*Dec, *Dec) {
	if a.Scale() == b.Scale() {
		return a, b
	}
	if a.Scale() > b.Scale() {
		bb := b.rescale(a.Scale())
		return a, bb
	}
	aa := a.rescale(b.Scale())
	return aa, b
}

func exp10(x Scale) *big.Int {
	if int(x) < len(exp10cache) {
		return &exp10cache[int(x)]
	}
	return new(big.Int).Exp(bigInt[10], big.NewInt(int64(x)), nil)
}

func (x *Dec) rescale(newScale Scale) *Dec {
	shift := newScale - x.Scale()
	switch {
	case shift < 0:
		e := exp10(-shift)
		return NewDecBig(new(big.Int).Quo(x.UnscaledBig(), e), newScale)
	case shift > 0:
		e := exp10(shift)
		return NewDecBig(new(big.Int).Mul(x.UnscaledBig(), e), newScale)
	}
	return x
}

var zeros = []byte("00000000000000000000000000000000" +
	"00000000000000000000000000000000")
var lzeros = Scale(len(zeros))

func appendZeros(s []byte, n Scale) []byte {
	for i := Scale(0); i < n; i += lzeros {
		if n > i+lzeros {
			s = append(s, zeros...)
		} else {
			s = append(s, zeros[0:n-i]...)
		}
	}
	return s
}

func (x *Dec) String() string {
	if x == nil {
		return "<nil>"
	}
	scale := x.Scale()
	s := []byte(x.UnscaledBig().String())
	if scale <= 0 {
		if scale != 0 && x.unscaled.Sign() != 0 {
			s = appendZeros(s, -scale)
		}
		return string(s)
	}
	negbit := Scale(-((x.Sign() - 1) / 2))
	// scale > 0
	lens := Scale(len(s))
	if lens-negbit <= scale {
		ss := make([]byte, 0, scale+2)
		if negbit == 1 {
			ss = append(ss, '-')
		}
		ss = append(ss, '0', '.')
		ss = appendZeros(ss, scale-lens+negbit)
		ss = append(ss, s[negbit:]...)
		return string(ss)
	}
	// lens > scale
	ss := make([]byte, 0, lens+1)
	ss = append(ss, s[:lens-scale]...)
	ss = append(ss, '.')
	ss = append(ss, s[lens-scale:]...)
	return string(ss)
}

// Format is a support routine for fmt.Formatter. It accepts the decimal
// formats 'd' and 'f', and handles both equivalently.
// Width, precision, flags and bases 2, 8, 16 are not supported.
func (x *Dec) Format(s fmt.State, ch rune) {
	if ch != 'd' && ch != 'f' && ch != 'v' && ch != 's' {
		fmt.Fprintf(s, "%%!%c(dec.Dec=%s)", ch, x.String())
		return
	}
	fmt.Fprintf(s, x.String())
}

func (z *Dec) scan(r io.RuneScanner) (*Dec, error) {
	unscaled := make([]byte, 0, 256) // collects chars of unscaled as bytes
	dp, dg := -1, -1                 // indexes of decimal point, first digit
loop:
	for {
		ch, _, err := r.ReadRune()
		if err == io.EOF {
			break loop
		}
		if err != nil {
			return nil, err
		}
		switch {
		case ch == '+' || ch == '-':
			if len(unscaled) > 0 || dp >= 0 { // must be first character
				r.UnreadRune()
				break loop
			}
		case ch == '.':
			if dp >= 0 {
				r.UnreadRune()
				break loop
			}
			dp = len(unscaled)
			continue // don't add to unscaled
		case ch >= '0' && ch <= '9':
			if dg == -1 {
				dg = len(unscaled)
			}
		default:
			r.UnreadRune()
			break loop
		}
		unscaled = append(unscaled, byte(ch))
	}
	if dg == -1 {
		return nil, fmt.Errorf("no digits read")
	}
	if dp >= 0 {
		z.SetScale(Scale(len(unscaled) - dp))
	} else {
		z.SetScale(0)
	}
	_, ok := z.UnscaledBig().SetString(string(unscaled), 10)
	if !ok {
		return nil, fmt.Errorf("invalid decimal: %s", string(unscaled))
	}
	return z, nil
}

// SetString sets z to the value of s, interpreted as a decimal (base 10),
// and returns z and a boolean indicating success. The scale of z is the
// number of digits after the decimal point (including any trailing 0s),
// or 0 if there is no decimal point. If SetString fails, the value of z
// is undefined but the returned value is nil.
func (z *Dec) SetString(s string) (*Dec, bool) {
	r := strings.NewReader(s)
	_, err := z.scan(r)
	if err != nil {
		return nil, false
	}
	_, _, err = r.ReadRune()
	if err != io.EOF {
		return nil, false
	}
	// err == io.EOF => scan consumed all of s
	return z, true
}

// Scan is a support routine for fmt.Scanner; it sets z to the value of
// the scanned number. It accepts the decimal formats 'd' and 'f', and
// handles both equivalently. Bases 2, 8, 16 are not supported.
// The scale of z is the number of digits after the decimal point
// (including any trailing 0s), or 0 if there is no decimal point.
func (z *Dec) Scan(s fmt.ScanState, ch rune) error {
	if ch != 'd' && ch != 'f' && ch != 's' && ch != 'v' {
		return fmt.Errorf("Dec.Scan: invalid verb '%c'", ch)
	}
	s.SkipSpace()
	_, err := z.scan(s)
	return err
}

// Gob encoding version
const decGobVersion byte = 1

func scaleBytes(s Scale) []byte {
	buf := make([]byte, scaleSize)
	i := scaleSize
	for j := 0; j < scaleSize; j++ {
		i--
		buf[i] = byte(s)
		s >>= 8
	}
	return buf
}

func scale(b []byte) (s Scale) {
	for j := 0; j < scaleSize; j++ {
		s <<= 8
		s |= Scale(b[j])
	}
	return
}

// GobEncode implements the gob.GobEncoder interface.
func (x *Dec) GobEncode() ([]byte, error) {
	buf, err := x.UnscaledBig().GobEncode()
	if err != nil {
		return nil, err
	}
	buf = append(append(buf, scaleBytes(x.Scale())...), decGobVersion)
	return buf, nil
}

// GobDecode implements the gob.GobDecoder interface.
func (z *Dec) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		return fmt.Errorf("Dec.GobDecode: no data")
	}
	b := buf[len(buf)-1]
	if b != decGobVersion {
		return fmt.Errorf("Dec.GobDecode: encoding version %d not supported", b)
	}
	l := len(buf) - scaleSize - 1
	err := z.UnscaledBig().GobDecode(buf[:l])
	if err != nil {
		return err
	}
	z.SetScale(scale(buf[l : l+scaleSize]))
	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
func (x *Dec) MarshalText() ([]byte, error) {
	return []byte(x.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
func (z *Dec) UnmarshalText(data []byte) error {
	_, ok := z.SetString(string(data))
	if !ok {
		return fmt.Errorf("invalid inf.Dec")
	}
	return nil
}
