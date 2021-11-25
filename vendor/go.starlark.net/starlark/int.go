// Copyright 2017 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package starlark

import (
	"fmt"
	"math"
	"math/big"
	"strconv"

	"go.starlark.net/syntax"
)

// Int is the type of a Starlark int.
type Int struct {
	// We use only the signed 32 bit range of small to ensure
	// that small+small and small*small do not overflow.

	small int64    // minint32 <= small <= maxint32
	big   *big.Int // big != nil <=> value is not representable as int32
}

// newBig allocates a new big.Int.
func newBig(x int64) *big.Int {
	if 0 <= x && int64(big.Word(x)) == x {
		// x is guaranteed to fit into a single big.Word.
		// Most starlark ints are small,
		// but math/big assumes that since you've chosen to use math/big,
		// your big.Ints will probably grow, so it over-allocates.
		// Avoid that over-allocation by manually constructing a single-word slice.
		// See https://golang.org/cl/150999, which will hopefully land in Go 1.13.
		return new(big.Int).SetBits([]big.Word{big.Word(x)})
	}
	return big.NewInt(x)
}

// MakeInt returns a Starlark int for the specified signed integer.
func MakeInt(x int) Int { return MakeInt64(int64(x)) }

// MakeInt64 returns a Starlark int for the specified int64.
func MakeInt64(x int64) Int {
	if math.MinInt32 <= x && x <= math.MaxInt32 {
		return Int{small: x}
	}
	return Int{big: newBig(x)}
}

// MakeUint returns a Starlark int for the specified unsigned integer.
func MakeUint(x uint) Int { return MakeUint64(uint64(x)) }

// MakeUint64 returns a Starlark int for the specified uint64.
func MakeUint64(x uint64) Int {
	if x <= math.MaxInt32 {
		return Int{small: int64(x)}
	}
	if uint64(big.Word(x)) == x {
		// See comment in newBig for an explanation of this optimization.
		return Int{big: new(big.Int).SetBits([]big.Word{big.Word(x)})}
	}
	return Int{big: new(big.Int).SetUint64(x)}
}

// MakeBigInt returns a Starlark int for the specified big.Int.
// The caller must not subsequently modify x.
func MakeBigInt(x *big.Int) Int {
	if n := x.BitLen(); n < 32 || n == 32 && x.Int64() == math.MinInt32 {
		return Int{small: x.Int64()}
	}
	return Int{big: x}
}

var (
	zero, one = Int{small: 0}, Int{small: 1}
	oneBig    = newBig(1)

	_ HasUnary = Int{}
)

// Unary implements the operations +int, -int, and ~int.
func (i Int) Unary(op syntax.Token) (Value, error) {
	switch op {
	case syntax.MINUS:
		return zero.Sub(i), nil
	case syntax.PLUS:
		return i, nil
	case syntax.TILDE:
		return i.Not(), nil
	}
	return nil, nil
}

// Int64 returns the value as an int64.
// If it is not exactly representable the result is undefined and ok is false.
func (i Int) Int64() (_ int64, ok bool) {
	if i.big != nil {
		x, acc := bigintToInt64(i.big)
		if acc != big.Exact {
			return // inexact
		}
		return x, true
	}
	return i.small, true
}

// BigInt returns the value as a big.Int.
// The returned variable must not be modified by the client.
func (i Int) BigInt() *big.Int {
	if i.big != nil {
		return i.big
	}
	return newBig(i.small)
}

// Uint64 returns the value as a uint64.
// If it is not exactly representable the result is undefined and ok is false.
func (i Int) Uint64() (_ uint64, ok bool) {
	if i.big != nil {
		x, acc := bigintToUint64(i.big)
		if acc != big.Exact {
			return // inexact
		}
		return x, true
	}
	if i.small < 0 {
		return // inexact
	}
	return uint64(i.small), true
}

// The math/big API should provide this function.
func bigintToInt64(i *big.Int) (int64, big.Accuracy) {
	sign := i.Sign()
	if sign > 0 {
		if i.Cmp(maxint64) > 0 {
			return math.MaxInt64, big.Below
		}
	} else if sign < 0 {
		if i.Cmp(minint64) < 0 {
			return math.MinInt64, big.Above
		}
	}
	return i.Int64(), big.Exact
}

// The math/big API should provide this function.
func bigintToUint64(i *big.Int) (uint64, big.Accuracy) {
	sign := i.Sign()
	if sign > 0 {
		if i.BitLen() > 64 {
			return math.MaxUint64, big.Below
		}
	} else if sign < 0 {
		return 0, big.Above
	}
	return i.Uint64(), big.Exact
}

var (
	minint64 = new(big.Int).SetInt64(math.MinInt64)
	maxint64 = new(big.Int).SetInt64(math.MaxInt64)
)

func (i Int) Format(s fmt.State, ch rune) {
	if i.big != nil {
		i.big.Format(s, ch)
		return
	}
	newBig(i.small).Format(s, ch)
}
func (i Int) String() string {
	if i.big != nil {
		return i.big.Text(10)
	}
	return strconv.FormatInt(i.small, 10)
}
func (i Int) Type() string { return "int" }
func (i Int) Freeze()      {} // immutable
func (i Int) Truth() Bool  { return i.Sign() != 0 }
func (i Int) Hash() (uint32, error) {
	var lo big.Word
	if i.big != nil {
		lo = i.big.Bits()[0]
	} else {
		lo = big.Word(i.small)
	}
	return 12582917 * uint32(lo+3), nil
}
func (x Int) CompareSameType(op syntax.Token, v Value, depth int) (bool, error) {
	y := v.(Int)
	if x.big != nil || y.big != nil {
		return threeway(op, x.BigInt().Cmp(y.BigInt())), nil
	}
	return threeway(op, signum64(x.small-y.small)), nil
}

// Float returns the float value nearest i.
func (i Int) Float() Float {
	if i.big != nil {
		f, _ := new(big.Float).SetInt(i.big).Float64()
		return Float(f)
	}
	return Float(i.small)
}

func (x Int) Sign() int {
	if x.big != nil {
		return x.big.Sign()
	}
	return signum64(x.small)
}

func (x Int) Add(y Int) Int {
	if x.big != nil || y.big != nil {
		return MakeBigInt(new(big.Int).Add(x.BigInt(), y.BigInt()))
	}
	return MakeInt64(x.small + y.small)
}
func (x Int) Sub(y Int) Int {
	if x.big != nil || y.big != nil {
		return MakeBigInt(new(big.Int).Sub(x.BigInt(), y.BigInt()))
	}
	return MakeInt64(x.small - y.small)
}
func (x Int) Mul(y Int) Int {
	if x.big != nil || y.big != nil {
		return MakeBigInt(new(big.Int).Mul(x.BigInt(), y.BigInt()))
	}
	return MakeInt64(x.small * y.small)
}
func (x Int) Or(y Int) Int {
	if x.big != nil || y.big != nil {
		return Int{big: new(big.Int).Or(x.BigInt(), y.BigInt())}
	}
	return Int{small: x.small | y.small}
}
func (x Int) And(y Int) Int {
	if x.big != nil || y.big != nil {
		return MakeBigInt(new(big.Int).And(x.BigInt(), y.BigInt()))
	}
	return Int{small: x.small & y.small}
}
func (x Int) Xor(y Int) Int {
	if x.big != nil || y.big != nil {
		return MakeBigInt(new(big.Int).Xor(x.BigInt(), y.BigInt()))
	}
	return Int{small: x.small ^ y.small}
}
func (x Int) Not() Int {
	if x.big != nil {
		return MakeBigInt(new(big.Int).Not(x.big))
	}
	return Int{small: ^x.small}
}
func (x Int) Lsh(y uint) Int { return MakeBigInt(new(big.Int).Lsh(x.BigInt(), y)) }
func (x Int) Rsh(y uint) Int { return MakeBigInt(new(big.Int).Rsh(x.BigInt(), y)) }

// Precondition: y is nonzero.
func (x Int) Div(y Int) Int {
	// http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
	if x.big != nil || y.big != nil {
		xb, yb := x.BigInt(), y.BigInt()

		var quo, rem big.Int
		quo.QuoRem(xb, yb, &rem)
		if (xb.Sign() < 0) != (yb.Sign() < 0) && rem.Sign() != 0 {
			quo.Sub(&quo, oneBig)
		}
		return MakeBigInt(&quo)
	}
	quo := x.small / y.small
	rem := x.small % y.small
	if (x.small < 0) != (y.small < 0) && rem != 0 {
		quo -= 1
	}
	return MakeInt64(quo)
}

// Precondition: y is nonzero.
func (x Int) Mod(y Int) Int {
	if x.big != nil || y.big != nil {
		xb, yb := x.BigInt(), y.BigInt()

		var quo, rem big.Int
		quo.QuoRem(xb, yb, &rem)
		if (xb.Sign() < 0) != (yb.Sign() < 0) && rem.Sign() != 0 {
			rem.Add(&rem, yb)
		}
		return MakeBigInt(&rem)
	}
	rem := x.small % y.small
	if (x.small < 0) != (y.small < 0) && rem != 0 {
		rem += y.small
	}
	return Int{small: rem}
}

func (i Int) rational() *big.Rat {
	if i.big != nil {
		return new(big.Rat).SetInt(i.big)
	}
	return new(big.Rat).SetInt64(i.small)
}

// AsInt32 returns the value of x if is representable as an int32.
func AsInt32(x Value) (int, error) {
	i, ok := x.(Int)
	if !ok {
		return 0, fmt.Errorf("got %s, want int", x.Type())
	}
	if i.big != nil {
		return 0, fmt.Errorf("%s out of range", i)
	}
	return int(i.small), nil
}

// NumberToInt converts a number x to an integer value.
// An int is returned unchanged, a float is truncated towards zero.
// NumberToInt reports an error for all other values.
func NumberToInt(x Value) (Int, error) {
	switch x := x.(type) {
	case Int:
		return x, nil
	case Float:
		f := float64(x)
		if math.IsInf(f, 0) {
			return zero, fmt.Errorf("cannot convert float infinity to integer")
		} else if math.IsNaN(f) {
			return zero, fmt.Errorf("cannot convert float NaN to integer")
		}
		return finiteFloatToInt(x), nil

	}
	return zero, fmt.Errorf("cannot convert %s to int", x.Type())
}

// finiteFloatToInt converts f to an Int, truncating towards zero.
// f must be finite.
func finiteFloatToInt(f Float) Int {
	if math.MinInt64 <= f && f <= math.MaxInt64 {
		// small values
		return MakeInt64(int64(f))
	}
	rat := f.rational()
	if rat == nil {
		panic(f) // non-finite
	}
	return MakeBigInt(new(big.Int).Div(rat.Num(), rat.Denom()))
}
