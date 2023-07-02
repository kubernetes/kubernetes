// Copyright 2017 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package starlark

import (
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strconv"

	"go.starlark.net/syntax"
)

// Int is the type of a Starlark int.
//
// The zero value is not a legal value; use MakeInt(0).
type Int struct{ impl intImpl }

// --- high-level accessors ---

// MakeInt returns a Starlark int for the specified signed integer.
func MakeInt(x int) Int { return MakeInt64(int64(x)) }

// MakeInt64 returns a Starlark int for the specified int64.
func MakeInt64(x int64) Int {
	if math.MinInt32 <= x && x <= math.MaxInt32 {
		return makeSmallInt(x)
	}
	return makeBigInt(big.NewInt(x))
}

// MakeUint returns a Starlark int for the specified unsigned integer.
func MakeUint(x uint) Int { return MakeUint64(uint64(x)) }

// MakeUint64 returns a Starlark int for the specified uint64.
func MakeUint64(x uint64) Int {
	if x <= math.MaxInt32 {
		return makeSmallInt(int64(x))
	}
	return makeBigInt(new(big.Int).SetUint64(x))
}

// MakeBigInt returns a Starlark int for the specified big.Int.
// The new Int value will contain a copy of x. The caller is safe to modify x.
func MakeBigInt(x *big.Int) Int {
	if isSmall(x) {
		return makeSmallInt(x.Int64())
	}
	z := new(big.Int).Set(x)
	return makeBigInt(z)
}

func isSmall(x *big.Int) bool {
	n := x.BitLen()
	return n < 32 || n == 32 && x.Int64() == math.MinInt32
}

var (
	zero, one = makeSmallInt(0), makeSmallInt(1)
	oneBig    = big.NewInt(1)

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
	iSmall, iBig := i.get()
	if iBig != nil {
		x, acc := bigintToInt64(iBig)
		if acc != big.Exact {
			return // inexact
		}
		return x, true
	}
	return iSmall, true
}

// BigInt returns a new big.Int with the same value as the Int.
func (i Int) BigInt() *big.Int {
	iSmall, iBig := i.get()
	if iBig != nil {
		return new(big.Int).Set(iBig)
	}
	return big.NewInt(iSmall)
}

// bigInt returns the value as a big.Int.
// It differs from BigInt in that this method returns the actual
// reference and any modification will change the state of i.
func (i Int) bigInt() *big.Int {
	iSmall, iBig := i.get()
	if iBig != nil {
		return iBig
	}
	return big.NewInt(iSmall)
}

// Uint64 returns the value as a uint64.
// If it is not exactly representable the result is undefined and ok is false.
func (i Int) Uint64() (_ uint64, ok bool) {
	iSmall, iBig := i.get()
	if iBig != nil {
		x, acc := bigintToUint64(iBig)
		if acc != big.Exact {
			return // inexact
		}
		return x, true
	}
	if iSmall < 0 {
		return // inexact
	}
	return uint64(iSmall), true
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
	iSmall, iBig := i.get()
	if iBig != nil {
		iBig.Format(s, ch)
		return
	}
	big.NewInt(iSmall).Format(s, ch)
}
func (i Int) String() string {
	iSmall, iBig := i.get()
	if iBig != nil {
		return iBig.Text(10)
	}
	return strconv.FormatInt(iSmall, 10)
}
func (i Int) Type() string { return "int" }
func (i Int) Freeze()      {} // immutable
func (i Int) Truth() Bool  { return i.Sign() != 0 }
func (i Int) Hash() (uint32, error) {
	iSmall, iBig := i.get()
	var lo big.Word
	if iBig != nil {
		lo = iBig.Bits()[0]
	} else {
		lo = big.Word(iSmall)
	}
	return 12582917 * uint32(lo+3), nil
}

// Required by the TotallyOrdered interface
func (x Int) Cmp(v Value, depth int) (int, error) {
	y := v.(Int)
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return x.bigInt().Cmp(y.bigInt()), nil
	}
	return signum64(xSmall - ySmall), nil // safe: int32 operands
}

// Float returns the float value nearest i.
func (i Int) Float() Float {
	iSmall, iBig := i.get()
	if iBig != nil {
		// Fast path for hardware int-to-float conversions.
		if iBig.IsUint64() {
			return Float(iBig.Uint64())
		} else if iBig.IsInt64() {
			return Float(iBig.Int64())
		}

		f, _ := new(big.Float).SetInt(iBig).Float64()
		return Float(f)
	}
	return Float(iSmall)
}

// finiteFloat returns the finite float value nearest i,
// or an error if the magnitude is too large.
func (i Int) finiteFloat() (Float, error) {
	f := i.Float()
	if math.IsInf(float64(f), 0) {
		return 0, fmt.Errorf("int too large to convert to float")
	}
	return f, nil
}

func (x Int) Sign() int {
	xSmall, xBig := x.get()
	if xBig != nil {
		return xBig.Sign()
	}
	return signum64(xSmall)
}

func (x Int) Add(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).Add(x.bigInt(), y.bigInt()))
	}
	return MakeInt64(xSmall + ySmall)
}
func (x Int) Sub(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).Sub(x.bigInt(), y.bigInt()))
	}
	return MakeInt64(xSmall - ySmall)
}
func (x Int) Mul(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).Mul(x.bigInt(), y.bigInt()))
	}
	return MakeInt64(xSmall * ySmall)
}
func (x Int) Or(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).Or(x.bigInt(), y.bigInt()))
	}
	return makeSmallInt(xSmall | ySmall)
}
func (x Int) And(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).And(x.bigInt(), y.bigInt()))
	}
	return makeSmallInt(xSmall & ySmall)
}
func (x Int) Xor(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		return MakeBigInt(new(big.Int).Xor(x.bigInt(), y.bigInt()))
	}
	return makeSmallInt(xSmall ^ ySmall)
}
func (x Int) Not() Int {
	xSmall, xBig := x.get()
	if xBig != nil {
		return MakeBigInt(new(big.Int).Not(xBig))
	}
	return makeSmallInt(^xSmall)
}
func (x Int) Lsh(y uint) Int { return MakeBigInt(new(big.Int).Lsh(x.bigInt(), y)) }
func (x Int) Rsh(y uint) Int { return MakeBigInt(new(big.Int).Rsh(x.bigInt(), y)) }

// Precondition: y is nonzero.
func (x Int) Div(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	// http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
	if xBig != nil || yBig != nil {
		xb, yb := x.bigInt(), y.bigInt()

		var quo, rem big.Int
		quo.QuoRem(xb, yb, &rem)
		if (xb.Sign() < 0) != (yb.Sign() < 0) && rem.Sign() != 0 {
			quo.Sub(&quo, oneBig)
		}
		return MakeBigInt(&quo)
	}
	quo := xSmall / ySmall
	rem := xSmall % ySmall
	if (xSmall < 0) != (ySmall < 0) && rem != 0 {
		quo -= 1
	}
	return MakeInt64(quo)
}

// Precondition: y is nonzero.
func (x Int) Mod(y Int) Int {
	xSmall, xBig := x.get()
	ySmall, yBig := y.get()
	if xBig != nil || yBig != nil {
		xb, yb := x.bigInt(), y.bigInt()

		var quo, rem big.Int
		quo.QuoRem(xb, yb, &rem)
		if (xb.Sign() < 0) != (yb.Sign() < 0) && rem.Sign() != 0 {
			rem.Add(&rem, yb)
		}
		return MakeBigInt(&rem)
	}
	rem := xSmall % ySmall
	if (xSmall < 0) != (ySmall < 0) && rem != 0 {
		rem += ySmall
	}
	return makeSmallInt(rem)
}

func (i Int) rational() *big.Rat {
	iSmall, iBig := i.get()
	if iBig != nil {
		return new(big.Rat).SetInt(iBig)
	}
	return new(big.Rat).SetInt64(iSmall)
}

// AsInt32 returns the value of x if is representable as an int32.
func AsInt32(x Value) (int, error) {
	i, ok := x.(Int)
	if !ok {
		return 0, fmt.Errorf("got %s, want int", x.Type())
	}
	iSmall, iBig := i.get()
	if iBig != nil {
		return 0, fmt.Errorf("%s out of range", i)
	}
	return int(iSmall), nil
}

// AsInt sets *ptr to the value of Starlark int x, if it is exactly representable,
// otherwise it returns an error.
// The type of ptr must be one of the pointer types *int, *int8, *int16, *int32, or *int64,
// or one of their unsigned counterparts including *uintptr.
func AsInt(x Value, ptr interface{}) error {
	xint, ok := x.(Int)
	if !ok {
		return fmt.Errorf("got %s, want int", x.Type())
	}

	bits := reflect.TypeOf(ptr).Elem().Size() * 8
	switch ptr.(type) {
	case *int, *int8, *int16, *int32, *int64:
		i, ok := xint.Int64()
		if !ok || bits < 64 && !(-1<<(bits-1) <= i && i < 1<<(bits-1)) {
			return fmt.Errorf("%s out of range (want value in signed %d-bit range)", xint, bits)
		}
		switch ptr := ptr.(type) {
		case *int:
			*ptr = int(i)
		case *int8:
			*ptr = int8(i)
		case *int16:
			*ptr = int16(i)
		case *int32:
			*ptr = int32(i)
		case *int64:
			*ptr = int64(i)
		}

	case *uint, *uint8, *uint16, *uint32, *uint64, *uintptr:
		i, ok := xint.Uint64()
		if !ok || bits < 64 && i >= 1<<bits {
			return fmt.Errorf("%s out of range (want value in unsigned %d-bit range)", xint, bits)
		}
		switch ptr := ptr.(type) {
		case *uint:
			*ptr = uint(i)
		case *uint8:
			*ptr = uint8(i)
		case *uint16:
			*ptr = uint16(i)
		case *uint32:
			*ptr = uint32(i)
		case *uint64:
			*ptr = uint64(i)
		case *uintptr:
			*ptr = uintptr(i)
		}
	default:
		panic(fmt.Sprintf("invalid argument type: %T", ptr))
	}
	return nil
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
	// We avoid '<= MaxInt64' so that both constants are exactly representable as floats.
	// See https://github.com/google/starlark-go/issues/375.
	if math.MinInt64 <= f && f < math.MaxInt64+1 {
		// small values
		return MakeInt64(int64(f))
	}
	rat := f.rational()
	if rat == nil {
		panic(f) // non-finite
	}
	return MakeBigInt(new(big.Int).Div(rat.Num(), rat.Denom()))
}
