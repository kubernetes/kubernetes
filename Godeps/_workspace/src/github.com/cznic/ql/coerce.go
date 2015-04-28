// Copyright 2013 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CAUTION: This file was generated automatically by
//
//	$ go run helper -o coerce.go
//
// DO NOT EDIT!

package ql

import (
	"math"
	"math/big"
	"reflect"
	"time"
)

func coerce(a, b interface{}) (x, y interface{}) {
	if reflect.TypeOf(a) == reflect.TypeOf(b) {
		return a, b
	}

	switch a.(type) {
	case idealComplex, idealFloat, idealInt, idealRune, idealUint:
		switch b.(type) {
		case idealComplex, idealFloat, idealInt, idealRune, idealUint:
			x, y = coerce1(a, b), b
			if reflect.TypeOf(x) == reflect.TypeOf(y) {
				return
			}

			return a, coerce1(b, a)
		default:
			return coerce1(a, b), b
		}
	default:
		switch b.(type) {
		case idealComplex, idealFloat, idealInt, idealRune, idealUint:
			return a, coerce1(b, a)
		default:
			return a, b
		}
	}
}

func coerce1(inVal, otherVal interface{}) (coercedInVal interface{}) {
	coercedInVal = inVal
	if otherVal == nil {
		return
	}

	switch x := inVal.(type) {
	case nil:
		return
	case idealComplex:
		switch otherVal.(type) {
		//case idealComplex:
		//case idealFloat:
		//case idealInt:
		//case idealRune:
		//case idealUint:
		//case bool:
		case complex64:
			return complex64(x)
		case complex128:
			return complex128(x)
			//case float32:
			//case float64:
			//case int8:
			//case int16:
			//case int32:
			//case int64:
			//case string:
			//case uint8:
			//case uint16:
			//case uint32:
			//case uint64:
			//case *big.Int:
			//case *big.Rat:
			//case time.Time:
			//case time.Duration:
		}
	case idealFloat:
		switch otherVal.(type) {
		case idealComplex:
			return idealComplex(complex(float64(x), 0))
		case idealFloat:
			return idealFloat(float64(x))
		//case idealInt:
		//case idealRune:
		//case idealUint:
		//case bool:
		case complex64:
			return complex64(complex(float32(x), 0))
		case complex128:
			return complex128(complex(float64(x), 0))
		case float32:
			return float32(float64(x))
		case float64:
			return float64(float64(x))
		//case int8:
		//case int16:
		//case int32:
		//case int64:
		//case string:
		//case uint8:
		//case uint16:
		//case uint32:
		//case uint64:
		//case *big.Int:
		case *big.Rat:
			return big.NewRat(1, 1).SetFloat64(float64(x))
			//case time.Time:
			//case time.Duration:
		}
	case idealInt:
		switch otherVal.(type) {
		case idealComplex:
			return idealComplex(complex(float64(x), 0))
		case idealFloat:
			return idealFloat(int64(x))
		case idealInt:
			return idealInt(int64(x))
		//case idealRune:
		case idealUint:
			if x >= 0 {
				return idealUint(int64(x))
			}
		//case bool:
		case complex64:
			return complex64(complex(float32(x), 0))
		case complex128:
			return complex128(complex(float64(x), 0))
		case float32:
			return float32(int64(x))
		case float64:
			return float64(int64(x))
		case int8:
			if x >= math.MinInt8 && x <= math.MaxInt8 {
				return int8(int64(x))
			}
		case int16:
			if x >= math.MinInt16 && x <= math.MaxInt16 {
				return int16(int64(x))
			}
		case int32:
			if x >= math.MinInt32 && x <= math.MaxInt32 {
				return int32(int64(x))
			}
		case int64:
			return int64(int64(x))
		//case string:
		case uint8:
			if x >= 0 && x <= math.MaxUint8 {
				return uint8(int64(x))
			}
		case uint16:
			if x >= 0 && x <= math.MaxUint16 {
				return uint16(int64(x))
			}
		case uint32:
			if x >= 0 && x <= math.MaxUint32 {
				return uint32(int64(x))
			}
		case uint64:
			if x >= 0 {
				return uint64(int64(x))
			}
		case *big.Int:
			return big.NewInt(int64(x))
		case *big.Rat:
			return big.NewRat(1, 1).SetInt64(int64(x))
		//case time.Time:
		case time.Duration:
			return time.Duration(int64(x))
		}
	case idealRune:
		switch otherVal.(type) {
		case idealComplex:
			return idealComplex(complex(float64(x), 0))
		case idealFloat:
			return idealFloat(int64(x))
		case idealInt:
			return idealInt(int64(x))
		case idealRune:
			return idealRune(int64(x))
		case idealUint:
			return idealUint(int64(x))
		//case bool:
		case complex64:
			return complex64(complex(float32(x), 0))
		case complex128:
			return complex128(complex(float64(x), 0))
		case float32:
			return float32(int64(x))
		case float64:
			return float64(int64(x))
		case int8:
			return int8(int64(x))
		case int16:
			return int16(int64(x))
		case int32:
			return int32(int64(x))
		case int64:
			return int64(int64(x))
		//case string:
		case uint8:
			return uint8(int64(x))
		case uint16:
			return uint16(int64(x))
		case uint32:
			return uint32(int64(x))
		case uint64:
			return uint64(int64(x))
		case *big.Int:
			return big.NewInt(int64(x))
		case *big.Rat:
			return big.NewRat(1, 1).SetInt64(int64(x))
		//case time.Time:
		case time.Duration:
			return time.Duration(int64(x))
		}
	case idealUint:
		switch otherVal.(type) {
		case idealComplex:
			return idealComplex(complex(float64(x), 0))
		case idealFloat:
			return idealFloat(uint64(x))
		case idealInt:
			if x <= math.MaxInt64 {
				return idealInt(int64(x))
			}
		//case idealRune:
		case idealUint:
			return idealUint(uint64(x))
		//case bool:
		case complex64:
			return complex64(complex(float32(x), 0))
		case complex128:
			return complex128(complex(float64(x), 0))
		case float32:
			return float32(uint64(x))
		case float64:
			return float64(uint64(x))
		case int8:
			if x <= math.MaxInt8 {
				return int8(int64(x))
			}
		case int16:
			if x <= math.MaxInt16 {
				return int16(int64(x))
			}
		case int32:
			if x <= math.MaxInt32 {
				return int32(int64(x))
			}
		case int64:
			if x <= math.MaxInt64 {
				return int64(int64(x))
			}
		//case string:
		case uint8:
			if x >= 0 && x <= math.MaxUint8 {
				return uint8(int64(x))
			}
		case uint16:
			if x >= 0 && x <= math.MaxUint16 {
				return uint16(int64(x))
			}
		case uint32:
			if x >= 0 && x <= math.MaxUint32 {
				return uint32(int64(x))
			}
		case uint64:
			return uint64(uint64(x))
		case *big.Int:
			return big.NewInt(0).SetUint64(uint64(x))
		case *big.Rat:
			return big.NewRat(1, 1).SetInt(big.NewInt(0).SetUint64(uint64(x)))
		//case time.Time:
		case time.Duration:
			if x <= math.MaxInt64 {
				return time.Duration(int64(x))
			}
		}
	}
	return
}
