// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"strings"
	"time"
)

// QL types.
const (
	qBool       = 0x62 // 'b'
	qComplex64  = 0x63 // 'c'
	qComplex128 = 0x64 // 'd'
	qFloat32    = 0x66 // 'f'
	qFloat64    = 0x67 // 'g', alias float
	qInt8       = 0x69 // 'i'
	qInt16      = 0x6a // 'j'
	qInt32      = 0x6b // 'k'
	qInt64      = 0x6c // 'l', alias int
	qString     = 0x73 // 's'
	qUint8      = 0x75 // 'u', alias byte
	qUint16     = 0x76 // 'v'
	qUint32     = 0x77 // 'w'
	qUint64     = 0x78 // 'x', alias uint

	qBigInt   = 0x49 // 'I'
	qBigRat   = 0x52 // 'R'
	qBlob     = 0x42 // 'B'
	qDuration = 0x44 // 'D'
	qTime     = 0x54 // 'T'
)

var (
	type2Str = map[int]string{
		qBigInt:     "bigint",
		qBigRat:     "bigrat",
		qBlob:       "blob",
		qBool:       "bool",
		qComplex128: "complex128",
		qComplex64:  "complex64",
		qDuration:   "duration",
		qFloat32:    "float32",
		qFloat64:    "float64",
		qInt16:      "int16",
		qInt32:      "int32",
		qInt64:      "int64",
		qInt8:       "int8",
		qString:     "string",
		qTime:       "time",
		qUint16:     "uint16",
		qUint32:     "uint32",
		qUint64:     "uint64",
		qUint8:      "uint8",
	}
)

func typeStr(typ int) (r string) {
	return type2Str[typ]
}

func noEOF(err error) error {
	if err == io.EOF {
		err = nil
	}
	return err
}

func runErr(err error) error { return fmt.Errorf("run time error: %s", err) }

func invXOp(s, x interface{}) error {
	return fmt.Errorf("invalid operation: %v[%v] (index of type %T)", s, x, x)
}

func invSOp(s interface{}) error {
	return fmt.Errorf("cannot slice %s (type %T)", s, s)
}

func invNegX(x interface{}) error {
	return fmt.Errorf("invalid string index %v (index must be non-negative)", x)
}

func invNegLO(x interface{}) error {
	return fmt.Errorf("invalid LIMIT or OFFSET value %v (must be non-negative)", x)
}

func invSliceNegX(x interface{}) error {
	return fmt.Errorf("invalid slice index %v (index must be non-negative)", x)
}

func invBoundX(s string, x uint64) error {
	return fmt.Errorf("invalid string index %d (out of bounds for %d-byte string)", x, len(s))
}

func invSliceBoundX(s string, x uint64) error {
	return fmt.Errorf("invalid slice index %d (out of bounds for %d-byte string)", x, len(s))
}

func intExpr(x interface{}) (i int64, err error) {
	switch x := x.(type) {
	case idealInt:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case idealRune:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case idealUint:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case int8:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case int16:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case int32:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case int64:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return int64(x), nil
	case uint8:
		return int64(x), nil
	case uint16:
		return int64(x), nil
	case uint32:
		return int64(x), nil
	case uint64:
		return int64(x), nil
	default:
		return 0, fmt.Errorf("non-integer expression: %v (value of type %T)", x, x)
	}
}

func limOffExpr(x interface{}) (i uint64, err error) {
	switch x := x.(type) {
	case idealInt:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case idealRune:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case idealUint:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case int8:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case int16:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case int32:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case int64:
		if x < 0 {
			return 0, invNegLO(x)
		}

		return uint64(x), nil
	case uint8:
		return uint64(x), nil
	case uint16:
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	case uint64:
		return uint64(x), nil
	default:
		return 0, fmt.Errorf("non-integer used in LIMIT or OFFSET: %v (value of type %T)", x, x)
	}
}

func indexExpr(s *string, x interface{}) (i uint64, err error) {
	switch x := x.(type) {
	case idealFloat:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealInt:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int64(x) >= int64(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealRune:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int32(x) >= int32(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealUint:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && uint64(x) >= uint64(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int8:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int16:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int32:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int64:
		if x < 0 {
			return 0, invNegX(x)
		}

		if s != nil && x >= int64(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint8:
		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint16:
		if s != nil && int(x) >= len(*s) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint32:
		if s != nil && x >= uint32(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint64:
		if s != nil && x >= uint64(len(*s)) {
			return 0, invBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	default:
		return 0, fmt.Errorf("non-integer string index %v", x)
	}
}

func sliceExpr(s *string, x interface{}, mod int) (i uint64, err error) {
	switch x := x.(type) {
	case idealFloat:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealInt:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int64(x) >= int64(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealRune:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int32(x) >= int32(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case idealUint:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && uint64(x) >= uint64(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int8:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int16:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int32:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case int64:
		if x < 0 {
			return 0, invSliceNegX(x)
		}

		if s != nil && x >= int64(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint8:
		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint16:
		if s != nil && int(x) >= len(*s)+mod {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint32:
		if s != nil && x >= uint32(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	case uint64:
		if s != nil && x >= uint64(len(*s)+mod) {
			return 0, invSliceBoundX(*s, uint64(x))
		}

		return uint64(x), nil
	default:
		return 0, fmt.Errorf("invalid slice index %s (type %T)", x, x)
	}
}

type iop int

func (o iop) String() string {
	switch i := int(o); i {
	case andand:
		return "&&"
	case andnot:
		return "&^"
	case lsh:
		return "<<"
	case le:
		return "<="
	case eq:
		return "=="
	case ge:
		return ">="
	case neq:
		return "!="
	case oror:
		return "||"
	case rsh:
		return ">>"
	default:
		return string(i)
	}
}

func ideal(v interface{}) interface{} {
	switch x := v.(type) {
	case idealComplex:
		return complex128(x)
	case idealFloat:
		return float64(x)
	case idealInt:
		return int64(x)
	case idealRune:
		return int64(x)
	case idealUint:
		return uint64(x)
	default:
		return v
	}
}

func eval(v expression, execCtx *execCtx, ctx map[interface{}]interface{}, arg []interface{}) (y interface{}) {
	y, err := expand1(v.eval(execCtx, ctx, arg))
	if err != nil {
		panic(err) // panic ok here
	}
	return
}

func eval2(a, b expression, execCtx *execCtx, ctx map[interface{}]interface{}, arg []interface{}) (x, y interface{}) {
	return eval(a, execCtx, ctx, arg), eval(b, execCtx, ctx, arg)
}

func invOp2(x, y interface{}, o int) (interface{}, error) {
	return nil, fmt.Errorf("invalid operation: %v %v %v (mismatched types %T and %T)", x, iop(o), y, ideal(x), ideal(y))
}

func undOp(x interface{}, o int) (interface{}, error) {
	return nil, fmt.Errorf("invalid operation: %v%v (operator %v not defined on %T)", iop(o), x, iop(o), x)
}

func undOp2(x, y interface{}, o int) (interface{}, error) {
	return nil, fmt.Errorf("invalid operation: %v %v %v (operator %v not defined on %T)", x, iop(o), y, iop(o), x)
}

func invConv(val interface{}, typ int) (interface{}, error) {
	return nil, fmt.Errorf("cannot convert %v (type %T) to type %s", val, val, typeStr(typ))
}

func truncConv(val interface{}) (interface{}, error) {
	return nil, fmt.Errorf("constant %v truncated to integer", val)
}

func convert(val interface{}, typ int) (v interface{}, err error) { //NTYPE
	if val == nil {
		return nil, nil
	}

	switch typ {
	case qBool:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		//case idealFloat:
		//case idealInt:
		//case idealRune:
		//case idealUint:
		case bool:
			return bool(x), nil
		//case complex64:
		//case complex128:
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
		default:
			return invConv(val, typ)
		}
	case qComplex64:
		switch x := val.(type) {
		//case nil:
		case idealComplex:
			return complex64(x), nil
		case idealFloat:
			return complex(float32(x), 0), nil
		case idealInt:
			return complex(float32(x), 0), nil
		case idealRune:
			return complex(float32(x), 0), nil
		case idealUint:
			return complex(float32(x), 0), nil
		//case bool:
		case complex64:
			return complex64(x), nil
		case complex128:
			return complex64(x), nil
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
		default:
			return invConv(val, typ)
		}
	case qComplex128:
		switch x := val.(type) {
		//case nil:
		case idealComplex:
			return complex128(x), nil
		case idealFloat:
			return complex(float64(x), 0), nil
		case idealInt:
			return complex(float64(x), 0), nil
		case idealRune:
			return complex(float64(x), 0), nil
		case idealUint:
			return complex(float64(x), 0), nil
		//case bool:
		case complex64:
			return complex128(x), nil
		case complex128:
			return complex128(x), nil
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
		default:
			return invConv(val, typ)
		}
	case qFloat32:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			return float32(x), nil
		case idealInt:
			return float32(x), nil
		case idealRune:
			return float32(x), nil
		case idealUint:
			return float32(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return float32(x), nil
		case float64:
			return float32(x), nil
		case int8:
			return float32(x), nil
		case int16:
			return float32(x), nil
		case int32:
			return float32(x), nil
		case int64:
			return float32(x), nil
		//case string:
		case uint8:
			return float32(x), nil
		case uint16:
			return float32(x), nil
		case uint32:
			return float32(x), nil
		case uint64:
			return float32(x), nil
		case *big.Int:
			v, _ := big.NewRat(1, 1).SetInt(x).Float64()
			return float32(v), nil
		case *big.Rat:
			v, _ := x.Float64()
			return float32(v), nil
		case time.Duration:
			return float32(x), nil
		default:
			return invConv(val, typ)
		}
	case qFloat64:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			return float64(x), nil
		case idealInt:
			return float64(x), nil
		case idealRune:
			return float64(x), nil
		case idealUint:
			return float64(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return float64(x), nil
		case float64:
			return float64(x), nil
		case int8:
			return float64(x), nil
		case int16:
			return float64(x), nil
		case int32:
			return float64(x), nil
		case int64:
			return float64(x), nil
		//case string:
		case uint8:
			return float64(x), nil
		case uint16:
			return float64(x), nil
		case uint32:
			return float64(x), nil
		case uint64:
			return float64(x), nil
		case *big.Int:
			v, _ := big.NewRat(1, 1).SetInt(x).Float64()
			return v, nil
		case *big.Rat:
			v, _ := x.Float64()
			return v, nil
		case time.Duration:
			return float64(x), nil
		default:
			return invConv(val, typ)
		}
	case qInt8:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return int8(x), nil
		case idealInt:
			return int8(x), nil
		case idealRune:
			return int8(x), nil
		case idealUint:
			return int8(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return int8(x), nil
		case float64:
			return int8(x), nil
		case int8:
			return int8(x), nil
		case int16:
			return int8(x), nil
		case int32:
			return int8(x), nil
		case int64:
			return int8(x), nil
		//case string:
		case uint8:
			return int8(x), nil
		case uint16:
			return int8(x), nil
		case uint32:
			return int8(x), nil
		case uint64:
			return int8(x), nil
		case *big.Int:
			return int8(x.Int64()), nil
		case time.Duration:
			return int8(x), nil
		default:
			return invConv(val, typ)
		}
	case qInt16:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return int16(x), nil
		case idealInt:
			return int16(x), nil
		case idealRune:
			return int16(x), nil
		case idealUint:
			return int16(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return int16(x), nil
		case float64:
			return int16(x), nil
		case int8:
			return int16(x), nil
		case int16:
			return int16(x), nil
		case int32:
			return int16(x), nil
		case int64:
			return int16(x), nil
		//case string:
		case uint8:
			return int16(x), nil
		case uint16:
			return int16(x), nil
		case uint32:
			return int16(x), nil
		case uint64:
			return int16(x), nil
		case *big.Int:
			return int16(x.Int64()), nil
		case time.Duration:
			return int16(x), nil
		default:
			return invConv(val, typ)
		}
	case qInt32:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return int32(x), nil
		case idealInt:
			return int32(x), nil
		case idealRune:
			return int32(x), nil
		case idealUint:
			return int32(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return int32(x), nil
		case float64:
			return int32(x), nil
		case int8:
			return int32(x), nil
		case int16:
			return int32(x), nil
		case int32:
			return int32(x), nil
		case int64:
			return int32(x), nil
		//case string:
		case uint8:
			return int32(x), nil
		case uint16:
			return int32(x), nil
		case uint32:
			return int32(x), nil
		case uint64:
			return int32(x), nil
		case *big.Int:
			return int32(x.Int64()), nil
		case time.Duration:
			return int32(x), nil
		default:
			return invConv(val, typ)
		}
	case qInt64:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return int64(x), nil
		case idealInt:
			return int64(x), nil
		case idealRune:
			return int64(x), nil
		case idealUint:
			return int64(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return int64(x), nil
		case float64:
			return int64(x), nil
		case int8:
			return int64(x), nil
		case int16:
			return int64(x), nil
		case int32:
			return int64(x), nil
		case int64:
			return int64(x), nil
		//case string:
		case uint8:
			return int64(x), nil
		case uint16:
			return int64(x), nil
		case uint32:
			return int64(x), nil
		case uint64:
			return int64(x), nil
		case *big.Int:
			return x.Int64(), nil
		case time.Duration:
			return int64(x), nil
		default:
			return invConv(val, typ)
		}
	case qString:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		//case idealFloat:
		case idealInt:
			return string(x), nil
		case idealRune:
			return string(x), nil
		case idealUint:
			return string(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		//case float32:
		//case float64:
		case int8:
			return string(x), nil
		case int16:
			return string(x), nil
		case int32:
			return string(x), nil
		case int64:
			return string(x), nil
		case string:
			return string(x), nil
		case uint8:
			return string(x), nil
		case uint16:
			return string(x), nil
		case uint32:
			return string(x), nil
		case uint64:
			return string(x), nil
		case []byte:
			return string(x), nil
		case *big.Int:
			return x.String(), nil
		case time.Time:
			return x.String(), nil
		case time.Duration:
			return x.String(), nil
		default:
			return invConv(val, typ)
		}
	case qUint8:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return uint8(x), nil
		case idealInt:
			return uint8(x), nil
		case idealRune:
			return uint8(x), nil
		case idealUint:
			return uint8(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return uint8(x), nil
		case float64:
			return uint8(x), nil
		case int8:
			return uint8(x), nil
		case int16:
			return uint8(x), nil
		case int32:
			return uint8(x), nil
		case int64:
			return uint8(x), nil
		//case string:
		case uint8:
			return uint8(x), nil
		case uint16:
			return uint8(x), nil
		case uint32:
			return uint8(x), nil
		case uint64:
			return uint8(x), nil
		case *big.Int:
			return uint8(x.Int64()), nil
		case time.Duration:
			return uint8(x), nil
		default:
			return invConv(val, typ)
		}
	case qUint16:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return uint16(x), nil
		case idealInt:
			return uint16(x), nil
		case idealRune:
			return uint16(x), nil
		case idealUint:
			return uint16(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return uint16(x), nil
		case float64:
			return uint16(x), nil
		case int8:
			return uint16(x), nil
		case int16:
			return uint16(x), nil
		case int32:
			return uint16(x), nil
		case int64:
			return uint16(x), nil
		//case string:
		case uint8:
			return uint16(x), nil
		case uint16:
			return uint16(x), nil
		case uint32:
			return uint16(x), nil
		case uint64:
			return uint16(x), nil
		case *big.Int:
			return uint16(x.Int64()), nil
		case time.Duration:
			return uint16(x), nil
		default:
			return invConv(val, typ)
		}
	case qUint32:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return uint32(x), nil
		case idealInt:
			return uint32(x), nil
		case idealRune:
			return uint32(x), nil
		case idealUint:
			return uint32(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return uint32(x), nil
		case float64:
			return uint32(x), nil
		case int8:
			return uint32(x), nil
		case int16:
			return uint32(x), nil
		case int32:
			return uint32(x), nil
		case int64:
			return uint32(x), nil
		//case string:
		case uint8:
			return uint32(x), nil
		case uint16:
			return uint32(x), nil
		case uint32:
			return uint32(x), nil
		case uint64:
			return uint32(x), nil
		case *big.Int:
			return uint32(x.Int64()), nil
		case time.Duration:
			return uint32(x), nil
		default:
			return invConv(val, typ)
		}
	case qUint64:
		switch x := val.(type) {
		//case nil:
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			return uint64(x), nil
		case idealInt:
			return uint64(x), nil
		case idealRune:
			return uint64(x), nil
		case idealUint:
			return uint64(x), nil
		//case bool:
		//case complex64:
		//case complex128:
		case float32:
			return uint64(x), nil
		case float64:
			return uint64(x), nil
		case int8:
			return uint64(x), nil
		case int16:
			return uint64(x), nil
		case int32:
			return uint64(x), nil
		case int64:
			return uint64(x), nil
		//case string:
		case uint8:
			return uint64(x), nil
		case uint16:
			return uint64(x), nil
		case uint32:
			return uint64(x), nil
		case uint64:
			return uint64(x), nil
		case *big.Int:
			return x.Uint64(), nil
		case time.Duration:
			return uint64(x), nil
		default:
			return invConv(val, typ)
		}
	case qBlob:
		switch x := val.(type) {
		case string:
			return []byte(x), nil
		case []byte:
			return x, nil
		default:
			return invConv(val, typ)
		}
	case qBigInt:
		switch x := val.(type) {
		// case blob
		// case bool
		//case idealComplex:
		case idealFloat:
			if _, frac := math.Modf(float64(x)); frac != 0 {
				return truncConv(x)
			}

			rr := big.NewRat(1, 1).SetFloat64(float64(x))
			ii := big.NewInt(0).Set(rr.Num())
			ii.Quo(ii, rr.Denom())
			return ii, nil
		case idealInt:
			return big.NewInt(0).SetInt64(int64(x)), nil
		case idealRune:
			return big.NewInt(0).SetInt64(int64(x)), nil
		case idealUint:
			return big.NewInt(0).SetUint64(uint64(x)), nil
		//case complex64
		//case complex128
		case float32:
			rr := big.NewRat(1, 1).SetFloat64(float64(x))
			ii := big.NewInt(0).Set(rr.Num())
			ii.Quo(ii, rr.Denom())
			return ii, nil
		case float64:
			rr := big.NewRat(1, 1).SetFloat64(float64(x))
			ii := big.NewInt(0).Set(rr.Num())
			ii.Quo(ii, rr.Denom())
			return ii, nil
		case int8:
			return big.NewInt(0).SetInt64(int64(x)), nil
		case int16:
			return big.NewInt(0).SetInt64(int64(x)), nil
		case int32:
			return big.NewInt(0).SetInt64(int64(x)), nil
		case int64:
			return big.NewInt(0).SetInt64(x), nil
		case string:
			y := big.NewInt(0)
			if _, ok := y.SetString(x, 0); !ok {
				return invConv(val, typ)
			}

			return y, nil
		case uint8:
			return big.NewInt(0).SetUint64(uint64(x)), nil
		case uint16:
			return big.NewInt(0).SetUint64(uint64(x)), nil
		case uint32:
			return big.NewInt(0).SetUint64(uint64(x)), nil
		case uint64:
			return big.NewInt(0).SetUint64(x), nil
		case *big.Int:
			return x, nil
		case *big.Rat:
			ii := big.NewInt(0).Set(x.Num())
			ii.Div(ii, x.Denom())
			return ii, nil
		default:
			return invConv(val, typ)
		}
	case qBigRat:
		switch x := val.(type) {
		// case blob
		// case bool
		//case idealComplex:
		case idealFloat:
			return big.NewRat(1, 1).SetFloat64(float64(x)), nil
		case idealInt:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case idealRune:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case idealUint:
			return big.NewRat(1, 1).SetInt(big.NewInt(0).SetUint64(uint64(x))), nil
		//case complex64
		//case complex128
		case float32:
			return big.NewRat(1, 1).SetFloat64(float64(x)), nil
		case float64:
			return big.NewRat(1, 1).SetFloat64(x), nil
		case int8:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case int16:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case int32:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case int64:
			return big.NewRat(1, 1).SetInt64(x), nil
		case string:
			y := big.NewRat(1, 1)
			if _, ok := y.SetString(x); !ok {
				return invConv(val, typ)
			}

			return y, nil
		case uint8:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case uint16:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case uint32:
			return big.NewRat(1, 1).SetInt64(int64(x)), nil
		case uint64:
			return big.NewRat(1, 1).SetInt(big.NewInt(0).SetUint64(x)), nil
		case *big.Int:
			return big.NewRat(1, 1).SetInt(x), nil
		case *big.Rat:
			return x, nil
		default:
			return invConv(val, typ)
		}
	case qDuration:
		switch x := val.(type) {
		// case blob
		// case bool
		//case idealComplex:
		case idealFloat:
			return time.Duration(x), nil
		case idealInt:
			return time.Duration(x), nil
		case idealRune:
			return time.Duration(x), nil
		case idealUint:
			return time.Duration(x), nil
		//case complex64
		//case complex128
		case float32:
			return time.Duration(x), nil
		case float64:
			return time.Duration(x), nil
		case int8:
			return time.Duration(x), nil
		case int16:
			return time.Duration(x), nil
		case int32:
			return time.Duration(x), nil
		case int64:
			return time.Duration(x), nil
		case string:
			return time.ParseDuration(x)
		case uint8:
			return time.Duration(x), nil
		case uint16:
			return time.Duration(x), nil
		case uint32:
			return time.Duration(x), nil
		case uint64:
			return time.Duration(x), nil
		case *big.Int:
			return time.Duration(x.Int64()), nil
		case *big.Rat:
			f, _ := x.Float64()
			return time.Duration(f), nil
		case time.Duration:
			return x, nil
		default:
			return invConv(val, typ)
		}
	case qTime:
		switch x := val.(type) {
		// case blob
		// case bool
		//case idealComplex:
		//case idealFloat:
		//case idealInt:
		//case idealRune:
		//case idealUint:
		//case complex64
		//case complex128
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
		//case time.Duration:
		case time.Time:
			return x, nil
		default:
			return invConv(val, typ)
		}
	default:
		log.Panic("internal error 006")
	}
	//dbg("%T(%v) %s", val, val, typeStr(typ))
	panic("unreachable")
}

func invShiftRHS(lhs, rhs interface{}) (interface{}, error) {
	return nil, fmt.Errorf("invalid operation: %v << %v (shift count type %T, must be unsigned integer)", lhs, rhs, rhs)
}

func invTruncInt(v interface{}) error {
	return fmt.Errorf("constant %v truncated to integer", v)
}

func overflow(v interface{}, typ int) error {
	return fmt.Errorf("constant %v overflows %s", v, typeStr(typ))
}

func typeCheck(rec []interface{}, cols []*col) (err error) {
	for _, c := range cols {
		i := c.index
		if v := rec[i]; !c.typeCheck(v) {
			switch v.(type) {
			case idealComplex:
				y := complex128(v.(idealComplex))
				switch c.typ {
				case qBool:
				case qComplex64:
					rec[i] = complex64(y)
					continue
				case qComplex128:
					rec[i] = complex128(y)
					continue
				case qFloat32, qFloat64, qInt8, qInt16, qInt32, qInt64, qUint8, qUint16, qUint32, qUint64:
					return fmt.Errorf("constant %v truncated to real", y)
				case qString:
				default:
					log.Panic("internal error 007")
				}
			case idealFloat:
				y := float64(v.(idealFloat))
				switch c.typ {
				case qBool:
				case qComplex64:
					rec[i] = complex(float32(y), 0)
					continue
				case qComplex128:
					rec[i] = complex(float64(y), 0)
					continue
				case qFloat32:
					rec[i] = float32(y)
					continue
				case qFloat64:
					rec[i] = float64(y)
					continue
				case qInt8:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < math.MinInt8 || y > math.MaxInt8 {
						return overflow(y, c.typ)
					}

					rec[i] = int8(y)
					continue
				case qInt16:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < math.MinInt16 || y > math.MaxInt16 {
						return overflow(y, c.typ)
					}

					rec[i] = int16(y)
					continue
				case qInt32:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < math.MinInt32 || y > math.MaxInt32 {
						return overflow(y, c.typ)
					}

					rec[i] = int32(y)
					continue
				case qInt64:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < math.MinInt64 || y > math.MaxInt64 {
						return overflow(y, c.typ)
					}

					rec[i] = int64(y)
					continue
				case qString:
				case qUint8:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < 0 || y > math.MaxUint8 {
						return overflow(y, c.typ)
					}

					rec[i] = uint8(y)
					continue
				case qUint16:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < 0 || y > math.MaxUint16 {
						return overflow(y, c.typ)
					}

					rec[i] = uint16(y)
					continue
				case qUint32:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < 0 || y > math.MaxUint32 {
						return overflow(y, c.typ)
					}

					rec[i] = uint32(y)
					continue
				case qUint64:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					if y < 0 || y > math.MaxUint64 {
						return overflow(y, c.typ)
					}

					rec[i] = uint64(y)
					continue
				case qBigInt:
					if math.Floor(y) != y {
						return invTruncInt(y)
					}

					rr := big.NewRat(1, 1).SetFloat64(y)
					ii := big.NewInt(0)
					ii.Set(rr.Num())
					ii.Quo(ii, rr.Denom())
					rec[i] = ii
					continue
				case qBigRat:
					rec[i] = big.NewRat(1, 1).SetFloat64(y)
					continue
				default:
					log.Panic("internal error 008")
				}
			case idealInt:
				y := int64(v.(idealInt))
				switch c.typ {
				case qBool:
				case qComplex64:
					rec[i] = complex(float32(y), 0)
					continue
				case qComplex128:
					rec[i] = complex(float64(y), 0)
					continue
				case qFloat32:
					rec[i] = float32(y)
					continue
				case qFloat64:
					rec[i] = float64(y)
					continue
				case qInt8:
					if y < math.MinInt8 || y > math.MaxInt8 {
						return overflow(y, c.typ)
					}

					rec[i] = int8(y)
					continue
				case qInt16:
					if y < math.MinInt16 || y > math.MaxInt16 {
						return overflow(y, c.typ)
					}

					rec[i] = int16(y)
					continue
				case qInt32:
					if y < math.MinInt32 || y > math.MaxInt32 {
						return overflow(y, c.typ)
					}

					rec[i] = int32(y)
					continue
				case qInt64:
					if y < math.MinInt64 || y > math.MaxInt64 {
						return overflow(y, c.typ)
					}

					rec[i] = int64(y)
					continue
				case qString:
				case qUint8:
					if y < 0 || y > math.MaxUint8 {
						return overflow(y, c.typ)
					}

					rec[i] = uint8(y)
					continue
				case qUint16:
					if y < 0 || y > math.MaxUint16 {
						return overflow(y, c.typ)
					}

					rec[i] = uint16(y)
					continue
				case qUint32:
					if y < 0 || y > math.MaxUint32 {
						return overflow(y, c.typ)
					}

					rec[i] = uint32(y)
					continue
				case qUint64:
					if y < 0 {
						return overflow(y, c.typ)
					}

					rec[i] = uint64(y)
					continue
				case qBigInt:
					rec[i] = big.NewInt(y)
					continue
				case qBigRat:
					rec[i] = big.NewRat(1, 1).SetInt64(y)
					continue
				default:
					log.Panic("internal error 009")
				}
			case idealRune:
				y := int64(v.(idealRune))
				switch c.typ {
				case qBool:
				case qComplex64:
					rec[i] = complex(float32(y), 0)
					continue
				case qComplex128:
					rec[i] = complex(float64(y), 0)
					continue
				case qFloat32:
					rec[i] = float32(y)
					continue
				case qFloat64:
					rec[i] = float64(y)
					continue
				case qInt8:
					if y < math.MinInt8 || y > math.MaxInt8 {
						return overflow(y, c.typ)
					}

					rec[i] = int8(y)
					continue
				case qInt16:
					if y < math.MinInt16 || y > math.MaxInt16 {
						return overflow(y, c.typ)
					}

					rec[i] = int16(y)
					continue
				case qInt32:
					if y < math.MinInt32 || y > math.MaxInt32 {
						return overflow(y, c.typ)
					}

					rec[i] = int32(y)
					continue
				case qInt64:
					if y < math.MinInt64 || y > math.MaxInt64 {
						return overflow(y, c.typ)
					}

					rec[i] = int64(y)
					continue
				case qString:
				case qUint8:
					if y < 0 || y > math.MaxUint8 {
						return overflow(y, c.typ)
					}

					rec[i] = uint8(y)
					continue
				case qUint16:
					if y < 0 || y > math.MaxUint16 {
						return overflow(y, c.typ)
					}

					rec[i] = uint16(y)
					continue
				case qUint32:
					if y < 0 {
						return overflow(y, c.typ)
					}

					rec[i] = uint32(y)
					continue
				case qUint64:
					if y < 0 {
						return overflow(y, c.typ)
					}

					rec[i] = uint64(y)
					continue
				case qBigInt:
					rec[i] = big.NewInt(y)
					continue
				case qBigRat:
					rec[i] = big.NewRat(1, 1).SetInt64(y)
					continue
				default:
					log.Panic("internal error 010")
				}
			case idealUint:
				y := uint64(v.(idealUint))
				switch c.typ {
				case qBool:
				case qComplex64:
					rec[i] = complex(float32(y), 0)
					continue
				case qComplex128:
					rec[i] = complex(float64(y), 0)
					continue
				case qFloat32:
					rec[i] = float32(y)
					continue
				case qFloat64:
					rec[i] = float64(y)
					continue
				case qInt8:
					if y > math.MaxInt8 {
						return overflow(y, c.typ)
					}

					rec[i] = int8(y)
					continue
				case qInt16:
					if y > math.MaxInt16 {
						return overflow(y, c.typ)
					}

					rec[i] = int16(y)
					continue
				case qInt32:
					if y > math.MaxInt32 {
						return overflow(y, c.typ)
					}

					rec[i] = int32(y)
					continue
				case qInt64:
					if y > math.MaxInt64 {
						return overflow(y, c.typ)
					}

					rec[i] = int64(y)
					continue
				case qString:
					rec[i] = string(y)
					continue
				case qUint8:
					if y > math.MaxUint8 {
						return overflow(y, c.typ)
					}

					rec[i] = uint8(y)
					continue
				case qUint16:
					if y > math.MaxUint16 {
						return overflow(y, c.typ)
					}

					rec[i] = uint16(y)
					continue
				case qUint32:
					if y > math.MaxUint32 {
						return overflow(y, c.typ)
					}

					rec[i] = uint32(y)
					continue
				case qUint64:
					rec[i] = uint64(y)
					continue
				case qBigInt:
					rec[i] = big.NewInt(0).SetUint64(y)
					continue
				case qBigRat:
					ii := big.NewInt(0).SetUint64(y)
					rec[i] = big.NewRat(1, 1).SetInt(ii)
					continue
				default:
					log.Panic("internal error 011")
				}
			}
			//dbg("v %T(%v), typ %s", v, v, typeStr(c.typ))
			return fmt.Errorf("cannot use %v (type %T) as %s in assignment/comparison to/with column %s", v, ideal(v), typeStr(c.typ), c.name)
		}
	}
	return
}

//TODO collate1 should return errors instead of panicing
func collate1(a, b interface{}) int {
	switch x := a.(type) {
	case nil:
		if b != nil {
			return -1
		}

		return 0
	case bool:
		switch y := b.(type) {
		case nil:
			return 1
		case bool:
			if !x && y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		default:
			// Make bool collate before anything except nil and
			// other bool for index seeking first non NULL value.
			return -1
		}
	case idealComplex:
		switch y := b.(type) {
		case nil:
			return 1
		case idealComplex:
			if x == y {
				return 0
			}

			if real(x) < real(y) {
				return -1
			}

			if real(x) > real(y) {
				return 1
			}

			if imag(x) < imag(y) {
				return -1
			}

			return 1
		case complex64:
			{
				x, y := complex64(x), complex64(y)
				if x == y {
					return 0
				}

				if real(x) < real(y) {
					return -1
				}

				if real(x) > real(y) {
					return 1
				}

				if imag(x) < imag(y) {
					return -1
				}

				return 1
			}
		case complex128:
			{
				x := complex128(x)
				if x == y {
					return 0
				}

				if real(x) < real(y) {
					return -1
				}

				if real(x) > real(y) {
					return 1
				}

				if imag(x) < imag(y) {
					return -1
				}

				return 1
			}
		default:
			panic("internal error 012")
		}
	case idealUint:
		switch y := b.(type) {
		case nil:
			return 1
		case idealUint:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case uint8:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case uint16:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case uint32:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case uint64:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case uint:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 013")
		}
	case idealRune:
		switch y := b.(type) {
		case nil:
			return 1
		case idealRune:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case int8:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int16:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int32:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int64:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 014")
		}
	case idealInt:
		switch y := b.(type) {
		case nil:
			return 1
		case idealInt:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case int8:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int16:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int32:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int64:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case int:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 015")
		}
	case idealFloat:
		switch y := b.(type) {
		case nil:
			return 1
		case idealFloat:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case float32:
			{
				x, y := float64(x), float64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case float64:
			{
				x, y := float64(x), float64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 016")
		}
	case complex64:
		switch y := b.(type) {
		case nil:
			return 1
		case complex64:
			if x == y {
				return 0
			}

			if real(x) < real(y) {
				return -1
			}

			if real(x) > real(y) {
				return 1
			}

			if imag(x) < imag(y) {
				return -1
			}

			return 1
		case idealComplex:
			{
				x, y := complex64(x), complex64(y)
				if x == y {
					return 0
				}

				if real(x) < real(y) {
					return -1
				}

				if real(x) > real(y) {
					return 1
				}

				if imag(x) < imag(y) {
					return -1
				}

				return 1
			}
		default:
			panic("internal error 017")
		}
	case complex128:
		switch y := b.(type) {
		case nil:
			return 1
		case complex128:
			if x == y {
				return 0
			}

			if real(x) < real(y) {
				return -1
			}

			if real(x) > real(y) {
				return 1
			}

			if imag(x) < imag(y) {
				return -1
			}

			return 1
		case idealComplex:
			{
				x, y := complex128(x), complex128(y)
				if x == y {
					return 0
				}

				if real(x) < real(y) {
					return -1
				}

				if real(x) > real(y) {
					return 1
				}

				if imag(x) < imag(y) {
					return -1
				}

				return 1
			}
		default:
			panic("internal error 018")
		}
	case float32:
		switch y := b.(type) {
		case nil:
			return 1
		case float32:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealFloat:
			{
				x, y := float32(x), float32(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 019")
		}
	case float64:
		switch y := b.(type) {
		case nil:
			return 1
		case float64:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealFloat:
			{
				x, y := float64(x), float64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 020")
		}
	case int8:
		switch y := b.(type) {
		case nil:
			return 1
		case int8:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 021")
		}
	case int16:
		switch y := b.(type) {
		case nil:
			return 1
		case int16:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 022")
		}
	case int32:
		switch y := b.(type) {
		case nil:
			return 1
		case int32:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 023")
		}
	case int64:
		switch y := b.(type) {
		case nil:
			return 1
		case int64:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := int64(x), int64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			//dbg("%T(%#v)", b, b)
			panic("internal error 024")
		}
	case uint8:
		switch y := b.(type) {
		case nil:
			return 1
		case uint8:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case idealUint:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 025")
		}
	case uint16:
		switch y := b.(type) {
		case nil:
			return 1
		case uint16:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case idealUint:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 026")
		}
	case uint32:
		switch y := b.(type) {
		case nil:
			return 1
		case uint32:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case idealUint:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 027")
		}
	case uint64:
		switch y := b.(type) {
		case nil:
			return 1
		case uint64:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		case idealInt:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		case idealUint:
			{
				x, y := uint64(x), uint64(y)
				if x < y {
					return -1
				}

				if x == y {
					return 0
				}

				return 1
			}
		default:
			panic("internal error 028")
		}
	case string:
		switch y := b.(type) {
		case nil:
			return 1
		case string:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		default:
			panic("internal error 029")
		}
	case []byte:
		switch y := b.(type) {
		case nil:
			return 1
		case []byte:
			return bytes.Compare(x, y)
		default:
			panic("internal error 030")
		}
	case *big.Int:
		switch y := b.(type) {
		case nil:
			return 1
		case *big.Int:
			return x.Cmp(y)
		case idealInt:
			{
				y := big.NewInt(int64(y))
				return x.Cmp(y)
			}
		case idealUint:
			{
				u := big.NewInt(0)
				u.SetUint64(uint64(y))
				return x.Cmp(u)
			}
		default:
			panic("internal error 031")
		}
	case *big.Rat:
		switch y := b.(type) {
		case nil:
			return 1
		case *big.Rat:
			return x.Cmp(y)
		case idealInt:
			{
				y := big.NewRat(int64(y), 1)
				return x.Cmp(y)
			}
		case idealUint:
			{
				u := big.NewInt(0)
				u.SetUint64(uint64(y))
				var y big.Rat
				y.SetInt(u)
				return x.Cmp(&y)
			}
		default:
			panic("internal error 032")
		}
	case time.Time:
		switch y := b.(type) {
		case nil:
			return 1
		case time.Time:
			if x.Before(y) {
				return -1
			}

			if x.Equal(y) {
				return 0
			}

			return 1
		default:
			panic("internal error 033")
		}
	case time.Duration:
		switch y := b.(type) {
		case nil:
			return 1
		case time.Duration:
			if x < y {
				return -1
			}

			if x == y {
				return 0
			}

			return 1
		default:
			panic("internal error 034")
		}
	case chunk:
		switch y := b.(type) {
		case nil:
			return 1
		case chunk:
			a, err := x.expand()
			if err != nil {
				log.Panic(err)
			}

			b, err := y.expand()
			if err != nil {
				log.Panic(err)
			}

			return collate1(a, b)
		default:
			panic("internal error 035")
		}
	default:
		//dbg("%T(%v) %T(%v)", a, a, b, b)
		panic("internal error 036")
	}
}

//TODO collate should return errors from collate1
func collate(x, y []interface{}) (r int) {
	//defer func() { dbg("%v %v -> %v", x, y, r) }()
	nx, ny := len(x), len(y)

	switch {
	case nx == 0 && ny != 0:
		return -1
	case nx == 0 && ny == 0:
		return 0
	case nx != 0 && ny == 0:
		return 1
	}

	r = 1
	if nx > ny {
		x, y, r = y, x, -r
	}

	for i, xi := range x {
		if c := collate1(xi, y[i]); c != 0 {
			return c * r
		}
	}

	if nx == ny {
		return 0
	}

	return -r
}

var collators = map[bool]func(a, b []interface{}) int{false: collateDesc, true: collate}

func collateDesc(a, b []interface{}) int {
	return -collate(a, b)
}

func isOrderedType(v interface{}) (y interface{}, r bool, err error) {
	//dbg("====")
	//dbg("%T(%v)", v, v)
	//defer func() { dbg("%T(%v)", y, y) }()
	switch x := v.(type) {
	case idealFloat, idealInt, idealRune, idealUint,
		float32, float64,
		int8, int16, int32, int64,
		uint8, uint16, uint32, uint64,
		string:
		return v, true, nil
	case *big.Int, *big.Rat, time.Time, time.Duration:
		return x, true, nil
	case chunk:
		if y, err = x.expand(); err != nil {
			return
		}

		return isOrderedType(y)
	}

	return v, false, nil
}

var isSystemName = map[string]bool{
	"__Column": true,
	"__Index":  true,
	"__Table":  true,
}

func qualifier(s string) string {
	if pos := strings.IndexByte(s, '.'); pos >= 0 {
		s = s[:pos]
	}
	return s
}

func mustQualifier(s string) string {
	q := qualifier(s)
	if q == s {
		panic("internal error 068")
	}

	return q
}

func selector(s string) string {
	if pos := strings.IndexByte(s, '.'); pos >= 0 {
		s = s[pos+1:]
	}
	return s
}

func mustSelector(s string) string {
	q := selector(s)
	if q == s {
		panic("internal error 069")
	}

	return q
}
