// +build ignore

// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
)

type t int

const (
	qNil t = iota
	idealComplex
	idealFloat
	idealInt
	idealRune
	idealUint
	qBool
	qComplex64
	qComplex128
	qFloat32
	qFloat64
	qInt8
	qInt16
	qInt32
	qInt64
	qString
	qUint8
	qUint16
	qUint32
	qUint64
	qBigInt
	qBigRat
	qTime
	qDuration

	qEnd
)

func (n t) String() string {
	switch n {
	case qNil:
		return "nil"
	case idealComplex:
		return "idealComplex"
	case idealFloat:
		return "idealFloat"
	case idealInt:
		return "idealInt"
	case idealRune:
		return "idealRune"
	case idealUint:
		return "idealUint"
	case qBool:
		return "bool"
	case qComplex64:
		return "complex64"
	case qComplex128:
		return "complex128"
	case qFloat32:
		return "float32"
	case qFloat64:
		return "float64"
	case qInt8:
		return "int8"
	case qInt16:
		return "int16"
	case qInt32:
		return "int32"
	case qInt64:
		return "int64"
	case qString:
		return "string"
	case qUint8:
		return "uint8"
	case qUint16:
		return "uint16"
	case qUint32:
		return "uint32"
	case qUint64:
		return "uint64"
	case qBigInt:
		return "*big.Int"
	case qBigRat:
		return "*big.Rat"
	case qTime:
		return "time.Time"
	case qDuration:
		return "time.Duration"
	default:
		panic("internal error 046")
	}
}

func coerceIdealComplex(typ t) string {
	switch typ {
	case qComplex64, qComplex128:
		return fmt.Sprintf("return %s(x)\n", typ)
	default:
		return ""
	}
}

func coerceIdealFloat(typ t) string {
	switch typ {
	case idealComplex:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case qComplex64:
		return fmt.Sprintf("return %s(complex(float32(x), 0))\n", typ)
	case qComplex128:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case idealFloat, qFloat32, qFloat64:
		return fmt.Sprintf("return %s(float64(x))\n", typ)
	case qBigRat:
		return fmt.Sprintf("return big.NewRat(1, 1).SetFloat64(float64(x))\n")
	default:
		return ""
	}
	return ""
}

func coerceIdealInt(typ t) string {
	switch typ {
	case idealComplex:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case qComplex64:
		return fmt.Sprintf("return %s(complex(float32(x), 0))\n", typ)
	case qComplex128:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case idealFloat, idealInt, qFloat32, qFloat64, qInt64:
		return fmt.Sprintf("return %s(int64(x))\n", typ)
	case idealUint:
		return fmt.Sprintf("if x >= 0 { return %s(int64(x)) }\n", typ)
	case qInt8:
		return fmt.Sprintf("if x >= math.MinInt8 && x<= math.MaxInt8 { return %s(int64(x)) }\n", typ)
	case qInt16:
		return fmt.Sprintf("if x >= math.MinInt16 && x<= math.MaxInt16 { return %s(int64(x)) }\n", typ)
	case qInt32:
		return fmt.Sprintf("if x >= math.MinInt32 && x<= math.MaxInt32 { return %s(int64(x)) }\n", typ)
	case qUint8:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint8 { return %s(int64(x)) }\n", typ)
	case qUint16:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint16 { return %s(int64(x)) }\n", typ)
	case qUint32:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint32 { return %s(int64(x)) }\n", typ)
	case qUint64:
		return fmt.Sprintf("if x >= 0 { return %s(int64(x)) }\n", typ)
	case qBigInt:
		return fmt.Sprintf("return big.NewInt(int64(x))\n")
	case qBigRat:
		return fmt.Sprintf("return big.NewRat(1, 1).SetInt64(int64(x))\n")
	case qDuration:
		return fmt.Sprintf("return time.Duration(int64(x))\n")
	default:
		return ""
	}
	return ""
}

func coerceIdealRune(typ t) string {
	switch typ {
	case idealComplex:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case qComplex64:
		return fmt.Sprintf("return %s(complex(float32(x), 0))\n", typ)
	case qComplex128:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case idealFloat, idealInt, idealRune, idealUint, qFloat32, qFloat64, qInt8, qInt16, qInt32, qInt64, qUint8, qUint16, qUint32, qUint64:
		return fmt.Sprintf("return %s(int64(x))\n", typ)
	case qBigInt:
		return fmt.Sprintf("return big.NewInt(int64(x))\n")
	case qBigRat:
		return fmt.Sprintf("return big.NewRat(1, 1).SetInt64(int64(x))\n")
	case qDuration:
		return fmt.Sprintf("return time.Duration(int64(x))\n")
	default:
		return ""
	}
	return ""
}

func coerceIdealUint(typ t) string {
	switch typ {
	case idealComplex:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case qComplex64:
		return fmt.Sprintf("return %s(complex(float32(x), 0))\n", typ)
	case qComplex128:
		return fmt.Sprintf("return %s(complex(float64(x), 0))\n", typ)
	case idealFloat, idealUint, qFloat32, qFloat64, qUint64:
		return fmt.Sprintf("return %s(uint64(x))\n", typ)
	case idealInt:
		return fmt.Sprintf("if x <= math.MaxInt64 { return %s(int64(x)) }\n", typ)
	case qInt8:
		return fmt.Sprintf("if x <= math.MaxInt8 { return %s(int64(x)) }\n", typ)
	case qInt16:
		return fmt.Sprintf("if  x<= math.MaxInt16 { return %s(int64(x)) }\n", typ)
	case qInt32:
		return fmt.Sprintf("if  x<= math.MaxInt32 { return %s(int64(x)) }\n", typ)
	case qInt64:
		return fmt.Sprintf("if  x<= math.MaxInt64 { return %s(int64(x)) }\n", typ)
	case qUint8:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint8 { return %s(int64(x)) }\n", typ)
	case qUint16:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint16 { return %s(int64(x)) }\n", typ)
	case qUint32:
		return fmt.Sprintf("if x >= 0 && x<= math.MaxUint32 { return %s(int64(x)) }\n", typ)
	case qBigInt:
		return fmt.Sprintf("return big.NewInt(0).SetUint64(uint64(x))\n")
	case qBigRat:
		return fmt.Sprintf("return big.NewRat(1, 1).SetInt(big.NewInt(0).SetUint64(uint64(x)))\n")
	case qDuration:
		return fmt.Sprintf("if x <= math.MaxInt64 { return time.Duration(int64(x)) }\n")
	default:
		return ""
	}
	return ""
}

func genCoerce1(w io.Writer, in t, f func(out t) string) {
	fmt.Fprintf(w, "\tcase %s:\n", in)
	fmt.Fprintf(w, "\t\tswitch otherVal.(type) {\n")

	for i := idealComplex; i < qEnd; i++ {
		s := f(i)
		switch s {
		case "":
			fmt.Fprintf(w, "\t\t//case %s:\n", i)
		default:
			fmt.Fprintf(w, "\t\tcase %s:\n", i)
			fmt.Fprintf(w, "\t\t\t%s", s)
		}
	}

	fmt.Fprintf(w, "\t\t}\n") // switch
}

func genCoerce(w io.Writer) {
	fmt.Fprintf(w,
		`
func coerce1(inVal, otherVal interface{}) (coercedInVal interface{}) {
	coercedInVal = inVal
	if otherVal == nil {
		return
	}

	switch x := inVal.(type) {
	case nil:
		return
`)
	genCoerce1(w, idealComplex, coerceIdealComplex)
	genCoerce1(w, idealFloat, coerceIdealFloat)
	genCoerce1(w, idealInt, coerceIdealInt)
	genCoerce1(w, idealRune, coerceIdealRune)
	genCoerce1(w, idealUint, coerceIdealUint)
	fmt.Fprintf(w, "\t}\n") // switch

	fmt.Fprintf(w, "\treturn\n}\n") // func
}

func main() {
	ofn := flag.String("o", "", "")
	flag.Parse()
	_, err := os.Stat(*ofn)
	if err == nil {
		log.Fatalf("%s exists", *ofn)
	}

	w := bufio.NewWriter(os.Stdout)
	if s := *ofn; s != "" {
		f, err := os.Create(s)
		if err != nil {
			log.Fatal(err)
		}

		defer f.Close()
		w = bufio.NewWriter(f)
	}
	defer w.Flush()

	fmt.Fprintf(w, `// Copyright 2013 The ql Authors. All rights reserved.
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
`)
	genCoerce(w)
}
