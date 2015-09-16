// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// All non-std package dependencies related to testing live in this file,
// so porting to different environment is easy (just update functions).
//
// This file sets up the variables used, including testInitFns.
// Each file should add initialization that should be performed
// after flags are parsed.
//
// init is a multi-step process:
//   - setup vars (handled by init functions in each file)
//   - parse flags
//   - setup derived vars (handled by pre-init registered functions - registered in init function)
//   - post init (handled by post-init registered functions - registered in init function)
// This way, no one has to manage carefully control the initialization
// using file names, etc.
//
// Tests which require external dependencies need the -tag=x parameter.
// They should be run as:
//    go test -tags=x -run=. <other parameters ...>
// Benchmarks should also take this parameter, to include the sereal, xdr, etc.
// To run against codecgen, etc, make sure you pass extra parameters.
// Example usage:
//    go test "-tags=x codecgen unsafe" -bench=. <other parameters ...>
//
// To fully test everything:
//    go test -tags=x -benchtime=100ms -tv -bg -bi  -brw -bu -v -run=. -bench=.

import (
	"errors"
	"flag"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

const (
	testLogToT    = true
	failNowOnFail = true
)

var (
	testNoopH      = NoopHandle(8)
	testMsgpackH   = &MsgpackHandle{}
	testBincH      = &BincHandle{}
	testBincHNoSym = &BincHandle{}
	testBincHSym   = &BincHandle{}
	testSimpleH    = &SimpleHandle{}
	testCborH      = &CborHandle{}
	testJsonH      = &JsonHandle{}

	testPreInitFns  []func()
	testPostInitFns []func()

	testOnce sync.Once
)

func init() {
	testBincHSym.AsSymbols = AsSymbolAll
	testBincHNoSym.AsSymbols = AsSymbolNone
}

func testInitAll() {
	flag.Parse()
	for _, f := range testPreInitFns {
		f()
	}
	for _, f := range testPostInitFns {
		f()
	}
}

func logT(x interface{}, format string, args ...interface{}) {
	if t, ok := x.(*testing.T); ok && t != nil && testLogToT {
		if testVerbose {
			t.Logf(format, args...)
		}
	} else if b, ok := x.(*testing.B); ok && b != nil && testLogToT {
		b.Logf(format, args...)
	} else {
		if len(format) == 0 || format[len(format)-1] != '\n' {
			format = format + "\n"
		}
		fmt.Printf(format, args...)
	}
}

func approxDataSize(rv reflect.Value) (sum int) {
	switch rk := rv.Kind(); rk {
	case reflect.Invalid:
	case reflect.Ptr, reflect.Interface:
		sum += int(rv.Type().Size())
		sum += approxDataSize(rv.Elem())
	case reflect.Slice:
		sum += int(rv.Type().Size())
		for j := 0; j < rv.Len(); j++ {
			sum += approxDataSize(rv.Index(j))
		}
	case reflect.String:
		sum += int(rv.Type().Size())
		sum += rv.Len()
	case reflect.Map:
		sum += int(rv.Type().Size())
		for _, mk := range rv.MapKeys() {
			sum += approxDataSize(mk)
			sum += approxDataSize(rv.MapIndex(mk))
		}
	case reflect.Struct:
		//struct size already includes the full data size.
		//sum += int(rv.Type().Size())
		for j := 0; j < rv.NumField(); j++ {
			sum += approxDataSize(rv.Field(j))
		}
	default:
		//pure value types
		sum += int(rv.Type().Size())
	}
	return
}

// ----- functions below are used only by tests (not benchmarks)

func checkErrT(t *testing.T, err error) {
	if err != nil {
		logT(t, err.Error())
		failT(t)
	}
}

func checkEqualT(t *testing.T, v1 interface{}, v2 interface{}, desc string) (err error) {
	if err = deepEqual(v1, v2); err != nil {
		logT(t, "Not Equal: %s: %v. v1: %v, v2: %v", desc, err, v1, v2)
		failT(t)
	}
	return
}

func failT(t *testing.T) {
	if failNowOnFail {
		t.FailNow()
	} else {
		t.Fail()
	}
}

func deepEqual(v1, v2 interface{}) (err error) {
	if !reflect.DeepEqual(v1, v2) {
		err = errors.New("Not Match")
	}
	return
}
