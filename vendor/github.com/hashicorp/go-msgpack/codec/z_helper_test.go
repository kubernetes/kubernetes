// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// All non-std package dependencies related to testing live in this file,
// so porting to different environment is easy (just update functions).
//
// Also, this file is called z_helper_test, to give a "hint" to compiler
// that its init() function should be called last. (not guaranteed by spec)

import (
	"errors"
	"reflect"
	"flag"
	"testing"
)

var (
	testLogToT    = true
	failNowOnFail = true
)

func init() {
	testInitFlags()
	benchInitFlags()
	flag.Parse()
	testInit()
	benchInit()
}

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

func logT(x interface{}, format string, args ...interface{}) {
	if t, ok := x.(*testing.T); ok && t != nil && testLogToT {
		t.Logf(format, args...)
	} else if b, ok := x.(*testing.B); ok && b != nil && testLogToT {
		b.Logf(format, args...)
	} else {
		debugf(format, args...)
	}
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
