// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mapsort_test

import (
	"strconv"
	"testing"

	"google.golang.org/protobuf/internal/mapsort"
	pref "google.golang.org/protobuf/reflect/protoreflect"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

func TestRange(t *testing.T) {
	m := (&testpb.TestAllTypes{
		MapBoolBool: map[bool]bool{
			false: false,
			true:  true,
		},
		MapInt32Int32: map[int32]int32{
			0: 0,
			1: 1,
			2: 2,
		},
		MapUint64Uint64: map[uint64]uint64{
			0: 0,
			1: 1,
			2: 2,
		},
		MapStringString: map[string]string{
			"0": "0",
			"1": "1",
			"2": "2",
		},
	}).ProtoReflect()
	m.Range(func(fd pref.FieldDescriptor, v pref.Value) bool {
		mapv := v.Map()
		var got []pref.MapKey
		mapsort.Range(mapv, fd.MapKey().Kind(), func(key pref.MapKey, _ pref.Value) bool {
			got = append(got, key)
			return true
		})
		for wanti, key := range got {
			var goti int
			switch x := mapv.Get(key).Interface().(type) {
			case bool:
				if x {
					goti = 1
				}
			case int32:
				goti = int(x)
			case uint64:
				goti = int(x)
			case string:
				goti, _ = strconv.Atoi(x)
			default:
				t.Fatalf("unhandled map value type %T", x)
			}
			if wanti != goti {
				t.Errorf("out of order range over map field %v: %v", fd.FullName(), got)
				break
			}
		}
		return true
	})
}
