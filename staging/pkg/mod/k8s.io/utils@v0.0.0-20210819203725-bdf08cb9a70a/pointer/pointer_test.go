/*
Copyright 2018 The Kubernetes Authors.

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

package pointer

import (
	"fmt"
	"testing"
)

func TestAllPtrFieldsNil(t *testing.T) {
	testCases := []struct {
		obj      interface{}
		expected bool
	}{
		{struct{}{}, true},
		{struct{ Foo int }{12345}, true},
		{&struct{ Foo int }{12345}, true},
		{struct{ Foo *int }{nil}, true},
		{&struct{ Foo *int }{nil}, true},
		{struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{&struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{&struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{struct{ Foo *int }{new(int)}, false},
		{&struct{ Foo *int }{new(int)}, false},
		{struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{&struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{(*struct{})(nil), true},
	}
	for i, tc := range testCases {
		name := fmt.Sprintf("case[%d]", i)
		t.Run(name, func(t *testing.T) {
			if actual := AllPtrFieldsNil(tc.obj); actual != tc.expected {
				t.Errorf("%s: expected %t, got %t", name, tc.expected, actual)
			}
		})
	}
}

func TestInt(t *testing.T) {
	val := int(0)
	ptr := Int(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}

	val = int(1)
	ptr = Int(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}
}

func TestIntDeref(t *testing.T) {
	var val, def int = 1, 0

	out := IntDeref(&val, def)
	if out != val {
		t.Errorf("expected %d, got %d", val, out)
	}

	out = IntDeref(nil, def)
	if out != def {
		t.Errorf("expected %d, got %d", def, out)
	}
}

func TestInt32(t *testing.T) {
	val := int32(0)
	ptr := Int32(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}

	val = int32(1)
	ptr = Int32(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}
}

func TestInt32Deref(t *testing.T) {
	var val, def int32 = 1, 0

	out := Int32Deref(&val, def)
	if out != val {
		t.Errorf("expected %d, got %d", val, out)
	}

	out = Int32Deref(nil, def)
	if out != def {
		t.Errorf("expected %d, got %d", def, out)
	}
}

func TestInt32Equal(t *testing.T) {
	if !Int32Equal(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !Int32Equal(Int32(123), Int32(123)) {
		t.Errorf("expected true (val == val)")
	}
	if Int32Equal(nil, Int32(123)) {
		t.Errorf("expected false (nil != val)")
	}
	if Int32Equal(Int32(123), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if Int32Equal(Int32(123), Int32(456)) {
		t.Errorf("expected false (val != val)")
	}
}

func TestInt64(t *testing.T) {
	val := int64(0)
	ptr := Int64(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}

	val = int64(1)
	ptr = Int64(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}
}

func TestInt64Deref(t *testing.T) {
	var val, def int64 = 1, 0

	out := Int64Deref(&val, def)
	if out != val {
		t.Errorf("expected %d, got %d", val, out)
	}

	out = Int64Deref(nil, def)
	if out != def {
		t.Errorf("expected %d, got %d", def, out)
	}
}

func TestInt64Equal(t *testing.T) {
	if !Int64Equal(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !Int64Equal(Int64(123), Int64(123)) {
		t.Errorf("expected true (val == val)")
	}
	if Int64Equal(nil, Int64(123)) {
		t.Errorf("expected false (nil != val)")
	}
	if Int64Equal(Int64(123), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if Int64Equal(Int64(123), Int64(456)) {
		t.Errorf("expected false (val != val)")
	}
}

func TestBool(t *testing.T) {
	val := false
	ptr := Bool(val)
	if *ptr != val {
		t.Errorf("expected %t, got %t", val, *ptr)
	}

	val = true
	ptr = Bool(true)
	if *ptr != val {
		t.Errorf("expected %t, got %t", val, *ptr)
	}
}

func TestBoolDeref(t *testing.T) {
	val, def := true, false

	out := BoolDeref(&val, def)
	if out != val {
		t.Errorf("expected %t, got %t", val, out)
	}

	out = BoolDeref(nil, def)
	if out != def {
		t.Errorf("expected %t, got %t", def, out)
	}
}

func TestBoolEqual(t *testing.T) {
	if !BoolEqual(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !BoolEqual(Bool(true), Bool(true)) {
		t.Errorf("expected true (val == val)")
	}
	if BoolEqual(nil, Bool(true)) {
		t.Errorf("expected false (nil != val)")
	}
	if BoolEqual(Bool(true), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if BoolEqual(Bool(true), Bool(false)) {
		t.Errorf("expected false (val != val)")
	}
}

func TestString(t *testing.T) {
	val := ""
	ptr := String(val)
	if *ptr != val {
		t.Errorf("expected %s, got %s", val, *ptr)
	}

	val = "a"
	ptr = String(val)
	if *ptr != val {
		t.Errorf("expected %s, got %s", val, *ptr)
	}
}

func TestStringDeref(t *testing.T) {
	val, def := "a", ""

	out := StringDeref(&val, def)
	if out != val {
		t.Errorf("expected %s, got %s", val, out)
	}

	out = StringDeref(nil, def)
	if out != def {
		t.Errorf("expected %s, got %s", def, out)
	}
}

func TestStringEqual(t *testing.T) {
	if !StringEqual(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !StringEqual(String("abc"), String("abc")) {
		t.Errorf("expected true (val == val)")
	}
	if StringEqual(nil, String("abc")) {
		t.Errorf("expected false (nil != val)")
	}
	if StringEqual(String("abc"), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if StringEqual(String("abc"), String("def")) {
		t.Errorf("expected false (val != val)")
	}
}

func TestFloat32(t *testing.T) {
	val := float32(0)
	ptr := Float32(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}

	val = float32(0.1)
	ptr = Float32(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}
}

func TestFloat32Deref(t *testing.T) {
	var val, def float32 = 0.1, 0

	out := Float32Deref(&val, def)
	if out != val {
		t.Errorf("expected %f, got %f", val, out)
	}

	out = Float32Deref(nil, def)
	if out != def {
		t.Errorf("expected %f, got %f", def, out)
	}
}

func TestFloat32Equal(t *testing.T) {
	if !Float32Equal(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !Float32Equal(Float32(1.25), Float32(1.25)) {
		t.Errorf("expected true (val == val)")
	}
	if Float32Equal(nil, Float32(1.25)) {
		t.Errorf("expected false (nil != val)")
	}
	if Float32Equal(Float32(1.25), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if Float32Equal(Float32(1.25), Float32(4.5)) {
		t.Errorf("expected false (val != val)")
	}
}

func TestFloat64(t *testing.T) {
	val := float64(0)
	ptr := Float64(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}

	val = float64(0.1)
	ptr = Float64(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}
}

func TestFloat64Deref(t *testing.T) {
	var val, def float64 = 0.1, 0

	out := Float64Deref(&val, def)
	if out != val {
		t.Errorf("expected %f, got %f", val, out)
	}

	out = Float64Deref(nil, def)
	if out != def {
		t.Errorf("expected %f, got %f", def, out)
	}
}

func TestFloat64Equal(t *testing.T) {
	if !Float64Equal(nil, nil) {
		t.Errorf("expected true (nil == nil)")
	}
	if !Float64Equal(Float64(1.25), Float64(1.25)) {
		t.Errorf("expected true (val == val)")
	}
	if Float64Equal(nil, Float64(1.25)) {
		t.Errorf("expected false (nil != val)")
	}
	if Float64Equal(Float64(1.25), nil) {
		t.Errorf("expected false (val != nil)")
	}
	if Float64Equal(Float64(1.25), Float64(4.5)) {
		t.Errorf("expected false (val != val)")
	}
}
