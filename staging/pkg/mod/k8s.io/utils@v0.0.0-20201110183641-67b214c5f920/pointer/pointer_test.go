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

func TestInt32Ptr(t *testing.T) {
	val := int32(0)
	ptr := Int32Ptr(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}

	val = int32(1)
	ptr = Int32Ptr(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}
}

func TestInt32PtrDerefOr(t *testing.T) {
	var val, def int32 = 1, 0

	out := Int32PtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %d, got %d", val, out)
	}

	out = Int32PtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %d, got %d", def, out)
	}
}

func TestInt64Ptr(t *testing.T) {
	val := int64(0)
	ptr := Int64Ptr(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}

	val = int64(1)
	ptr = Int64Ptr(val)
	if *ptr != val {
		t.Errorf("expected %d, got %d", val, *ptr)
	}
}

func TestInt64PtrDerefOr(t *testing.T) {
	var val, def int64 = 1, 0

	out := Int64PtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %d, got %d", val, out)
	}

	out = Int64PtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %d, got %d", def, out)
	}
}

func TestBoolPtr(t *testing.T) {
	val := false
	ptr := BoolPtr(val)
	if *ptr != val {
		t.Errorf("expected %t, got %t", val, *ptr)
	}

	val = true
	ptr = BoolPtr(true)
	if *ptr != val {
		t.Errorf("expected %t, got %t", val, *ptr)
	}
}

func TestBoolPtrDerefOr(t *testing.T) {
	val, def := true, false

	out := BoolPtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %t, got %t", val, out)
	}

	out = BoolPtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %t, got %t", def, out)
	}
}

func TestStringPtr(t *testing.T) {
	val := ""
	ptr := StringPtr(val)
	if *ptr != val {
		t.Errorf("expected %s, got %s", val, *ptr)
	}

	val = "a"
	ptr = StringPtr(val)
	if *ptr != val {
		t.Errorf("expected %s, got %s", val, *ptr)
	}
}

func TestStringPtrDerefOr(t *testing.T) {
	val, def := "a", ""

	out := StringPtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %s, got %s", val, out)
	}

	out = StringPtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %s, got %s", def, out)
	}
}

func TestFloat32Ptr(t *testing.T) {
	val := float32(0)
	ptr := Float32Ptr(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}

	val = float32(0.1)
	ptr = Float32Ptr(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}
}

func TestFloat32PtrDerefOr(t *testing.T) {
	var val, def float32 = 0.1, 0

	out := Float32PtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %f, got %f", val, out)
	}

	out = Float32PtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %f, got %f", def, out)
	}
}

func TestFloat64Ptr(t *testing.T) {
	val := float64(0)
	ptr := Float64Ptr(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}

	val = float64(0.1)
	ptr = Float64Ptr(val)
	if *ptr != val {
		t.Errorf("expected %f, got %f", val, *ptr)
	}
}

func TestFloat64PtrDerefOr(t *testing.T) {
	var val, def float64 = 0.1, 0

	out := Float64PtrDerefOr(&val, def)
	if out != val {
		t.Errorf("expected %f, got %f", val, out)
	}

	out = Float64PtrDerefOr(nil, def)
	if out != def {
		t.Errorf("expected %f, got %f", def, out)
	}
}
