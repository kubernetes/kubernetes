/*
Copyright 2025 The Kubernetes Authors.

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

package validate

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

type TestStruct struct {
	I int
	D string
}

func TestEachSliceVal(t *testing.T) {
	testEachSliceVal(t, "valid", []int{11, 12, 13})
	testEachSliceVal(t, "valid", []string{"a", "b", "c"})
	testEachSliceVal(t, "valid", []TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachSliceVal(t, "empty", []int{})
	testEachSliceVal(t, "empty", []string{})
	testEachSliceVal(t, "empty", []TestStruct{})

	testEachSliceVal[int](t, "nil", nil)
	testEachSliceVal[string](t, "nil", nil)
	testEachSliceVal[TestStruct](t, "nil", nil)

	testEachSliceValUpdate(t, "valid", []int{11, 12, 13})
	testEachSliceValUpdate(t, "valid", []string{"a", "b", "c"})
	testEachSliceValUpdate(t, "valid", []TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachSliceValUpdate(t, "empty", []int{})
	testEachSliceValUpdate(t, "empty", []string{})
	testEachSliceValUpdate(t, "empty", []TestStruct{})

	testEachSliceValUpdate[int](t, "nil", nil)
	testEachSliceValUpdate[string](t, "nil", nil)
	testEachSliceValUpdate[TestStruct](t, "nil", nil)
}

func testEachSliceVal[T any](t *testing.T, name string, input []T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			if oldVal != nil {
				t.Errorf("expected nil oldVal, got %v", *oldVal)
			}
			calls++
			return nil
		}
		_ = EachSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func testEachSliceValUpdate[T any](t *testing.T, name string, input []T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			if oldVal == nil {
				t.Fatalf("expected non-nil oldVal")
			}
			if !reflect.DeepEqual(*newVal, *oldVal) {
				t.Errorf("expected oldVal == newVal, got %v, %v", *oldVal, *newVal)
			}
			calls++
			return nil
		}
		old := make([]T, len(input))
		copy(old, input)
		slices.Reverse(old)
		cmp := func(a, b T) bool { return reflect.DeepEqual(a, b) }
		_ = EachSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, old, cmp, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachSliceValNilablePointer(t *testing.T) {
	testEachSliceValNilable(t, "valid", []*int{ptr.To(11), ptr.To(12), ptr.To(13)})
	testEachSliceValNilable(t, "valid", []*string{ptr.To("a"), ptr.To("b"), ptr.To("c")})
	testEachSliceValNilable(t, "valid", []*TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachSliceValNilable(t, "empty", []*int{})
	testEachSliceValNilable(t, "empty", []*string{})
	testEachSliceValNilable(t, "empty", []*TestStruct{})

	testEachSliceValNilable[int](t, "nil", nil)
	testEachSliceValNilable[string](t, "nil", nil)
	testEachSliceValNilable[TestStruct](t, "nil", nil)

	testEachSliceValNilableUpdate(t, "valid", []*int{ptr.To(11), ptr.To(12), ptr.To(13)})
	testEachSliceValNilableUpdate(t, "valid", []*string{ptr.To("a"), ptr.To("b"), ptr.To("c")})
	testEachSliceValNilableUpdate(t, "valid", []*TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachSliceValNilableUpdate(t, "empty", []*int{})
	testEachSliceValNilableUpdate(t, "empty", []*string{})
	testEachSliceValNilableUpdate(t, "empty", []*TestStruct{})

	testEachSliceValNilableUpdate[int](t, "nil", nil)
	testEachSliceValNilableUpdate[string](t, "nil", nil)
	testEachSliceValNilableUpdate[TestStruct](t, "nil", nil)
}

func TestEachSliceValNilableSlice(t *testing.T) {
	testEachSliceValNilable(t, "valid", [][]int{{11, 12, 13}, {12, 13, 14}, {13, 14, 15}})
	testEachSliceValNilable(t, "valid", [][]string{{"a", "b", "c"}, {"b", "c", "d"}, {"c", "d", "e"}})
	testEachSliceValNilable(t, "valid", [][]TestStruct{
		{{11, "a"}, {12, "b"}, {13, "c"}},
		{{12, "a"}, {13, "b"}, {14, "c"}},
		{{13, "a"}, {14, "b"}, {15, "c"}}})

	testEachSliceValNilable(t, "empty", [][]int{{}, {}, {}})
	testEachSliceValNilable(t, "empty", [][]string{{}, {}, {}})
	testEachSliceValNilable(t, "empty", [][]TestStruct{{}, {}, {}})

	testEachSliceValNilable[int](t, "nil", nil)
	testEachSliceValNilable[string](t, "nil", nil)
	testEachSliceValNilable[TestStruct](t, "nil", nil)

	testEachSliceValNilableUpdate(t, "valid", [][]int{{11, 12, 13}, {12, 13, 14}, {13, 14, 15}})
	testEachSliceValNilableUpdate(t, "valid", [][]string{{"a", "b", "c"}, {"b", "c", "d"}, {"c", "d", "e"}})
	testEachSliceValNilableUpdate(t, "valid", [][]TestStruct{
		{{11, "a"}, {12, "b"}, {13, "c"}},
		{{12, "a"}, {13, "b"}, {14, "c"}},
		{{13, "a"}, {14, "b"}, {15, "c"}}})

	testEachSliceValNilableUpdate(t, "empty", [][]int{{}, {}, {}})
	testEachSliceValNilableUpdate(t, "empty", [][]string{{}, {}, {}})
	testEachSliceValNilableUpdate(t, "empty", [][]TestStruct{{}, {}, {}})

	testEachSliceValNilableUpdate[int](t, "nil", nil)
	testEachSliceValNilableUpdate[string](t, "nil", nil)
	testEachSliceValNilableUpdate[TestStruct](t, "nil", nil)
}

func testEachSliceValNilable[T any](t *testing.T, name string, input []T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal T) field.ErrorList {
			if !reflect.DeepEqual(oldVal, zero) {
				t.Errorf("expected nil oldVal, got %v", oldVal)
			}
			calls++
			return nil
		}
		_ = EachSliceValNilable(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func testEachSliceValNilableUpdate[T any](t *testing.T, name string, input []T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal T) field.ErrorList {
			if reflect.DeepEqual(oldVal, zero) {
				t.Fatalf("expected non-nil oldVal")
			}
			if !reflect.DeepEqual(newVal, oldVal) {
				t.Errorf("expected oldVal == newVal, got %v, %v", oldVal, newVal)
			}
			calls++
			return nil
		}
		old := make([]T, len(input))
		copy(old, input)
		slices.Reverse(old)
		cmp := func(a, b T) bool { return reflect.DeepEqual(a, b) }
		_ = EachSliceValNilable(context.Background(), operation.Operation{}, field.NewPath("test"), input, old, cmp, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachMapVal(t *testing.T) {
	testEachMapVal(t, "valid", map[string]int{"one": 11, "two": 12, "three": 13})
	testEachMapVal(t, "valid", map[string]string{"A": "a", "B": "b", "C": "c"})
	testEachMapVal(t, "valid", map[string]TestStruct{"one": {11, "a"}, "two": {12, "b"}, "three": {13, "c"}})

	testEachMapVal(t, "empty", map[string]int{})
	testEachMapVal(t, "empty", map[string]string{})
	testEachMapVal(t, "empty", map[string]TestStruct{})

	testEachMapVal[int](t, "nil", nil)
	testEachMapVal[string](t, "nil", nil)
	testEachMapVal[TestStruct](t, "nil", nil)
}

func testEachMapVal[T any](t *testing.T, name string, input map[string]T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			if oldVal != nil {
				t.Errorf("expected nil oldVal, got %v", *oldVal)
			}
			calls++
			return nil
		}
		_ = EachMapVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachMapValNilablePointer(t *testing.T) {
	testEachMapValNilable(t, "valid", map[string]*int{"one": ptr.To(11), "two": ptr.To(12), "three": ptr.To(13)})
	testEachMapValNilable(t, "valid", map[string]*string{"A": ptr.To("a"), "B": ptr.To("b"), "C": ptr.To("c")})
	testEachMapValNilable(t, "valid", map[string]*TestStruct{"one": {11, "a"}, "two": {12, "b"}, "three": {13, "c"}})

	testEachMapValNilable(t, "empty", map[string]*int{})
	testEachMapValNilable(t, "empty", map[string]*string{})
	testEachMapValNilable(t, "empty", map[string]*TestStruct{})

	testEachMapValNilable[int](t, "nil", nil)
	testEachMapValNilable[string](t, "nil", nil)
	testEachMapValNilable[TestStruct](t, "nil", nil)
}

func TestEachMapValNilableSlice(t *testing.T) {
	testEachMapValNilable(t, "valid", map[string][]int{
		"one":   {11, 12, 13},
		"two":   {12, 13, 14},
		"three": {13, 14, 15}})
	testEachMapValNilable(t, "valid", map[string][]string{
		"A": {"a", "b", "c"},
		"B": {"b", "c", "d"},
		"C": {"c", "d", "e"}})
	testEachMapValNilable(t, "valid", map[string][]TestStruct{
		"one":   {{11, "a"}, {12, "b"}, {13, "c"}},
		"two":   {{12, "a"}, {13, "b"}, {14, "c"}},
		"three": {{13, "a"}, {14, "b"}, {15, "c"}}})

	testEachMapValNilable(t, "empty", map[string][]int{"a": {}, "b": {}, "c": {}})
	testEachMapValNilable(t, "empty", map[string][]string{"a": {}, "b": {}, "c": {}})
	testEachMapValNilable(t, "empty", map[string][]TestStruct{"a": {}, "b": {}, "c": {}})

	testEachMapValNilable[int](t, "nil", nil)
	testEachMapValNilable[string](t, "nil", nil)
	testEachMapValNilable[TestStruct](t, "nil", nil)
}

func testEachMapValNilable[T any](t *testing.T, name string, input map[string]T) {
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal T) field.ErrorList {
			if !reflect.DeepEqual(oldVal, zero) {
				t.Errorf("expected nil oldVal, got %v", oldVal)
			}
			calls++
			return nil
		}
		_ = EachMapValNilable(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

type StringType string

func TestEachMapKey(t *testing.T) {
	testEachMapKey(t, "valid", map[string]int{"one": 11, "two": 12, "three": 13})
	testEachMapKey(t, "valid", map[StringType]string{"A": "a", "B": "b", "C": "c"})
}

func testEachMapKey[K ~string, V any](t *testing.T, name string, input map[K]V) {
	var zero K
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *K) field.ErrorList {
			if oldVal != nil {
				t.Errorf("expected nil oldVal, got %v", *oldVal)
			}
			calls++
			return nil
		}
		_ = EachMapKey(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}
