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

type TestStructWithKey struct {
	Key string
	I   int
	D   string
}

type NonComparableKey struct {
	I *int
}

type NonComparableStruct struct {
	I int
	S []string
}

type NonComparableStructWithKey struct {
	Key string
	I   int
	S   []string
}

type NonComparableStructWithPtr struct {
	I int
	P *int
}

func TestEachValSliceVal(t *testing.T) {
	testEachValSliceVal(t, "valid", []int{11, 12, 13})
	testEachValSliceVal(t, "valid", []string{"a", "b", "c"})
	testEachValSliceVal(t, "valid", []TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachValSliceVal(t, "empty", []int{})
	testEachValSliceVal(t, "empty", []string{})
	testEachValSliceVal(t, "empty", []TestStruct{})

	testEachValSliceVal[int](t, "nil", nil)
	testEachValSliceVal[string](t, "nil", nil)
	testEachValSliceVal[TestStruct](t, "nil", nil)

	testEachValSliceValUpdate(t, "valid", []int{11, 12, 13})
	testEachValSliceValUpdate(t, "valid", []string{"a", "b", "c"})
	testEachValSliceValUpdate(t, "valid", []TestStruct{{11, "a"}, {12, "b"}, {13, "c"}})

	testEachValSliceValUpdate(t, "empty", []int{})
	testEachValSliceValUpdate(t, "empty", []string{})
	testEachValSliceValUpdate(t, "empty", []TestStruct{})

	testEachValSliceValUpdate[int](t, "nil", nil)
	testEachValSliceValUpdate[string](t, "nil", nil)
	testEachValSliceValUpdate[TestStruct](t, "nil", nil)
}

func testEachValSliceVal[T any](t *testing.T, name string, input []T) {
	t.Helper()
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
		_ = EachValSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func testEachValSliceValUpdate[T any](t *testing.T, name string, input []T) {
	t.Helper()
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
		match := func(a, b *T) bool { return reflect.DeepEqual(*a, *b) }
		_ = EachValSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, old, match, match, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachValSliceValRatcheting(t *testing.T) {
	testEachValSliceValRatcheting(t, "ComparableStruct same data different order",
		[]TestStruct{
			{11, "a"}, {12, "b"}, {13, "c"},
		},
		[]TestStruct{
			{11, "a"}, {13, "c"}, {12, "b"},
		},
		SemanticDeepEqual,
		nil,
	)
	testEachValSliceValRatcheting(t, "ComparableStruct less data in new, exist in old",
		[]TestStruct{
			{11, "a"}, {12, "b"}, {13, "c"},
		},
		[]TestStruct{
			{11, "a"}, {13, "c"},
		},
		DirectEqual,
		nil,
	)
	testEachValSliceValRatcheting(t, "Comparable struct with key same data different order",
		[]TestStructWithKey{
			{Key: "a", I: 11, D: "a"}, {Key: "b", I: 12, D: "b"}, {Key: "c", I: 13, D: "c"},
		},
		[]TestStructWithKey{
			{Key: "a", I: 11, D: "a"}, {Key: "c", I: 13, D: "c"}, {Key: "b", I: 12, D: "b"},
		},
		MatchFunc[*TestStructWithKey](func(a, b *TestStructWithKey) bool {
			return a.Key == b.Key
		}),
		DirectEqual,
	)
	testEachValSliceValRatcheting(t, "Comparable struct with key less data in new, exist in old",
		[]TestStructWithKey{
			{Key: "a", I: 11, D: "a"}, {Key: "b", I: 12, D: "b"}, {Key: "c", I: 13, D: "c"},
		},
		[]TestStructWithKey{
			{Key: "a", I: 11, D: "a"}, {Key: "c", I: 13, D: "c"},
		},
		MatchFunc[*TestStructWithKey](func(a, b *TestStructWithKey) bool {
			return a.Key == b.Key
		}),
		DirectEqual,
	)
	testEachValSliceValRatcheting(t, "NonComparableStruct same data different order",
		[]NonComparableStruct{
			{I: 11, S: []string{"a"}}, {I: 12, S: []string{"b"}}, {I: 13, S: []string{"c"}},
		},
		[]NonComparableStruct{
			{I: 11, S: []string{"a"}}, {I: 13, S: []string{"c"}}, {I: 12, S: []string{"b"}},
		},
		SemanticDeepEqual,
		nil,
	)
	testEachValSliceValRatcheting(t, "NonComparableStructWithKey same data different order",
		[]NonComparableStructWithKey{
			{Key: "a", I: 11, S: []string{"a"}}, {Key: "b", I: 12, S: []string{"b"}}, {Key: "c", I: 13, S: []string{"c"}},
		},
		[]NonComparableStructWithKey{
			{Key: "a", I: 11, S: []string{"a"}}, {Key: "b", I: 12, S: []string{"b"}}, {Key: "c", I: 13, S: []string{"c"}},
		},
		MatchFunc[*NonComparableStructWithKey](func(a, b *NonComparableStructWithKey) bool {
			return a.Key == b.Key
		}),
		SemanticDeepEqual,
	)

}

func testEachValSliceValRatcheting[T any](t *testing.T, name string, old, new []T, match, equiv MatchFunc[*T]) {
	t.Helper()
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			return field.ErrorList{field.Invalid(fldPath, *newVal, "expected no calls")}
		}
		errs := EachValSliceVal(context.Background(), operation.Operation{Type: operation.Update}, field.NewPath("test"), new, old, match, equiv, vfn)
		if len(errs) > 0 {
			t.Errorf("expected no errors, got %d: %s", len(errs), fmtErrs(errs))
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
	t.Helper()
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
		_ = EachMapVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, vfn)
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachMapValRatcheting(t *testing.T) {
	testEachMapValRatcheting(t, "primitive same data",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13, "two": 12},
		DirectEqual,
		0,
	)
	testEachMapValRatcheting(t, "primitive less data in new, exist in old",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13},
		DirectEqual,
		0,
	)
	testEachMapValRatcheting(t, "primitive new data, not exist in old",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13, "two": 12, "four": 14},
		DirectEqual,
		1,
	)
	testEachMapValRatcheting(t, "non comparable value, same data",
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"two":   {I: 12, S: []string{"b"}},
			"three": {I: 13, S: []string{"c"}},
		},
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"three": {I: 13, S: []string{"c"}},
			"two":   {I: 12, S: []string{"b"}},
		},
		SemanticDeepEqual,
		0,
	)
	testEachMapValRatcheting(t, "non comparable value, less data in new, exist in old",
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"two":   {I: 12, S: []string{"b"}},
			"three": {I: 13, S: []string{"c"}},
		},
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"three": {I: 13, S: []string{"c"}},
		},
		SemanticDeepEqual,
		0,
	)
	testEachMapValRatcheting(t, "non comparable value, new data, not exist in old",
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"two":   {I: 12, S: []string{"b"}},
			"three": {I: 13, S: []string{"c"}},
		},
		map[string]NonComparableStruct{
			"one":   {I: 11, S: []string{"a"}},
			"three": {I: 13, S: []string{"c"}},
			"two":   {I: 12, S: []string{"b"}},
			"four":  {I: 14, S: []string{"d"}},
		},
		SemanticDeepEqual,
		1,
	)
	testEachMapValRatcheting(t, "struct with pointer field, same value different pointer",
		map[string]NonComparableStructWithPtr{
			"one": {I: 11, P: ptr.To(1)},
			"two": {I: 12, P: ptr.To(2)},
		},
		map[string]NonComparableStructWithPtr{
			"one": {I: 11, P: ptr.To(1)},
			"two": {I: 12, P: ptr.To(2)},
		},
		SemanticDeepEqual,
		0,
	)
	testEachMapValRatcheting(t, "nil map to empty map",
		nil,
		map[string]int{},
		DirectEqual,
		0,
	)

	testEachMapValRatcheting(t, "nil map to non-empty map",
		nil,
		map[string]int{"one": 1},
		DirectEqual,
		1, // Expect validation for new entry
	)

	testEachMapValRatcheting(t, "empty map to nil map",
		map[string]int{},
		nil,
		DirectEqual,
		0,
	)

	testEachMapValRatcheting(t, "non-empty map to nil map",
		map[string]int{"one": 1},
		nil,
		DirectEqual,
		0,
	)
}

func testEachMapValRatcheting[K ~string, V any](t *testing.T, name string, old, new map[K]V, equiv MatchFunc[*V], wantCalls int) {
	t.Helper()
	var zero V
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *V) field.ErrorList {
			calls++
			return nil
		}
		_ = EachMapVal(context.Background(), operation.Operation{Type: operation.Update}, field.NewPath("test"), new, old, equiv, vfn)
		if calls != wantCalls {
			t.Errorf("expected %d calls, got %d", wantCalls, calls)
		}
	})
}

type StringType string

func TestEachMapKey(t *testing.T) {
	testEachMapKey(t, "valid", map[string]int{"one": 11, "two": 12, "three": 13})
	testEachMapKey(t, "valid", map[StringType]string{"A": "a", "B": "b", "C": "c"})
}

func testEachMapKey[K ~string, V any](t *testing.T, name string, input map[K]V) {
	t.Helper()
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

func TestEachMapKeyRatcheting(t *testing.T) {
	testEachMapKeyRatcheting(t, "same data, 0 validation calls",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13, "two": 12},
		0,
	)
	testEachMapKeyRatcheting(t, "less data in new, exist in old, 0 validation calls",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13},
		0,
	)
	testEachMapKeyRatcheting(t, "new data, not exist in old, 1 validation call",
		map[string]int{"one": 11, "two": 12, "three": 13},
		map[string]int{"one": 11, "three": 13, "two": 12, "four": 14},
		1,
	)
}

func testEachMapKeyRatcheting[K ~string, V any](t *testing.T, name string, old, new map[K]V, wantCalls int) {
	t.Helper()
	var zero V
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *K) field.ErrorList {
			calls++
			return nil
		}
		_ = EachMapKey(context.Background(), operation.Operation{Type: operation.Update}, field.NewPath("test"), new, old, vfn)
		if calls != wantCalls {
			t.Errorf("expected %d calls, got %d", wantCalls, calls)
		}
	})
}

func TestValSliceUniqueComparableValues(t *testing.T) {
	testValSliceUnique(t, "int_nil", []int(nil), 0)
	testValSliceUnique(t, "int_empty", []int{}, 0)
	testValSliceUnique(t, "int_uniq", []int{1, 2, 3}, 0)
	testValSliceUnique(t, "int_dup", []int{1, 2, 3, 2, 1}, 2)

	testValSliceUnique(t, "string_nil", []string(nil), 0)
	testValSliceUnique(t, "string_empty", []string{}, 0)
	testValSliceUnique(t, "string_uniq", []string{"a", "b", "c"}, 0)
	testValSliceUnique(t, "string_dup", []string{"a", "a", "c", "b", "a"}, 2)

	type isComparable struct {
		I int
		S string
	}

	testValSliceUnique(t, "struct_nil", []isComparable(nil), 0)
	testValSliceUnique(t, "struct_empty", []isComparable{}, 0)
	testValSliceUnique(t, "struct_uniq", []isComparable{{1, "a"}, {2, "b"}, {3, "c"}}, 0)
	testValSliceUnique(t, "struct_dup", []isComparable{{1, "a"}, {2, "b"}, {3, "c"}, {2, "b"}, {1, "a"}}, 2)
}

func testValSliceUnique[T comparable](t *testing.T, name string, input []T, wantErrs int) {
	t.Helper()
	t.Run(fmt.Sprintf("%s(direct)", name), func(t *testing.T) {
		errs := ValSliceUnique(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, DirectEqual)
		if len(errs) != wantErrs {
			t.Errorf("expected %d errors, got %d: %s", wantErrs, len(errs), fmtErrs(errs))
		}
	})
	t.Run(fmt.Sprintf("%s(reflect)", name), func(t *testing.T) {
		errs := ValSliceUnique(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, SemanticDeepEqual)
		if len(errs) != wantErrs {
			t.Errorf("expected %d errors, got %d: %s", wantErrs, len(errs), fmtErrs(errs))
		}
	})
}

func TestValSliceUniqueNonComparableValues(t *testing.T) {
	type nonComparable struct {
		I int
		S []string
	}

	testValSliceUniqueByReflect(t, "noncomp_nil", []nonComparable(nil), 0)
	testValSliceUniqueByReflect(t, "noncomp_empty", []nonComparable{}, 0)
	testValSliceUniqueByReflect(t, "noncomp_uniq", []nonComparable{{1, []string{"a"}}, {2, []string{"b"}}, {3, []string{"c"}}}, 0)
	testValSliceUniqueByReflect(t, "noncomp_dup", []nonComparable{
		{1, []string{"a"}},
		{2, []string{"b"}},
		{3, []string{"c"}},
		{2, []string{"b"}},
		{1, []string{"a"}}}, 2)
}

func testValSliceUniqueByReflect[T any](t *testing.T, name string, input []T, wantErrs int) {
	t.Helper()
	var zero T
	t.Run(fmt.Sprintf("%s(%T)", name, zero), func(t *testing.T) {
		errs := ValSliceUnique(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, SemanticDeepEqual)
		if len(errs) != wantErrs {
			t.Errorf("expected %d errors, got %d: %s", wantErrs, len(errs), fmtErrs(errs))
		}
	})
}

func TestEachPtrSliceVal(t *testing.T) {
	testEachPtrSliceVal(t, "valid", []*int{ptr.To(11), ptr.To(12), ptr.To(13)})
	testEachPtrSliceVal(t, "valid", []*string{ptr.To("a"), ptr.To("b"), ptr.To("c")})
	testEachPtrSliceVal(t, "valid", []*TestStruct{ptr.To(TestStruct{11, "a"}), ptr.To(TestStruct{12, "b"}), ptr.To(TestStruct{13, "c"})})

	testEachPtrSliceVal(t, "empty", []*int{})
	testEachPtrSliceVal[int](t, "nil", nil)

	// Test nil elements
	t.Run("nil elements", func(t *testing.T) {
		input := []*int{ptr.To(11), nil, ptr.To(13)}
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *int) field.ErrorList {
			calls++
			return nil
		}
		errs := EachPtrSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, nil, vfn)
		if len(errs) != 0 {
			t.Errorf("expected 0 errors, got %d", len(errs))
		}
		if calls != 2 {
			t.Errorf("expected 2 calls, got %d", calls)
		}
	})

	testEachPtrSliceValUpdate(t, "valid", []*int{ptr.To(11), ptr.To(12), ptr.To(13)})
}

func testEachPtrSliceVal[T any](t *testing.T, name string, input []*T) {
	t.Helper()
	var zero T
	t.Run(fmt.Sprintf("%s(*%T)", name, zero), func(t *testing.T) {
		calls := 0
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			if oldVal != nil {
				t.Errorf("expected nil oldVal, got %v", *oldVal)
			}
			calls++
			return nil
		}
		errs := EachPtrSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, nil, nil, vfn)
		if len(errs) != 0 {
			t.Errorf("expected 0 errors, got %d: %v", len(errs), errs)
		}
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func testEachPtrSliceValUpdate[T any](t *testing.T, name string, input []*T) {
	t.Helper()
	var zero T
	t.Run(fmt.Sprintf("%s(*%T) update", name, zero), func(t *testing.T) {
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
		old := make([]*T, len(input))
		copy(old, input)
		slices.Reverse(old)
		match := func(a, b *T) bool { return reflect.DeepEqual(*a, *b) }
		errs := EachPtrSliceVal(context.Background(), operation.Operation{}, field.NewPath("test"), input, old, match, match, vfn)
		if len(errs) != 0 {
			t.Errorf("expected 0 errors, got %d: %v", len(errs), errs)
		}
		if calls != len(input) {
			t.Errorf("expected %d calls, got %d", len(input), calls)
		}
	})
}

func TestEachPtrSliceValRatcheting(t *testing.T) {
	testEachPtrSliceValRatcheting(t, "ComparableStruct same data different order",
		[]*TestStruct{
			ptr.To(TestStruct{11, "a"}), ptr.To(TestStruct{12, "b"}), ptr.To(TestStruct{13, "c"}),
		},
		[]*TestStruct{
			ptr.To(TestStruct{11, "a"}), ptr.To(TestStruct{13, "c"}), ptr.To(TestStruct{12, "b"}),
		},
		SemanticDeepEqual,
		nil,
	)
}

func testEachPtrSliceValRatcheting[T any](t *testing.T, name string, old, new []*T, match, equiv MatchFunc[*T]) {
	t.Helper()
	var zero T
	t.Run(fmt.Sprintf("%s(*%T)", name, zero), func(t *testing.T) {
		vfn := func(ctx context.Context, op operation.Operation, fldPath *field.Path, newVal, oldVal *T) field.ErrorList {
			return field.ErrorList{field.Invalid(fldPath, *newVal, "expected no calls")}
		}
		errs := EachPtrSliceVal(context.Background(), operation.Operation{Type: operation.Update}, field.NewPath("test"), new, old, match, equiv, vfn)
		if len(errs) != 0 {
			t.Errorf("expected 0 errors, got %d: %v", len(errs), errs)
		}
	})
}

func TestPtrSliceUnique(t *testing.T) {
	testPtrSliceUnique(t, "int_nil", []*int(nil), 0, 0)
	testPtrSliceUnique(t, "int_empty", []*int{}, 0, 0)
	testPtrSliceUnique(t, "int_uniq", []*int{ptr.To(1), ptr.To(2), ptr.To(3)}, 0, 0)
	testPtrSliceUnique(t, "int_dup", []*int{ptr.To(1), ptr.To(2), ptr.To(3), ptr.To(2), ptr.To(1)}, 2, 0)
	testPtrSliceUnique(t, "int_nil_element", []*int{ptr.To(1), nil, ptr.To(3), nil}, 0, 0)
	testPtrSliceUnique(t, "int_dup_and_nil", []*int{ptr.To(1), nil, ptr.To(1), nil}, 1, 0)
}

func testPtrSliceUnique[T comparable](t *testing.T, name string, input []*T, wantDupErrs, wantReqErrs int) {
	t.Helper()
	t.Run(fmt.Sprintf("%s(direct)", name), func(t *testing.T) {
		errs := PtrSliceUnique(context.Background(), operation.Operation{}, field.NewPath("test"), input, nil, DirectEqual)
		gotDup, gotReq := countErrors(errs)
		if gotDup != wantDupErrs || gotReq != wantReqErrs {
			t.Errorf("expected %d dup, %d req errors; got %d dup, %d req: %s", wantDupErrs, wantReqErrs, gotDup, gotReq, fmtErrs(errs))
		}
	})
}

func countErrors(errs field.ErrorList) (dup, req int) {
	for _, err := range errs {
		switch err.Type {
		case field.ErrorTypeDuplicate:
			dup++
		case field.ErrorTypeRequired:
			req++
		}
	}
	return
}

func TestPtrSliceNoNils(t *testing.T) {
	tests := []struct {
		name     string
		input    []*int
		wantErrs int
	}{
		{"nil", nil, 0},
		{"empty", []*int{}, 0},
		{"no_nil", []*int{ptr.To(1), ptr.To(2)}, 0},
		{"has_nil", []*int{ptr.To(1), nil, ptr.To(3), nil}, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := PtrSliceNoNils(context.Background(), operation.Operation{}, field.NewPath("test"), tt.input, nil)
			if len(errs) != tt.wantErrs {
				t.Errorf("expected %d errors, got %d: %v", tt.wantErrs, len(errs), errs)
			}
			for _, err := range errs {
				if err.Type != field.ErrorTypeRequired {
					t.Errorf("expected Required error, got %v", err.Type)
				}
			}
		})
	}
}
