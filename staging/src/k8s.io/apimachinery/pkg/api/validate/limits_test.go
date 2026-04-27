/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/constraints"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMaxLength(t *testing.T) {
	cases := []struct {
		name     string
		value    string
		max      int
		wantErrs field.ErrorList
	}{{
		name:     "empty string",
		value:    "",
		max:      0,
		wantErrs: nil,
	}, {
		name:  "zero length",
		value: "0",
		max:   0,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", 0).WithOrigin("maxLength"),
		},
	}, {
		name:     "one character",
		value:    "0",
		max:      1,
		wantErrs: nil,
	}, {
		name:  "two characters",
		value: "01",
		max:   1,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", 1).WithOrigin("maxLength"),
		},
	}, {
		value: "",
		max:   -1,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", -1).WithOrigin("maxLength"),
		},
	}, {
		name:     "ascii-only characters, less characters than max (n-1)",
		value:    "abcdefghi",
		max:      10,
		wantErrs: nil,
	}, {
		name:     "multi-byte characters, less characters than max (n-1)",
		value:    "©®©®©®©®©",
		max:      10,
		wantErrs: nil,
	}, {
		name:  "ascii-only characters, more characters than max (n+1)",
		value: "abcdefghijkl",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", 10).WithOrigin("maxLength"),
		},
	}, {
		name:  "multi-byte characters, more characters than max (n+1)",
		value: "©®©®©®©®©®©",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", 10).WithOrigin("maxLength"),
		},
	}, {
		name:  "mixture of characters, minimum possible size of input is less than max, rune count exceed maximum",
		value: "©abc®defghi",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLongCharacters(field.NewPath("fldpath"), "", 10).WithOrigin("maxLength"),
		},
	}, {
		name:     "multi-byte characters, exact characters as max (n)",
		value:    "©®©®©®©®©®",
		max:      10,
		wantErrs: nil,
	}, {
		name:     "ascii-only characters, exact characters as max (n)",
		value:    "abcdefghij",
		max:      10,
		wantErrs: nil,
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := tc.value
			gotErrs := MaxLength(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &v, nil, tc.max)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMinItems(t *testing.T) {
	cases := []struct {
		name     string
		items    int
		min      int
		wantErrs field.ErrorList
	}{{
		name:  "0 items, min 0",
		items: 0,
		min:   0,
	}, {
		name:  "1 item, min 0",
		items: 1,
		min:   0,
	}, {
		name:  "1 item, min 1",
		items: 1,
		min:   1,
	}, {
		name:  "0 items, min 1",
		items: 0,
		min:   1,
		wantErrs: field.ErrorList{
			field.TooFew(field.NewPath("fldpath"), 0, 1).WithOrigin("minItems"),
		},
	}, {
		name:  "1 item, min 2",
		items: 1,
		min:   2,
		wantErrs: field.ErrorList{
			field.TooFew(field.NewPath("fldpath"), 1, 2).WithOrigin("minItems"),
		},
	}, {
		name:  "0 items, min -1",
		items: 0,
		min:   -1,
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := make([]bool, tc.items)
			gotErrs := MinItems(context.Background(), operation.Operation{}, field.NewPath("fldpath"), value, nil, tc.min)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMaxItems(t *testing.T) {
	cases := []struct {
		name     string
		items    int
		max      int
		wantErrs field.ErrorList
	}{{
		name:  "0 items, max 0",
		items: 0,
		max:   0,
	}, {
		name:  "1 item, max 0",
		items: 1,
		max:   0,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 1, 0).WithOrigin("maxItems"),
		},
	}, {
		name:  "1 item, max 1",
		items: 1,
		max:   1,
	}, {
		name:  "2 items, max 1",
		items: 2,
		max:   1,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 2, 1).WithOrigin("maxItems"),
		},
	}, {
		name:  "0 items, max -1",
		items: 0,
		max:   -1,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 0, -1).WithOrigin("maxItems"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := make([]bool, tc.items)
			gotErrs := MaxItems(context.Background(), operation.Operation{}, field.NewPath("fldpath"), value, nil, tc.max)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMaxProperties(t *testing.T) {
	cases := []struct {
		name       string
		properties int
		max        int
		wantErrs   field.ErrorList
	}{{
		name:       "0 properties, max 0",
		properties: 0,
		max:        0,
	}, {
		name:       "1 property, max 0",
		properties: 1,
		max:        0,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 1, 0).WithOrigin("maxProperties"),
		},
	}, {
		name:       "1 property, max 1",
		properties: 1,
		max:        1,
	}, {
		name:       "2 properties, max 1",
		properties: 2,
		max:        1,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 2, 1).WithOrigin("maxProperties"),
		},
	}, {
		name:       "100000 properties, max 100000",
		properties: 100000,
		max:        100000,
	}, {
		name:       "100001 properties, max 100000",
		properties: 100001,
		max:        100000,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 100001, 100000).WithOrigin("maxProperties"),
		},
	}, {
		// Note: While JSON Schema does not allow negative values for maxProperties,
		// we test that the validator handles it safely if it ever occurs at runtime.
		name:       "0 properties, max -1",
		properties: 0,
		max:        -1,
		wantErrs: field.ErrorList{
			field.TooMany(field.NewPath("fldpath"), 0, -1).WithOrigin("maxProperties"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := make(map[string]string, tc.properties)
			for i := 0; i < tc.properties; i++ {
				value[fmt.Sprintf("%d", i)] = "value"
			}

			gotErrs := MaxProperties(context.Background(), operation.Operation{}, field.NewPath("fldpath"), value, nil, tc.max)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMinimum(t *testing.T) {
	testMinimumPositive[int](t)
	testMinimumNegative[int](t)
	testMinimumPositive[int8](t)
	testMinimumNegative[int8](t)
	testMinimumPositive[int16](t)
	testMinimumNegative[int16](t)
	testMinimumPositive[int32](t)
	testMinimumNegative[int32](t)
	testMinimumPositive[int64](t)
	testMinimumNegative[int64](t)

	testMinimumPositive[uint](t)
	testMinimumPositive[uint8](t)
	testMinimumPositive[uint16](t)
	testMinimumPositive[uint32](t)
	testMinimumPositive[uint64](t)
}

type minimumTestCase[T constraints.Integer] struct {
	min      T
	value    T
	wantErrs field.ErrorList
}

func testMinimumPositive[T constraints.Integer](t *testing.T) {
	t.Helper()
	cases := []minimumTestCase[T]{{
		min:   0,
		value: 0,
	}, {
		min:   1,
		value: 0,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be greater than or equal to").WithOrigin("minimum"),
		},
	}, {
		min:   1,
		value: 1,
	}, {
		min:   2,
		value: 1,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be greater than or equal to").WithOrigin("minimum"),
		},
	}}
	doTestMinimum[T](t, cases)
}

func testMinimumNegative[T constraints.Signed](t *testing.T) {
	t.Helper()
	cases := []minimumTestCase[T]{{
		min:   -1,
		value: -1,
	}, {
		min:   -1,
		value: -2,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be greater than or equal to").WithOrigin("minimum"),
		},
	}}

	doTestMinimum[T](t, cases)
}

func doTestMinimum[T constraints.Integer](t *testing.T, cases []minimumTestCase[T]) {
	t.Helper()
	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		name := fmt.Sprintf("%T (%v >= %v)", tc.value, tc.value, tc.min)
		t.Run(name, func(t *testing.T) {
			v := tc.value
			gotErrs := Minimum(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &v, nil, tc.min)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMaximum(t *testing.T) {
	testMaximumPositive[int](t)
	testMaximumNegative[int](t)
	testMaximumPositive[int8](t)
	testMaximumNegative[int8](t)
	testMaximumPositive[int16](t)
	testMaximumNegative[int16](t)
	testMaximumPositive[int32](t)
	testMaximumNegative[int32](t)
	testMaximumPositive[int64](t)
	testMaximumNegative[int64](t)

	testMaximumPositive[uint](t)
	testMaximumPositive[uint8](t)
	testMaximumPositive[uint16](t)
	testMaximumPositive[uint32](t)
	testMaximumPositive[uint64](t)
}

type maximumTestCase[T constraints.Integer] struct {
	max      T
	value    T
	wantErrs field.ErrorList
}

func testMaximumPositive[T constraints.Integer](t *testing.T) {
	t.Helper()
	cases := []maximumTestCase[T]{{
		max:   0,
		value: 0,
	}, {
		max:   0,
		value: 1,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be less than or equal to").WithOrigin("maximum"),
		},
	}, {
		max:   1,
		value: 1,
	}, {
		max:   1,
		value: 2,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be less than or equal to").WithOrigin("maximum"),
		},
	}}
	doTestMaximum[T](t, cases)
}

func testMaximumNegative[T constraints.Signed](t *testing.T) {
	t.Helper()
	cases := []maximumTestCase[T]{{
		max:   -1,
		value: -1,
	}, {
		max:   -2,
		value: -1,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), nil, "must be less than or equal to").WithOrigin("maximum"),
		},
	}}

	doTestMaximum[T](t, cases)
}

func doTestMaximum[T constraints.Integer](t *testing.T, cases []maximumTestCase[T]) {
	t.Helper()
	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		name := fmt.Sprintf("%T (%v <= %v)", tc.value, tc.value, tc.max)
		t.Run(name, func(t *testing.T) {
			v := tc.value
			gotErrs := Maximum(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &v, nil, tc.max)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMaxBytes(t *testing.T) {
	cases := []struct {
		name     string
		value    string
		max      int
		wantErrs field.ErrorList
	}{{
		name:     "empty string",
		value:    "",
		max:      0,
		wantErrs: nil,
	}, {
		name:  "zero length",
		value: "0",
		max:   0,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", 0).WithOrigin("maxBytes"),
		},
	}, {
		name:     "one character",
		value:    "0",
		max:      1,
		wantErrs: nil,
	}, {
		name:  "two characters",
		value: "01",
		max:   1,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", 1).WithOrigin("maxBytes"),
		},
	}, {
		value: "",
		max:   -1,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", -1).WithOrigin("maxBytes"),
		},
	}, {
		name:     "ascii-only characters, less bytes than max",
		value:    "abcdefghi",
		max:      10,
		wantErrs: nil,
	}, {
		name:     "multi-byte characters, less bytes than max",
		value:    "©®©®",
		max:      10,
		wantErrs: nil,
	}, {
		name:  "ascii-only characters, more bytes than max",
		value: "abcdefghijkl",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", 10).WithOrigin("maxBytes"),
		},
	}, {
		name:  "multi-byte characters, more bytes than max",
		value: "©®©®©©",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", 10).WithOrigin("maxBytes"),
		},
	}, {
		name:     "mixture of characters, less bytes than max",
		value:    "©abc®®",
		max:      10,
		wantErrs: nil,
	}, {
		name:  "mixture of characters, more bytes than max",
		value: "©abc®®abc",
		max:   10,
		wantErrs: field.ErrorList{
			field.TooLong(field.NewPath("fldpath"), "", 10).WithOrigin("maxBytes"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := tc.value
			gotErrs := MaxBytes(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &v, nil, tc.max)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestMinLength(t *testing.T) {
	cases := []struct {
		name     string
		value    string
		min      int
		wantErrs field.ErrorList // regex
	}{{
		name:     "empty string allowed",
		value:    "",
		min:      0,
		wantErrs: nil,
	}, {
		name:  "minimum length of one, empty string",
		value: "",
		min:   1,
		wantErrs: field.ErrorList{
			field.TooShort(field.NewPath("fldpath"), "", 1).WithOrigin("minLength"),
		},
	}, {
		name:     "minimum length of one, non-empty string",
		value:    "test",
		min:      1,
		wantErrs: nil,
	}, {
		name:  "minimum length of 10, 9 character string",
		value: "012345678",
		min:   10,
		wantErrs: field.ErrorList{
			field.TooShort(field.NewPath("fldpath"), "012345678", 10).WithOrigin("minLength"),
		},
	}, {
		name:     "minimum length of 10, 10 character string",
		value:    "0123456789",
		min:      10,
		wantErrs: nil,
	}, {
		name:     "negative minimum value",
		value:    "",
		min:      -1,
		wantErrs: nil,
	}, {
		name:  "ascii-only characters, less characters than min (n-1)",
		value: "abcdefghi",
		min:   10,
		wantErrs: field.ErrorList{
			field.TooShort(field.NewPath("fldpath"), "abcdefghi", 10).WithOrigin("minLength"),
		},
	}, {
		name:  "multi-byte characters, less characters than min (n-1)",
		value: "©®©®©®©®©",
		min:   10,
		wantErrs: field.ErrorList{
			field.TooShort(field.NewPath("fldpath"), "©®©®©®©®©", 10).WithOrigin("minLength"),
		},
	}, {
		name:     "ascii-only characters, more characters than min (n+1)",
		value:    "abcdefghijkl",
		min:      10,
		wantErrs: nil,
	}, {
		name:     "multi-byte characters, more characters than min (n+1)",
		value:    "©®©®©®©®©®©",
		min:      10,
		wantErrs: nil,
	}, {
		name:  "mixture of characters, maximum size in bytes of input is greater than min, rune count less than min",
		value: "©®©®©®", // 12 bytes, but 6 characters
		min:   10,
		wantErrs: field.ErrorList{
			field.TooShort(field.NewPath("fldpath"), "©®©®©®", 10).WithOrigin("minLength"),
		},
	}, {
		name:     "multi-byte characters, exact characters as min (n)",
		value:    "©®©®©®©®©®",
		min:      10,
		wantErrs: nil,
	}, {
		name:     "ascii-only characters, exact characters as min (n)",
		value:    "abcdefghij",
		min:      10,
		wantErrs: nil,
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := tc.value
			gotErrs := MinLength(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &v, nil, tc.min)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}
