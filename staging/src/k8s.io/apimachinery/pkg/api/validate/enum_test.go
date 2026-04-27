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
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestEnum(t *testing.T) {
	cases := []struct {
		name      string
		value     string
		valid     sets.Set[string]
		expectErr string
	}{{
		name:      "valid value",
		value:     "a",
		valid:     sets.New("a", "b", "c"),
		expectErr: "",
	}, {
		name:      "invalid value",
		value:     "x",
		valid:     sets.New("a", "b", "c"),
		expectErr: `fldpath: Unsupported value: "x": supported values: "a", "b", "c"`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			op := operation.Operation{Type: operation.Create}
			errs := Enum(context.Background(), op, field.NewPath("fldpath"), &tc.value, nil, tc.valid, nil)

			if tc.expectErr == "" {
				if len(errs) > 0 {
					t.Fatalf("expected no error, but got: %v", errs)
				}
			} else {
				if len(errs) == 0 {
					t.Fatal("expected an error, but got none")
				}
				if len(errs) > 1 {
					t.Fatalf("expected a single error, but got: %v", errs)
				}
				if errs[0].Error() != tc.expectErr {
					t.Errorf("expected error %q, but got %q", tc.expectErr, errs[0].Error())
				}
			}
		})
	}
}

func TestEnumTypedef(t *testing.T) {
	type StringType string
	const (
		NotStringFoo StringType = "foo"
		NotStringBar StringType = "bar"
		NotStringQux StringType = "qux"
	)

	cases := []struct {
		name      string
		value     StringType
		valid     sets.Set[StringType]
		expectErr string
	}{{
		name:      "valid value",
		value:     "foo",
		valid:     sets.New(NotStringFoo, NotStringBar, NotStringQux),
		expectErr: "",
	}, {
		name:      "invalid value",
		value:     "x",
		valid:     sets.New(NotStringFoo, NotStringBar, NotStringQux),
		expectErr: `fldpath: Unsupported value: "x": supported values: "bar", "foo", "qux"`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			op := operation.Operation{Type: operation.Create}
			errs := Enum(context.Background(), op, field.NewPath("fldpath"), &tc.value, nil, tc.valid, nil)

			if tc.expectErr == "" {
				if len(errs) > 0 {
					t.Fatalf("expected no error, but got: %v", errs)
				}
			} else {
				if len(errs) == 0 {
					t.Fatal("expected an error, but got none")
				}
				if len(errs) > 1 {
					t.Fatalf("expected a single error, but got: %v", errs)
				}
				if errs[0].Error() != tc.expectErr {
					t.Errorf("expected error %q, but got %q", tc.expectErr, errs[0].Error())
				}
			}
		})
	}
}

func TestEnumExclude(t *testing.T) {
	type TestEnum string
	const (
		ValueA TestEnum = "A"
		ValueB TestEnum = "B"
		ValueC TestEnum = "C"
		ValueD TestEnum = "D"
	)

	const (
		FeatureA = "FeatureA"
		FeatureB = "FeatureB"
	)

	testEnumValues := sets.New(ValueA, ValueB, ValueC, ValueD)
	testEnumExclusions := []EnumExclusion[TestEnum]{
		{Value: ValueA, Option: FeatureA, ExcludeWhen: true},
		{Value: ValueB, Option: FeatureB, ExcludeWhen: false},
		{Value: ValueD, Option: FeatureA, ExcludeWhen: true},
		{Value: ValueD, Option: FeatureB, ExcludeWhen: false},
	}

	testCases := []struct {
		name      string
		value     TestEnum
		opts      []string
		expectErr string
	}{
		{
			name:  "no options, A is valid",
			value: ValueA,
		},
		{
			name:      "no options, B is invalid",
			value:     ValueB,
			expectErr: `fld: Unsupported value: "B": supported values: "A", "C"`,
		},
		{
			name:      "no options, D is invalid",
			value:     ValueD,
			expectErr: `fld: Unsupported value: "D": supported values: "A", "C"`,
		},
		{
			name:      "FeatureA enabled, A is invalid",
			value:     ValueA,
			opts:      []string{FeatureA},
			expectErr: `fld: Unsupported value: "A": supported values: "C"`,
		},
		{
			name:      "FeatureA enabled, B is invalid",
			value:     ValueB,
			opts:      []string{FeatureA},
			expectErr: `fld: Unsupported value: "B": supported values: "C"`,
		},
		{
			name:  "FeatureB enabled, A is valid",
			value: ValueA,
			opts:  []string{FeatureB},
		},
		{
			name:  "FeatureB enabled, B is valid",
			value: ValueB,
			opts:  []string{FeatureB},
		},
		{
			name:      "FeatureA and FeatureB enabled, A is invalid",
			value:     ValueA,
			opts:      []string{FeatureA, FeatureB},
			expectErr: `fld: Unsupported value: "A": supported values: "B", "C"`,
		},
		{
			name:  "FeatureA and FeatureB enabled, B is valid",
			value: ValueB,
			opts:  []string{FeatureA, FeatureB},
		},
		{
			name:      "FeatureA and FeatureB enabled, D is invalid",
			value:     ValueD,
			opts:      []string{FeatureA, FeatureB},
			expectErr: `fld: Unsupported value: "D": supported values: "B", "C"`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			op := operation.Operation{Type: operation.Create, Options: tc.opts}
			errs := Enum(context.Background(), op, field.NewPath("fld"), &tc.value, nil, testEnumValues, testEnumExclusions)

			if tc.expectErr == "" {
				if len(errs) > 0 {
					t.Fatalf("expected no error, but got: %v", errs)
				}
			} else {
				if len(errs) == 0 {
					t.Fatal("expected an error, but got none")
				}
				if len(errs) > 1 {
					t.Fatalf("expected a single error, but got: %v", errs)
				}
				if errs[0].Error() != tc.expectErr {
					t.Errorf("expected error %q, but got %q", tc.expectErr, errs[0].Error())
				}
			}
		})
	}
}
