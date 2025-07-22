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
		value string
		valid sets.Set[string]
		err   bool
	}{{
		value: "a",
		valid: sets.New("a", "b", "c"),
		err:   false,
	}, {
		value: "x",
		valid: sets.New("c", "a", "b"),
		err:   true,
	}}

	for i, tc := range cases {
		result := Enum(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &tc.value, nil, tc.valid)
		if len(result) > 0 && !tc.err {
			t.Errorf("case %d: unexpected failure: %v", i, fmtErrs(result))
			continue
		}
		if len(result) == 0 && tc.err {
			t.Errorf("case %d: unexpected success", i)
			continue
		}
		if len(result) > 0 {
			if len(result) > 1 {
				t.Errorf("case %d: unexepected multi-error: %v", i, fmtErrs(result))
				continue
			}
			if want, got := `supported values: "a", "b", "c"`, result[0].Detail; got != want {
				t.Errorf("case %d: wrong error, expected: %q, got: %q", i, want, got)
			}
		}
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
		value StringType
		valid sets.Set[StringType]
		err   bool
	}{{
		value: "foo",
		valid: sets.New(NotStringFoo, NotStringBar, NotStringQux),
		err:   false,
	}, {
		value: "x",
		valid: sets.New(NotStringFoo, NotStringBar, NotStringQux),
		err:   true,
	}}

	for i, tc := range cases {
		result := Enum(context.Background(), operation.Operation{}, field.NewPath("fldpath"), &tc.value, nil, tc.valid)
		if len(result) > 0 && !tc.err {
			t.Errorf("case %d: unexpected failure: %v", i, fmtErrs(result))
			continue
		}
		if len(result) == 0 && tc.err {
			t.Errorf("case %d: unexpected success", i)
			continue
		}
		if len(result) > 0 {
			if len(result) > 1 {
				t.Errorf("case %d: unexepected multi-error: %v", i, fmtErrs(result))
				continue
			}
			if want, got := `supported values: "bar", "foo", "qux"`, result[0].Detail; got != want {
				t.Errorf("case %d: wrong error, expected: %q, got: %q", i, want, got)
			}
		}
	}
}
