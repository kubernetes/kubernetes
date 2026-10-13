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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestFixedResult(t *testing.T) {
	cases := []struct {
		value any
		pass  bool
	}{{
		value: "",
		pass:  false,
	}, {
		value: "",
		pass:  true,
	}, {
		value: "nonempty",
		pass:  false,
	}, {
		value: "nonempty",
		pass:  true,
	}, {
		value: 0,
		pass:  false,
	}, {
		value: 0,
		pass:  true,
	}, {
		value: 1,
		pass:  false,
	}, {
		value: 1,
		pass:  true,
	}, {
		value: false,
		pass:  false,
	}, {
		value: false,
		pass:  true,
	}, {
		value: true,
		pass:  false,
	}, {
		value: true,
		pass:  true,
	}, {
		value: nil,
		pass:  false,
	}, {
		value: nil,
		pass:  true,
	}, {
		value: ptr.To(""),
		pass:  false,
	}, {
		value: ptr.To(""),
		pass:  true,
	}, {
		value: ptr.To("nonempty"),
		pass:  false,
	}, {
		value: ptr.To("nonempty"),
		pass:  true,
	}, {
		value: []string(nil),
		pass:  false,
	}, {
		value: []string(nil),
		pass:  true,
	}, {
		value: []string{},
		pass:  false,
	}, {
		value: []string{},
		pass:  true,
	}, {
		value: []string{"s"},
		pass:  false,
	}, {
		value: []string{"s"},
		pass:  true,
	}, {
		value: map[string]string(nil),
		pass:  false,
	}, {
		value: map[string]string(nil),
		pass:  true,
	}, {
		value: map[string]string{},
		pass:  false,
	}, {
		value: map[string]string{},
		pass:  true,
	}, {
		value: map[string]string{"k": "v"},
		pass:  false,
	}, {
		value: map[string]string{"k": "v"},
		pass:  true,
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailExact()
	for i, tc := range cases {
		result := FixedResult(context.Background(), operation.Operation{}, field.NewPath("fldpath"), tc.value, nil, tc.pass, "detail string")
		if len(result) != 0 && tc.pass {
			t.Errorf("case %d: unexpected failure: %v", i, fmtErrs(result))
			continue
		}
		if len(result) == 0 && !tc.pass {
			t.Errorf("case %d: unexpected success", i)
			continue
		}
		if len(result) > 0 {
			if len(result) > 1 {
				t.Errorf("case %d: unexepected multi-error: %v", i, fmtErrs(result))
				continue
			}
			wantErrorList := field.ErrorList{
				field.Invalid(field.NewPath("fldpath"), tc.value, "forced failure: detail string").WithOrigin("validateFalse"),
			}
			matcher.Test(t, wantErrorList, result)
		}
	}
}
