/*
Copyright The Kubernetes Authors.

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
	"regexp"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestDependentRequired(t *testing.T) {
	type obj struct {
		Trigger    *string
		Dependent  *string
		OtherField *string
	}

	triggerIsSet := func(o *obj) bool { return o != nil && o.Trigger != nil }
	dependentIsSet := func(o *obj) bool { return o != nil && o.Dependent != nil }

	cases := []struct {
		name   string
		op     operation.Operation
		obj    *obj
		oldObj *obj
		err    string // regex; empty means expect no error
	}{{
		name: "create: trigger unset, dependent set",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Dependent: ptr.To("d")},
	}, {
		name: "create: trigger set, dependent set",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: ptr.To("t"), Dependent: ptr.To("d")},
	}, {
		name: "create: trigger set, dependent unset",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: ptr.To("t")},
		err:  `fldpath\.dependent: Required value: must be set when trigger is set`,
	}, {
		name: "create: nil obj",
		op:   operation.Operation{Type: operation.Create},
		obj:  nil,
	}, {
		name:   "ratchet: unrelated field changed, trigger and dependent set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: ptr.To("t"), OtherField: ptr.To("new")},
		oldObj: &obj{Trigger: ptr.To("t"), OtherField: ptr.To("old")},
	}, {
		name:   "ratchet: trigger value changed, set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: ptr.To("t")},
		oldObj: &obj{Trigger: ptr.To("old")},
	}, {
		name:   "update: trigger newly set",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: ptr.To("t")},
		oldObj: &obj{},
		err:    `fldpath\.dependent: Required value: must be set when trigger is set`,
	}, {
		name:   "update: dependent newly cleared",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: ptr.To("t")},
		oldObj: &obj{Trigger: ptr.To("t"), Dependent: ptr.To("d")},
		err:    `fldpath\.dependent: Required value: must be set when trigger is set`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := DependentRequired(context.Background(), tc.op,
				field.NewPath("fldpath"), tc.obj, tc.oldObj,
				"trigger", triggerIsSet,
				"dependent", dependentIsSet)
			if len(result) > 0 && tc.err == "" {
				t.Fatalf("unexpected failure: %v", fmtErrs(result))
			}
			if len(result) == 0 && tc.err != "" {
				t.Fatalf("unexpected success: expected %q", tc.err)
			}
			if len(result) > 1 {
				t.Fatalf("unexpected multi-error: %v", fmtErrs(result))
			}
			if len(result) > 0 {
				if !regexp.MustCompile(tc.err).MatchString(result[0].Error()) {
					t.Errorf("wrong error\nexpected: %q\n     got: %v", tc.err, fmtErrs(result))
				}
				if result[0].Origin != "dependentRequired" {
					t.Errorf("expected origin %q, got %q", "dependentRequired", result[0].Origin)
				}
			}
		})
	}
}
