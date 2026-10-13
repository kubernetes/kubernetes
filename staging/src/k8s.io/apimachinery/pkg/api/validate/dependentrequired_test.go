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
		obj:  &obj{Dependent: new("d")},
	}, {
		name: "create: trigger set, dependent set",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: new("t"), Dependent: new("d")},
	}, {
		name: "create: trigger set, dependent unset",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: new("t")},
		err:  `fldpath\.dependent: Required value: must be set when trigger is set`,
	}, {
		name: "create: nil obj",
		op:   operation.Operation{Type: operation.Create},
		obj:  nil,
	}, {
		name:   "ratchet: unrelated field changed, trigger and dependent set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t"), OtherField: new("new")},
		oldObj: &obj{Trigger: new("t"), OtherField: new("old")},
	}, {
		name:   "ratchet: trigger value changed, set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t")},
		oldObj: &obj{Trigger: new("old")},
	}, {
		name:   "update: trigger newly set",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t")},
		oldObj: &obj{},
		err:    `fldpath\.dependent: Required value: must be set when trigger is set`,
	}, {
		name:   "update: dependent newly cleared",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t")},
		oldObj: &obj{Trigger: new("t"), Dependent: new("d")},
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

func TestDependentForbidden(t *testing.T) {
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
		obj:  &obj{Dependent: new("d")},
	}, {
		name: "create: trigger set, dependent unset",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: new("t")},
	}, {
		name: "create: trigger set, dependent set",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{Trigger: new("t"), Dependent: new("d")},
		err:  `fldpath\.dependent: Forbidden: may not be set when trigger is set`,
	}, {
		name: "create: nil obj",
		op:   operation.Operation{Type: operation.Create},
		obj:  nil,
	}, {
		name:   "ratchet: unrelated field changed, trigger and dependent set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t"), Dependent: new("d"), OtherField: new("new")},
		oldObj: &obj{Trigger: new("t"), Dependent: new("d"), OtherField: new("old")},
	}, {
		name:   "ratchet: trigger value changed, set-ness unchanged",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t"), Dependent: new("d")},
		oldObj: &obj{Trigger: new("old"), Dependent: new("d")},
	}, {
		name:   "update: trigger newly set",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t"), Dependent: new("d")},
		oldObj: &obj{Dependent: new("d")},
		err:    `fldpath\.dependent: Forbidden: may not be set when trigger is set`,
	}, {
		name:   "update: dependent newly set",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{Trigger: new("t"), Dependent: new("d")},
		oldObj: &obj{Trigger: new("t")},
		err:    `fldpath\.dependent: Forbidden: may not be set when trigger is set`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := DependentForbidden(context.Background(), tc.op,
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
				if result[0].Origin != "dependentForbidden" {
					t.Errorf("expected origin %q, got %q", "dependentForbidden", result[0].Origin)
				}
			}
		})
	}
}
