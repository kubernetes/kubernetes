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
	"regexp"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestEqualTo(t *testing.T) {
	type obj struct {
		FieldA string
		FieldB string
	}

	extractA := func(o *obj) string {
		if o == nil {
			return ""
		}
		return o.FieldA
	}
	extractB := func(o *obj) string {
		if o == nil {
			return ""
		}
		return o.FieldB
	}

	cases := []struct {
		name   string
		op     operation.Operation
		obj    *obj
		oldObj *obj
		err    string // regex; empty means expect no error
	}{{
		name: "create: fields equal",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{FieldA: "same", FieldB: "same"},
	}, {
		name: "create: fields differ",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{FieldA: "aaa", FieldB: "bbb"},
		err:  `fldpath\.fieldA: Invalid value: .*: must be equal to ` + "`fieldB`",
	}, {
		name: "create: both zero",
		op:   operation.Operation{Type: operation.Create},
		obj:  &obj{},
	}, {
		name: "create: nil obj",
		op:   operation.Operation{Type: operation.Create},
		obj:  nil,
	}, {
		name:   "ratchet: neither field changed",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{FieldA: "aaa", FieldB: "bbb"},
		oldObj: &obj{FieldA: "aaa", FieldB: "bbb"},
	}, {
		name:   "update: fieldA changed, now differs",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{FieldA: "new", FieldB: "bbb"},
		oldObj: &obj{FieldA: "old", FieldB: "bbb"},
		err:    `fldpath\.fieldA: Invalid value: .*: must be equal to ` + "`fieldB`",
	}, {
		name:   "update: fieldB changed, now differs",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{FieldA: "aaa", FieldB: "new"},
		oldObj: &obj{FieldA: "aaa", FieldB: "old"},
		err:    `fldpath\.fieldA: Invalid value: .*: must be equal to ` + "`fieldB`",
	}, {
		name:   "update: both changed, now equal",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{FieldA: "same", FieldB: "same"},
		oldObj: &obj{FieldA: "old-a", FieldB: "old-b"},
	}, {
		name:   "update: no oldObj (nil), fields differ",
		op:     operation.Operation{Type: operation.Update},
		obj:    &obj{FieldA: "aaa", FieldB: "bbb"},
		oldObj: nil,
		err:    `fldpath\.fieldA: Invalid value: .*: must be equal to ` + "`fieldB`",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := EqualTo(context.Background(), tc.op,
				field.NewPath("fldpath"), tc.obj, tc.oldObj,
				"fieldA", extractA,
				"fieldB", extractB)
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
				if result[0].Origin != "equalTo" {
					t.Errorf("expected origin %q, got %q", "equalTo", result[0].Origin)
				}
			}
		})
	}
}
