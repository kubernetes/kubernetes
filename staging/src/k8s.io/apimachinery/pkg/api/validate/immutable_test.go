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
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

type Struct struct {
	S string
	I int
	B bool
}

func TestImmutable(t *testing.T) {
	structA := Struct{"abc", 123, true}
	structB := Struct{"xyz", 456, false}

	for _, tc := range []struct {
		name string
		fn   func(operation.Operation, *field.Path) field.ErrorList
		fail bool
	}{{
		name: "nil both values",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable[int](context.Background(), op, fld, nil, nil)
		},
	}, {
		name: "nil value",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, nil, ptr.To(123))
		},
		fail: true,
	}, {
		name: "nil oldValue",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(123), nil)
		},
		fail: true,
	}, {
		name: "int",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(123), ptr.To(123))
		},
	}, {
		name: "int fail",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(123), ptr.To(456))
		},
		fail: true,
	}, {
		name: "string",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To("abc"), ptr.To("abc"))
		},
	}, {
		name: "string fail",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To("abc"), ptr.To("xyz"))
		},
		fail: true,
	}, {
		name: "bool",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(true), ptr.To(true))
		},
	}, {
		name: "bool fail",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(true), ptr.To(false))
		},
		fail: true,
	}, {
		name: "struct",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(structA), ptr.To(structA))
		},
	}, {
		name: "struct fail",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(structA), ptr.To(structB))
		},
		fail: true,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.fn(operation.Operation{Type: operation.Create}, field.NewPath(""))
			if len(errs) != 0 { // Create should always succeed
				t.Errorf("case %q (create): expected success: %v", tc.name, errs)
			}
			errs = tc.fn(operation.Operation{Type: operation.Update}, field.NewPath(""))
			if tc.fail && len(errs) == 0 {
				t.Errorf("case %q (update): expected failure", tc.name)
			} else if !tc.fail && len(errs) != 0 {
				t.Errorf("case %q (update): expected success: %v", tc.name, errs)
			}
		})
	}
}
