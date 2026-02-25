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

func TestImmutable(t *testing.T) {
	// The Immutable function relies on validation ratcheting to avoid being
	// called when old and new values are equivalent. This unit test only needs
	// to confirm two behaviors:
	// 1. The function does nothing for non-update operations (e.g., create).
	// 2. The function *always* returns an error for update operations, since
	//    ratcheting should have prevented the call if the values were unchanged.

	type simpleStruct struct {
		S string
	}

	for _, tc := range []struct {
		name string
		fn   func(op operation.Operation, fldPath *field.Path) field.ErrorList
	}{{
		name: "with primitive type",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, ptr.To(123), ptr.To(456))
		},
	}, {
		name: "with struct type",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			return Immutable(context.Background(), op, fld, &simpleStruct{S: "a"}, &simpleStruct{S: "b"})
		},
	}, {
		name: "with nil values",
		fn: func(op operation.Operation, fld *field.Path) field.ErrorList {
			// Explicitly type the nil to satisfy the generic function signature.
			return Immutable[*int](context.Background(), op, fld, nil, nil)
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			// Create operations should never return an error.
			errs := tc.fn(operation.Operation{Type: operation.Create}, field.NewPath("field"))
			if len(errs) != 0 {
				t.Errorf("expected success for create operation, but got errors: %v", errs)
			}

			// Update operations should always return exactly one error.
			errs = tc.fn(operation.Operation{Type: operation.Update}, field.NewPath("field"))
			if len(errs) == 0 {
				t.Errorf("expected a failure for update operation, but got success")
			} else if len(errs) > 1 {
				t.Errorf("expected exactly one error for update operation, but got %d: %v", len(errs), errs)
			}
		})
	}
}
