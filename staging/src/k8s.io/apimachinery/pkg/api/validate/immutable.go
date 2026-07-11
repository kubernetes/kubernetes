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

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Immutable verifies that the specified value has not changed in the course of
// an update operation. It does nothing if the old value is not provided.
//
// This function unconditionally returns a validation error as it
// relies on the default ratcheting mechanism to only be called when a
// change to the field has already been detected.  This avoids a redundant
// equivalence check across ratcheting and this function.
func Immutable[T any](_ context.Context, op operation.Operation, fldPath *field.Path, _, _ T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}
	return field.ErrorList{
		field.Invalid(fldPath, nil, "field is immutable").WithOrigin("immutable"),
	}
}
