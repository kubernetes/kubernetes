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

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ImmutableByCompare verifies that the specified value has not changed in the
// course of an update operation.  It does nothing if the old value is not
// provided. If the caller needs to compare types that are not trivially
// comparable, they should use ImmutableByReflect instead.
//
// Caution: structs with pointer fields satisfy comparable, but this function
// will only compare pointer values.  It does not compare the pointed-to
// values.
func ImmutableByCompare[T comparable](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}
	if value == nil && oldValue == nil {
		return nil
	}
	if value == nil || oldValue == nil || *value != *oldValue {
		return field.ErrorList{
			field.Forbidden(fldPath, "field is immutable"),
		}
	}
	return nil
}

// ImmutableByReflect verifies that the specified value has not changed in
// the course of an update operation.  It does nothing if the old value is not
// provided. Unlike ImmutableByCompare, this function can be used with types that are
// not directly comparable, at the cost of performance.
func ImmutableByReflect[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}
	if !equality.Semantic.DeepEqual(value, oldValue) {
		return field.ErrorList{
			field.Forbidden(fldPath, "field is immutable"),
		}
	}
	return nil
}
