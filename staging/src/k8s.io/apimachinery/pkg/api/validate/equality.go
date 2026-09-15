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
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NEQ validates that the specified comparable value is not equal to the disallowed value.
func NEQ[T comparable](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, disallowed T) field.ErrorList {
	if value == nil {
		return nil
	}
	if *value == disallowed {
		return field.ErrorList{
			field.Invalid(fldPath, *value, content.NEQError(disallowed)).WithOrigin("neq"),
		}
	}
	return nil
}

// EqualTo verifies that two sibling fields have equal values. The tagged field
// (fieldName/fieldExtractor) must have the same value as the referenced sibling
// (siblingName/siblingExtractor). The error is reported at
// fldPath.Child(fieldName). On Update, the check is skipped if neither field's
// value changed from oldObj, so unrelated updates can proceed past a
// pre-existing violation.
func EqualTo[T any, V comparable](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *T,
	fieldName string, fieldExtractor ExtractorFn[*T, V],
	siblingName string, siblingExtractor ExtractorFn[*T, V],
) field.ErrorList {
	if obj == nil {
		return nil
	}
	fieldVal := fieldExtractor(obj)
	siblingVal := siblingExtractor(obj)
	if op.Type == operation.Update && oldObj != nil {
		oldFieldVal := fieldExtractor(oldObj)
		oldSiblingVal := siblingExtractor(oldObj)
		// Skip if neither side changed - ratcheting.
		if fieldVal == oldFieldVal && siblingVal == oldSiblingVal {
			return nil
		}
	}
	if fieldVal != siblingVal {
		return field.ErrorList{
			field.Invalid(fldPath.Child(fieldName), fieldVal,
				content.EqualToError(siblingName)).WithOrigin("equalTo"),
		}
	}
	return nil
}
