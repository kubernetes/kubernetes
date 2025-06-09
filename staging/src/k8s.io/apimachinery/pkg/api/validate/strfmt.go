/*
Copyright 2014 The Kubernetes Authors.

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

// ShortName verifies that the specified value is a valid "short name"
// (sometimes known as a "DNS label".  It must:
//   - not be empty
//   - start and end with lower-case alphanumeric characters
//   - contain only lower-case alphanumeric characters or dashes
//   - be less than 64 characters long
//
// All errors returned by this function will be "invalid" type errors. If the
// caller wants better errors, it must take responsibility for checking things
// like required/optional and max-length.
func ShortName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123Label((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-short-name"))
	}
	return allErrs
}
