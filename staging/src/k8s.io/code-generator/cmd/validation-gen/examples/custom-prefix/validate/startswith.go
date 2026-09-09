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

// Package validate holds the runtime validation functions that this project's
// tags generate calls to. They follow the same signature as
// k8s.io/apimachinery/pkg/api/validate.
package validate

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// StartsWith verifies that the specified value begins with prefix.
func StartsWith[T ~string](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, prefix string) field.ErrorList {
	if value == nil || strings.HasPrefix(string(*value), prefix) {
		return nil
	}
	return field.ErrorList{
		field.Invalid(fldPath, *value, fmt.Sprintf("must start with %q", prefix)).WithOrigin("startsWith"),
	}
}
