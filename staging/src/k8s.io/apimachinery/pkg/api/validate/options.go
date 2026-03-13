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

// IfOption conditionally evaluates a validation function. If the option and enabled are both true the validator
// is called. If the option and enabled are both false the validator is called. Otherwise, the validator is not called.
func IfOption[T any](ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T,
	optionName string, enabled bool, validator func(context.Context, operation.Operation, *field.Path, *T, *T) field.ErrorList,
) field.ErrorList {
	if op.HasOption(optionName) == enabled {
		return validator(ctx, op, fldPath, value, oldValue)
	}
	return nil
}
