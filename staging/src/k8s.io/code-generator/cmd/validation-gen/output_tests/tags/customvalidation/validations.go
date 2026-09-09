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

package customvalidation

import (
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// These functions are structural markers, not real rules: each emits one error
// at the path it is called with, so tests can assert where the generated
// traversal invokes custom validation.

func ValidateCustom_Struct(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *Struct) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_Struct")}
}

func ValidateCustom_Struct_StringField(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *string) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_Struct_StringField")}
}

func ValidateCustom_Struct_MaxLengthField(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *string) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_Struct_MaxLengthField")}
}

func ValidateCustom_StringType(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *StringType) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_StringType")}
}

func ValidateCustom_OtherStruct_StringField(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *string) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_OtherStruct_StringField")}
}

func ValidateCustom_OptionStruct_StringField(_ context.Context, _ operation.Operation, fldPath *field.Path, _, _ *string) field.ErrorList {
	return field.ErrorList{field.Invalid(fldPath, nil, "ValidateCustom_OptionStruct_StringField")}
}
