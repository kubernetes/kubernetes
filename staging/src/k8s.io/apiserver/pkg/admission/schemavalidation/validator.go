/*
Copyright 2023 The Kubernetes Authors.

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

package schemavalidation

import (
	"context"
	"encoding/json"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	openapierrors "k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
	"k8s.io/kube-openapi/pkg/validation/validate"
)

type ValidatorFactory interface {
	// ForGroupKind prepares and returns a validator that converts any input
	// of the given GroupResource into the given GVK and validates it with
	// against the OpenAPI schema for the GVK.
	ForGroupVersionKind(gvk schema.GroupVersionKind, converter *runtime.Scheme) (Validator, error)
}

type Validator interface {
	Validate(ctx context.Context, new runtime.Object) field.ErrorList
	ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList
}

// Returns a validator that is capable of reading the GVK off of the provided
// object, and validating it against its schema.
//
// Should perform following validations from the single entrypoint:
//  1. OpenAPI
//  2. CEL
//  3. Name/Metadata
//  4. ListType & MapType
//
// The validator is also expected to implement ratcheting
func NewFactory(resolver resolver.SchemaResolver) (ValidatorFactory, error) {
	return &delegatingValidator{resolver: resolver}, nil
}

func (v *delegatingValidator) ForGroupVersionKind(gvk schema.GroupVersionKind, converter *runtime.Scheme) (Validator, error) {
	schema, err := v.resolver.ResolveSchema(gvk)
	if err != nil {
		return nil, err
	}
	return newSchemaValidator(schema, converter, gvk), nil
}

func newSchemaValidator(sch *spec.Schema, converter *runtime.Scheme, gvk schema.GroupVersionKind) Validator {
	return &schemaValidator{schema: sch, converter: converter, gvk: gvk}
}

type schemaValidator struct {
	schema    *spec.Schema
	converter *runtime.Scheme
	gvk       schema.GroupVersionKind
}

func (v *schemaValidator) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return v.ValidateUpdate(ctx, obj, nil)
}

func (v *schemaValidator) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	// Convert the object to the target version
	// Use unsafe since we will only read the object and toss when done, so sharing
	// fields is acceptable.
	converted, err := v.converter.UnsafeConvertToVersion(obj, v.gvk.GroupVersion())
	if err != nil {
		return field.ErrorList{field.InternalError(field.NewPath(""), err)}
	}

	var allErrs field.ErrorList

	// Convert the converted object to unstructured
	// In a future release we may instrument validation to work with
	// SMD's value.Value which is also used for SSA.
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(converted)
	if err != nil {
		return field.ErrorList{field.InternalError(field.NewPath(""), err)}
	}

	// Old is not yet considered for now, since ratcheting is not yet implemented
	// included with DeclarativeValidation
	//
	// Also not yet validating XListType, XMapType, or CEL expressions for now.
	validator := validate.NewSchemaValidator(v.schema, nil, "", strfmt.Default)
	allErrs = append(allErrs, kubeOpenAPIResultToFieldErrors(nil, validator.Validate(u))...)
	return allErrs
}

// delegatingValidator reads the GVK from the input object and delegates the
// validation request to other validators. It throws a validation error
// if the GVK being requested is not recognized.
type delegatingValidator struct {
	resolver resolver.SchemaResolver
}

// Temporarily copied from apiextensions-apiserver
func kubeOpenAPIResultToFieldErrors(fldPath *field.Path, result *validate.Result) field.ErrorList {
	var allErrs field.ErrorList
	for _, err := range result.Errors {
		switch err := err.(type) {

		case *openapierrors.Validation:
			errPath := fldPath
			if len(err.Name) > 0 && err.Name != "." {
				errPath = errPath.Child(strings.TrimPrefix(err.Name, "."))
			}

			switch err.Code() {
			case openapierrors.RequiredFailCode:
				allErrs = append(allErrs, field.Required(errPath, ""))

			case openapierrors.EnumFailCode:
				values := []string{}
				for _, allowedValue := range err.Values {
					if s, ok := allowedValue.(string); ok {
						values = append(values, s)
					} else {
						allowedJSON, _ := json.Marshal(allowedValue)
						values = append(values, string(allowedJSON))
					}
				}
				allErrs = append(allErrs, field.NotSupported(errPath, err.Value, values))

			case openapierrors.TooLongFailCode:
				value := interface{}("")
				if err.Value != nil {
					value = err.Value
				}
				max := int64(-1)
				if i, ok := err.Valid.(int64); ok {
					max = i
				}
				allErrs = append(allErrs, field.TooLongMaxLength(errPath, value, int(max)))

			case openapierrors.MaxItemsFailCode:
				actual := int64(-1)
				if i, ok := err.Value.(int64); ok {
					actual = i
				}
				max := int64(-1)
				if i, ok := err.Valid.(int64); ok {
					max = i
				}
				allErrs = append(allErrs, field.TooMany(errPath, int(actual), int(max)))

			case openapierrors.TooManyPropertiesCode:
				actual := int64(-1)
				if i, ok := err.Value.(int64); ok {
					actual = i
				}
				max := int64(-1)
				if i, ok := err.Valid.(int64); ok {
					max = i
				}
				allErrs = append(allErrs, field.TooMany(errPath, int(actual), int(max)))

			case openapierrors.InvalidTypeCode:
				value := interface{}("")
				if err.Value != nil {
					value = err.Value
				}
				allErrs = append(allErrs, field.TypeInvalid(errPath, value, err.Error()))

			default:
				value := interface{}("")
				if err.Value != nil {
					value = err.Value
				}
				allErrs = append(allErrs, field.Invalid(errPath, value, err.Error()))
			}

		default:
			allErrs = append(allErrs, field.Invalid(fldPath, "", err.Error()))
		}
	}
	return allErrs
}
