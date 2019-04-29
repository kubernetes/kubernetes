/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"reflect"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

var intOrStringAnyOf = []NestedValueValidation{
	{ForbiddenGenerics: Generic{
		Type: "integer",
	}},
	{ForbiddenGenerics: Generic{
		Type: "string",
	}},
}

// ValidateStructural checks that s is a structural schema with the invariants:
//
// * structurality: both `ForbiddenGenerics` and `ForbiddenExtensions` only have zero values, with the two exceptions of (4).
// * RawExtension: for every schema with `x-kubernetes-embedded-resource: true`, `x-kubernetes-preserve-unknown-fields: true` and `type: object` are set
// * IntOrString: for `x-kubernetes-int-or-string: true` either `type` is empty under `anyOf` and `allOf` or the schema structure is either
//
// ValidateStructural does not check for completeness.
func ValidateStructural(s *Structural, fldPath *field.Path) field.ErrorList {
	if s == nil {
		return nil
	}

	allErrs := field.ErrorList{}

	allErrs = append(allErrs, ValidateStructural(s.Items, fldPath.Child("items"))...)
	for k, v := range s.Properties {
		allErrs = append(allErrs, ValidateStructural(&v, fldPath.Child("properties").Key(k))...)
	}
	allErrs = append(allErrs, ValidateGeneric(&s.Generic, fldPath)...)
	allErrs = append(allErrs, ValidateExtensions(&s.Extensions, fldPath)...)

	// detect the two IntOrString exceptions:
	// 1) anyOf:
	//    - type: integer
	//    - type: string
	// 2) allOf:
	//    - anyOf:
	//      - type: integer
	//      - type: string
	//    - <NestedValueValidation>
	skipAnyOf := false
	skipFirstAllOfAnyOf := false
	if !s.XIntOrString {
	} else if s.ValueValidation == nil {
	} else if len(s.ValueValidation.AnyOf) == 2 && reflect.DeepEqual(s.ValueValidation.AnyOf, intOrStringAnyOf) {
		skipAnyOf = true
	} else if len(s.ValueValidation.AllOf) >= 1 && len(s.ValueValidation.AllOf[0].AnyOf) == 2 && reflect.DeepEqual(s.ValueValidation.AllOf[0].AnyOf, intOrStringAnyOf) {
		skipFirstAllOfAnyOf = true
	}

	allErrs = append(allErrs, ValidateValueValidation(s.ValueValidation, skipAnyOf, skipFirstAllOfAnyOf, fldPath)...)

	if s.XEmbeddedResource && s.Type != "object" {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("type"), s.Type, "must be object if x-kubernetes-embedded-resource is true"))
	} else if len(s.Type) == 0 && !s.Extensions.XIntOrString && !s.Extensions.XPreserveUnknownFields && !s.Extensions.XEmbeddedResource {
		allErrs = append(allErrs, field.Required(fldPath.Child("type"), "must not be empty for specified object fields and arrays"))
	}

	return allErrs
}

// ValidateGeneric checks the generic fields of a structural schema.
func ValidateGeneric(g *Generic, fldPath *field.Path) field.ErrorList {
	if g == nil {
		return nil
	}

	allErrs := field.ErrorList{}

	if g.AdditionalProperties != nil {
		if g.AdditionalProperties.Structural != nil {
			allErrs = append(allErrs, ValidateStructural(g.AdditionalProperties.Structural, fldPath.Child("additionalProperties"))...)
		}
	}

	return allErrs
}

// ValidateExtensions checks Kubernetes vendor extensions of a structural schema.
func ValidateExtensions(x *Extensions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if x.XIntOrString && x.XPreserveUnknownFields {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("x-kubernetes-preserve-unknown-fields"), x.XPreserveUnknownFields, "must be false if x-kubernetes-int-or-string is true"))
	}
	if x.XIntOrString && x.XEmbeddedResource {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("x-kubernetes-embedded-resource"), x.XEmbeddedResource, "must be false if x-kubernetes-int-or-string is true"))
	}
	if x.XEmbeddedResource && !x.XPreserveUnknownFields {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("x-kubernetes-preserve-unknown-fields"), x.XPreserveUnknownFields, "must be true if x-kubernetes-embedded-resource is true"))
	}

	return allErrs
}

// ValidateValueValidation checks the value validation in a structural schema.
func ValidateValueValidation(v *ValueValidation, skipAnyOf, skipFirstAllOfAnyOf bool, fldPath *field.Path) field.ErrorList {
	if v == nil {
		return nil
	}

	allErrs := field.ErrorList{}

	if !skipAnyOf {
		for i := range v.AnyOf {
			allErrs = append(allErrs, ValidateNestedValueValidation(&v.AnyOf[i], false, false, fldPath.Child("anyOf").Index(i))...)
		}
	}

	for i := range v.AllOf {
		skipAnyOf := false
		if skipFirstAllOfAnyOf && i == 0 {
			skipAnyOf = true
		}
		allErrs = append(allErrs, ValidateNestedValueValidation(&v.AllOf[i], skipAnyOf, false, fldPath.Child("allOf").Index(i))...)
	}

	for i := range v.OneOf {
		allErrs = append(allErrs, ValidateNestedValueValidation(&v.OneOf[i], false, false, fldPath.Child("oneOf").Index(i))...)
	}

	allErrs = append(allErrs, ValidateNestedValueValidation(v.Not, false, false, fldPath.Child("not"))...)

	return allErrs
}

// ValidateNestedValueValidation checks the nested value validation under a logic junctor in a structural schema.
func ValidateNestedValueValidation(v *NestedValueValidation, skipAnyOf, skipAllOfAnyOf bool, fldPath *field.Path) field.ErrorList {
	if v == nil {
		return nil
	}

	allErrs := field.ErrorList{}

	allErrs = append(allErrs, ValidateValueValidation(&v.ValueValidation, skipAnyOf, skipAllOfAnyOf, fldPath)...)
	allErrs = append(allErrs, ValidateNestedValueValidation(v.Items, false, false, fldPath.Child("items"))...)

	for k, fld := range v.Properties {
		allErrs = append(allErrs, ValidateNestedValueValidation(&fld, false, false, fldPath.Child("properties").Key(k))...)
	}

	if len(v.ForbiddenGenerics.Type) > 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("type"), "must be empty to be structural"))
	}
	if v.ForbiddenGenerics.AdditionalProperties != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("additionalProperties"), "must be undefined to be structural"))
	}
	if v.ForbiddenGenerics.Default != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("default"), "must be undefined to be structural"))
	}
	if len(v.ForbiddenGenerics.Title) > 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("title"), "must be empty to be structural"))
	}
	if len(v.ForbiddenGenerics.Description) > 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("description"), "must be empty to be structural"))
	}
	if v.ForbiddenGenerics.Nullable {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("nullable"), "must be false to be structural"))
	}

	if v.ForbiddenExtensions.XPreserveUnknownFields {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("x-kubernetes-preserve-unknown-fields"), "must be false to be structural"))
	}
	if v.ForbiddenExtensions.XEmbeddedResource {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("x-kubernetes-embedded-resource"), "must be false to be structural"))
	}
	if v.ForbiddenExtensions.XIntOrString {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("x-kubernetes-int-or-string"), "must be false to be structural"))
	}

	return allErrs
}
