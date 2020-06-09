/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"encoding/json"
	"strings"

	openapierrors "github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NewSchemaValidator creates an openapi schema validator for the given CRD validation.
func NewSchemaValidator(customResourceValidation *apiextensions.CustomResourceValidation) (*validate.SchemaValidator, *spec.Schema, error) {
	// Convert CRD schema to openapi schema
	openapiSchema := &spec.Schema{}
	if customResourceValidation != nil {
		// TODO: replace with NewStructural(...).ToGoOpenAPI
		if err := ConvertJSONSchemaPropsWithPostProcess(customResourceValidation.OpenAPIV3Schema, openapiSchema, StripUnsupportedFormatsPostProcess); err != nil {
			return nil, nil, err
		}
	}
	return validate.NewSchemaValidator(openapiSchema, nil, "", strfmt.Default), openapiSchema, nil
}

// ValidateCustomResource validates the Custom Resource against the schema in the CustomResourceDefinition.
// CustomResource is a JSON data structure.
func ValidateCustomResource(fldPath *field.Path, customResource interface{}, validator *validate.SchemaValidator) field.ErrorList {
	if validator == nil {
		return nil
	}

	result := validator.Validate(customResource)
	if result.IsValid() {
		return nil
	}
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

// ConvertJSONSchemaProps converts the schema from apiextensions.JSONSchemaPropos to go-openapi/spec.Schema.
func ConvertJSONSchemaProps(in *apiextensions.JSONSchemaProps, out *spec.Schema) error {
	return ConvertJSONSchemaPropsWithPostProcess(in, out, nil)
}

// PostProcessFunc post-processes one node of a spec.Schema.
type PostProcessFunc func(*spec.Schema) error

// ConvertJSONSchemaPropsWithPostProcess converts the schema from apiextensions.JSONSchemaPropos to go-openapi/spec.Schema
// and run a post process step on each JSONSchemaProps node. postProcess is never called for nil schemas.
func ConvertJSONSchemaPropsWithPostProcess(in *apiextensions.JSONSchemaProps, out *spec.Schema, postProcess PostProcessFunc) error {
	if in == nil {
		return nil
	}

	out.ID = in.ID
	out.Schema = spec.SchemaURL(in.Schema)
	out.Description = in.Description
	if in.Type != "" {
		out.Type = spec.StringOrArray([]string{in.Type})
	}
	if in.XIntOrString {
		out.VendorExtensible.AddExtension("x-kubernetes-int-or-string", true)
		out.Type = spec.StringOrArray{"integer", "string"}
	}
	out.Nullable = in.Nullable
	out.Format = in.Format
	out.Title = in.Title
	out.Maximum = in.Maximum
	out.ExclusiveMaximum = in.ExclusiveMaximum
	out.Minimum = in.Minimum
	out.ExclusiveMinimum = in.ExclusiveMinimum
	out.MaxLength = in.MaxLength
	out.MinLength = in.MinLength
	out.Pattern = in.Pattern
	out.MaxItems = in.MaxItems
	out.MinItems = in.MinItems
	out.UniqueItems = in.UniqueItems
	out.MultipleOf = in.MultipleOf
	out.MaxProperties = in.MaxProperties
	out.MinProperties = in.MinProperties
	out.Required = in.Required

	if in.Default != nil {
		out.Default = *(in.Default)
	}
	if in.Example != nil {
		out.Example = *(in.Example)
	}

	if in.Enum != nil {
		out.Enum = make([]interface{}, len(in.Enum))
		for k, v := range in.Enum {
			out.Enum[k] = v
		}
	}

	if err := convertSliceOfJSONSchemaProps(&in.AllOf, &out.AllOf, postProcess); err != nil {
		return err
	}
	if err := convertSliceOfJSONSchemaProps(&in.OneOf, &out.OneOf, postProcess); err != nil {
		return err
	}
	if err := convertSliceOfJSONSchemaProps(&in.AnyOf, &out.AnyOf, postProcess); err != nil {
		return err
	}

	if in.Not != nil {
		in, out := &in.Not, &out.Not
		*out = new(spec.Schema)
		if err := ConvertJSONSchemaPropsWithPostProcess(*in, *out, postProcess); err != nil {
			return err
		}
	}

	var err error
	out.Properties, err = convertMapOfJSONSchemaProps(in.Properties, postProcess)
	if err != nil {
		return err
	}

	out.PatternProperties, err = convertMapOfJSONSchemaProps(in.PatternProperties, postProcess)
	if err != nil {
		return err
	}

	out.Definitions, err = convertMapOfJSONSchemaProps(in.Definitions, postProcess)
	if err != nil {
		return err
	}

	if in.Ref != nil {
		out.Ref, err = spec.NewRef(*in.Ref)
		if err != nil {
			return err
		}
	}

	if in.AdditionalProperties != nil {
		in, out := &in.AdditionalProperties, &out.AdditionalProperties
		*out = new(spec.SchemaOrBool)
		if err := convertJSONSchemaPropsorBool(*in, *out, postProcess); err != nil {
			return err
		}
	}

	if in.AdditionalItems != nil {
		in, out := &in.AdditionalItems, &out.AdditionalItems
		*out = new(spec.SchemaOrBool)
		if err := convertJSONSchemaPropsorBool(*in, *out, postProcess); err != nil {
			return err
		}
	}

	if in.Items != nil {
		in, out := &in.Items, &out.Items
		*out = new(spec.SchemaOrArray)
		if err := convertJSONSchemaPropsOrArray(*in, *out, postProcess); err != nil {
			return err
		}
	}

	if in.Dependencies != nil {
		in, out := &in.Dependencies, &out.Dependencies
		*out = make(spec.Dependencies, len(*in))
		for key, val := range *in {
			newVal := new(spec.SchemaOrStringArray)
			if err := convertJSONSchemaPropsOrStringArray(&val, newVal, postProcess); err != nil {
				return err
			}
			(*out)[key] = *newVal
		}
	}

	if in.ExternalDocs != nil {
		out.ExternalDocs = &spec.ExternalDocumentation{}
		out.ExternalDocs.Description = in.ExternalDocs.Description
		out.ExternalDocs.URL = in.ExternalDocs.URL
	}

	if postProcess != nil {
		if err := postProcess(out); err != nil {
			return err
		}
	}

	if in.XPreserveUnknownFields != nil {
		out.VendorExtensible.AddExtension("x-kubernetes-preserve-unknown-fields", *in.XPreserveUnknownFields)
	}
	if in.XEmbeddedResource {
		out.VendorExtensible.AddExtension("x-kubernetes-embedded-resource", true)
	}
	if len(in.XListMapKeys) != 0 {
		out.VendorExtensible.AddExtension("x-kubernetes-list-map-keys", in.XListMapKeys)
	}
	if in.XListType != nil {
		out.VendorExtensible.AddExtension("x-kubernetes-list-type", *in.XListType)
	}
	if in.XMapType != nil {
		out.VendorExtensible.AddExtension("x-kubernetes-map-type", *in.XMapType)
	}
	return nil
}

func convertSliceOfJSONSchemaProps(in *[]apiextensions.JSONSchemaProps, out *[]spec.Schema, postProcess PostProcessFunc) error {
	if in != nil {
		for _, jsonSchemaProps := range *in {
			schema := spec.Schema{}
			if err := ConvertJSONSchemaPropsWithPostProcess(&jsonSchemaProps, &schema, postProcess); err != nil {
				return err
			}
			*out = append(*out, schema)
		}
	}
	return nil
}

func convertMapOfJSONSchemaProps(in map[string]apiextensions.JSONSchemaProps, postProcess PostProcessFunc) (map[string]spec.Schema, error) {
	if in == nil {
		return nil, nil
	}

	out := make(map[string]spec.Schema)
	for k, jsonSchemaProps := range in {
		schema := spec.Schema{}
		if err := ConvertJSONSchemaPropsWithPostProcess(&jsonSchemaProps, &schema, postProcess); err != nil {
			return nil, err
		}
		out[k] = schema
	}
	return out, nil
}

func convertJSONSchemaPropsOrArray(in *apiextensions.JSONSchemaPropsOrArray, out *spec.SchemaOrArray, postProcess PostProcessFunc) error {
	if in.Schema != nil {
		in, out := &in.Schema, &out.Schema
		*out = new(spec.Schema)
		if err := ConvertJSONSchemaPropsWithPostProcess(*in, *out, postProcess); err != nil {
			return err
		}
	}
	if in.JSONSchemas != nil {
		in, out := &in.JSONSchemas, &out.Schemas
		*out = make([]spec.Schema, len(*in))
		for i := range *in {
			if err := ConvertJSONSchemaPropsWithPostProcess(&(*in)[i], &(*out)[i], postProcess); err != nil {
				return err
			}
		}
	}
	return nil
}

func convertJSONSchemaPropsorBool(in *apiextensions.JSONSchemaPropsOrBool, out *spec.SchemaOrBool, postProcess PostProcessFunc) error {
	out.Allows = in.Allows
	if in.Schema != nil {
		in, out := &in.Schema, &out.Schema
		*out = new(spec.Schema)
		if err := ConvertJSONSchemaPropsWithPostProcess(*in, *out, postProcess); err != nil {
			return err
		}
	}
	return nil
}

func convertJSONSchemaPropsOrStringArray(in *apiextensions.JSONSchemaPropsOrStringArray, out *spec.SchemaOrStringArray, postProcess PostProcessFunc) error {
	out.Property = in.Property
	if in.Schema != nil {
		in, out := &in.Schema, &out.Schema
		*out = new(spec.Schema)
		if err := ConvertJSONSchemaPropsWithPostProcess(*in, *out, postProcess); err != nil {
			return err
		}
	}
	return nil
}
