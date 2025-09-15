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

package fuzzer

import (
	"reflect"
	"strings"

	"sigs.k8s.io/randfill"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/utils/ptr"
)

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// Funcs returns the fuzzer functions for the apiextensions apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *apiextensions.CustomResourceDefinitionSpec, c randfill.Continue) {
			c.FillNoCustom(obj)

			// match our defaulter
			if len(obj.Scope) == 0 {
				obj.Scope = apiextensions.NamespaceScoped
			}
			if len(obj.Names.Singular) == 0 {
				obj.Names.Singular = strings.ToLower(obj.Names.Kind)
			}
			if len(obj.Names.ListKind) == 0 && len(obj.Names.Kind) > 0 {
				obj.Names.ListKind = obj.Names.Kind + "List"
			}
			if len(obj.Versions) == 0 && len(obj.Version) == 0 {
				// internal object must have a version to roundtrip all fields
				obj.Version = "v1"
			}
			if len(obj.Versions) == 0 && len(obj.Version) != 0 {
				obj.Versions = []apiextensions.CustomResourceDefinitionVersion{
					{
						Name:    obj.Version,
						Served:  true,
						Storage: true,
					},
				}
			} else if len(obj.Versions) != 0 {
				obj.Version = obj.Versions[0].Name
			}
			if len(obj.AdditionalPrinterColumns) == 0 {
				obj.AdditionalPrinterColumns = []apiextensions.CustomResourceColumnDefinition{
					{Name: "Age", Type: "date", Description: swaggerMetadataDescriptions["creationTimestamp"], JSONPath: ".metadata.creationTimestamp"},
				}
			}
			c.Fill(&obj.SelectableFields)
			if obj.Conversion == nil {
				obj.Conversion = &apiextensions.CustomResourceConversion{
					Strategy: apiextensions.NoneConverter,
				}
			}
			if obj.Conversion.Strategy == apiextensions.WebhookConverter && len(obj.Conversion.ConversionReviewVersions) == 0 {
				obj.Conversion.ConversionReviewVersions = []string{"v1beta1"}
			}
			if obj.PreserveUnknownFields == nil {
				obj.PreserveUnknownFields = ptr.To(true)
			}

			// Move per-version schema, subresources, additionalPrinterColumns, selectableFields to the top-level.
			// This is required by validation in v1beta1, and by round-tripping in v1.
			if len(obj.Versions) == 1 {
				if obj.Versions[0].Schema != nil {
					obj.Validation = obj.Versions[0].Schema
					obj.Versions[0].Schema = nil
				}
				if obj.Versions[0].AdditionalPrinterColumns != nil {
					obj.AdditionalPrinterColumns = obj.Versions[0].AdditionalPrinterColumns
					obj.Versions[0].AdditionalPrinterColumns = nil
				}
				if obj.Versions[0].SelectableFields != nil {
					obj.SelectableFields = obj.Versions[0].SelectableFields
					obj.Versions[0].SelectableFields = nil
				}
				if obj.Versions[0].Subresources != nil {
					obj.Subresources = obj.Versions[0].Subresources
					obj.Versions[0].Subresources = nil
				}
			}
		},
		func(obj *apiextensions.CustomResourceDefinition, c randfill.Continue) {
			c.FillNoCustom(obj)

			if len(obj.Status.StoredVersions) == 0 {
				for _, v := range obj.Spec.Versions {
					if v.Storage && !apiextensions.IsStoredVersion(obj, v.Name) {
						obj.Status.StoredVersions = append(obj.Status.StoredVersions, v.Name)
					}
				}
			}
		},
		func(obj *apiextensions.JSONSchemaProps, c randfill.Continue) {
			// we cannot use c.FuzzNoCustom because of the interface{} fields. So let's loop with reflection.
			vobj := reflect.ValueOf(obj).Elem()
			tobj := reflect.TypeOf(obj).Elem()
			for i := 0; i < tobj.NumField(); i++ {
				field := tobj.Field(i)
				switch field.Name {
				case "Default", "Enum", "Example", "Ref":
					continue
				default:
					isValue := true
					switch field.Type.Kind() {
					case reflect.Interface, reflect.Map, reflect.Slice, reflect.Pointer:
						isValue = false
					}
					if isValue || c.Intn(10) == 0 {
						c.Fill(vobj.Field(i).Addr().Interface())
					}
				}
			}
			if c.Bool() {
				validJSON := apiextensions.JSON(`{"some": {"json": "test"}, "string": 42}`)
				obj.Default = &validJSON
			}
			if c.Bool() {
				obj.Enum = []apiextensions.JSON{c.Float64(), c.String(0), c.Bool()}
			}
			if c.Bool() {
				validJSON := apiextensions.JSON(`"foobarbaz"`)
				obj.Example = &validJSON
			}
			if c.Bool() {
				validRef := "validRef"
				obj.Ref = &validRef
			}
			if len(obj.Type) == 0 {
				obj.Nullable = false // because this does not roundtrip through go-openapi
			}
			if obj.XIntOrString {
				obj.Type = ""
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrBool, c randfill.Continue) {
			if c.Bool() {
				obj.Allows = true
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fill(obj.Schema)
			} else {
				obj.Allows = c.Bool()
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrArray, c randfill.Continue) {
			// disallow both Schema and JSONSchemas to be nil.
			if c.Bool() {
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fill(obj.Schema)
			} else {
				obj.JSONSchemas = make([]apiextensions.JSONSchemaProps, c.Intn(3)+1)
				for i := range obj.JSONSchemas {
					c.Fill(&obj.JSONSchemas[i])
				}
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrStringArray, c randfill.Continue) {
			if c.Bool() {
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fill(obj.Schema)
			} else {
				c.Fill(&obj.Property)
			}
		},
		func(obj *int64, c randfill.Continue) {
			// JSON only supports 53 bits because everything is a float
			*obj = int64(c.Uint64()) & ((int64(1) << 53) - 1)
		},
		func(obj *apiextensions.ValidationRule, c randfill.Continue) {
			c.FillNoCustom(obj)
			if obj.Reason != nil && *(obj.Reason) == "" {
				obj.Reason = nil
			}
		},
	}
}
