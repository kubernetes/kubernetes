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

	"github.com/google/gofuzz"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
)

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// Funcs returns the fuzzer functions for the apiextensions apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *apiextensions.CustomResourceDefinitionSpec, c fuzz.Continue) {
			c.FuzzNoCustom(obj)

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
		},
		func(obj *apiextensions.CustomResourceDefinition, c fuzz.Continue) {
			c.FuzzNoCustom(obj)

			if len(obj.Status.StoredVersions) == 0 {
				for _, v := range obj.Spec.Versions {
					if v.Storage && !apiextensions.IsStoredVersion(obj, v.Name) {
						obj.Status.StoredVersions = append(obj.Status.StoredVersions, v.Name)
					}
				}
			}
		},
		func(obj *apiextensions.JSONSchemaProps, c fuzz.Continue) {
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
					case reflect.Interface, reflect.Map, reflect.Slice, reflect.Ptr:
						isValue = false
					}
					if isValue || c.Intn(10) == 0 {
						c.Fuzz(vobj.Field(i).Addr().Interface())
					}
				}
			}
			if c.RandBool() {
				validJSON := apiextensions.JSON(`{"some": {"json": "test"}, "string": 42}`)
				obj.Default = &validJSON
			}
			if c.RandBool() {
				obj.Enum = []apiextensions.JSON{c.Float64(), c.RandString(), c.RandBool()}
			}
			if c.RandBool() {
				validJSON := apiextensions.JSON(`"foobarbaz"`)
				obj.Example = &validJSON
			}
			if c.RandBool() {
				validRef := "validRef"
				obj.Ref = &validRef
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrBool, c fuzz.Continue) {
			if c.RandBool() {
				obj.Allows = true
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fuzz(obj.Schema)
			} else {
				obj.Allows = c.RandBool()
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrArray, c fuzz.Continue) {
			// disallow both Schema and JSONSchemas to be nil.
			if c.RandBool() {
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fuzz(obj.Schema)
			} else {
				obj.JSONSchemas = make([]apiextensions.JSONSchemaProps, c.Intn(3)+1)
				for i := range obj.JSONSchemas {
					c.Fuzz(&obj.JSONSchemas[i])
				}
			}
		},
		func(obj *apiextensions.JSONSchemaPropsOrStringArray, c fuzz.Continue) {
			if c.RandBool() {
				obj.Schema = &apiextensions.JSONSchemaProps{}
				c.Fuzz(obj.Schema)
			} else {
				c.Fuzz(&obj.Property)
			}
		},
	}
}
