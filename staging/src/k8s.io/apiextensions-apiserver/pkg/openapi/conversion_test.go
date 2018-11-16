/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/go-openapi/spec"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

func Test_ConvertJSONSchemaPropsToOpenAPIv2Schema(t *testing.T) {
	testStr := "test"
	testStr2 := "test2"
	testFloat64 := float64(6.4)
	testInt64 := int64(64)
	raw, _ := json.Marshal(testStr)
	raw2, _ := json.Marshal(testStr2)
	testApiextensionsJSON := v1beta1.JSON{Raw: raw}

	tests := []struct {
		name     string
		in       *v1beta1.JSONSchemaProps
		expected *spec.Schema
	}{
		{
			name: "id",
			in: &v1beta1.JSONSchemaProps{
				ID: testStr,
			},
			expected: new(spec.Schema).
				WithID(testStr),
		},
		{
			name: "$schema",
			in: &v1beta1.JSONSchemaProps{
				Schema: "test",
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Schema: "test",
				},
			},
		},
		{
			name: "$ref",
			in: &v1beta1.JSONSchemaProps{
				Ref: &testStr,
			},
			expected: spec.RefSchema(testStr),
		},
		{
			name: "description",
			in: &v1beta1.JSONSchemaProps{
				Description: testStr,
			},
			expected: new(spec.Schema).
				WithDescription(testStr),
		},
		{
			name: "type and format",
			in: &v1beta1.JSONSchemaProps{
				Type:   testStr,
				Format: testStr2,
			},
			expected: new(spec.Schema).
				Typed(testStr, testStr2),
		},
		{
			name: "title",
			in: &v1beta1.JSONSchemaProps{
				Title: testStr,
			},
			expected: new(spec.Schema).
				WithTitle(testStr),
		},
		{
			name: "default",
			in: &v1beta1.JSONSchemaProps{
				Default: &testApiextensionsJSON,
			},
			expected: new(spec.Schema).
				WithDefault(testStr),
		},
		{
			name: "maximum and exclusiveMaximum",
			in: &v1beta1.JSONSchemaProps{
				Maximum:          &testFloat64,
				ExclusiveMaximum: true,
			},
			expected: new(spec.Schema).
				WithMaximum(testFloat64, true),
		},
		{
			name: "minimum and exclusiveMinimum",
			in: &v1beta1.JSONSchemaProps{
				Minimum:          &testFloat64,
				ExclusiveMinimum: true,
			},
			expected: new(spec.Schema).
				WithMinimum(testFloat64, true),
		},
		{
			name: "maxLength",
			in: &v1beta1.JSONSchemaProps{
				MaxLength: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxLength(testInt64),
		},
		{
			name: "minLength",
			in: &v1beta1.JSONSchemaProps{
				MinLength: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinLength(testInt64),
		},
		{
			name: "pattern",
			in: &v1beta1.JSONSchemaProps{
				Pattern: testStr,
			},
			expected: new(spec.Schema).
				WithPattern(testStr),
		},
		{
			name: "maxItems",
			in: &v1beta1.JSONSchemaProps{
				MaxItems: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxItems(testInt64),
		},
		{
			name: "minItems",
			in: &v1beta1.JSONSchemaProps{
				MinItems: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinItems(testInt64),
		},
		{
			name: "uniqueItems",
			in: &v1beta1.JSONSchemaProps{
				UniqueItems: true,
			},
			expected: new(spec.Schema).
				UniqueValues(),
		},
		{
			name: "multipleOf",
			in: &v1beta1.JSONSchemaProps{
				MultipleOf: &testFloat64,
			},
			expected: new(spec.Schema).
				WithMultipleOf(testFloat64),
		},
		{
			name: "enum",
			in: &v1beta1.JSONSchemaProps{
				Enum: []v1beta1.JSON{{Raw: raw}, {Raw: raw2}},
			},
			expected: new(spec.Schema).
				WithEnum(testStr, testStr2),
		},
		{
			name: "maxProperties",
			in: &v1beta1.JSONSchemaProps{
				MaxProperties: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxProperties(testInt64),
		},
		{
			name: "minProperties",
			in: &v1beta1.JSONSchemaProps{
				MinProperties: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinProperties(testInt64),
		},
		{
			name: "required",
			in: &v1beta1.JSONSchemaProps{
				Required: []string{testStr, testStr2},
			},
			expected: new(spec.Schema).
				WithRequired(testStr, testStr2),
		},
		{
			name: "items single props",
			in: &v1beta1.JSONSchemaProps{
				Items: &v1beta1.JSONSchemaPropsOrArray{
					Schema: &v1beta1.JSONSchemaProps{
						Type: "boolean",
					},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Items: &spec.SchemaOrArray{
						Schema: spec.BooleanProperty(),
					},
				},
			},
		},
		{
			name: "items array props",
			in: &v1beta1.JSONSchemaProps{
				Items: &v1beta1.JSONSchemaPropsOrArray{
					JSONSchemas: []v1beta1.JSONSchemaProps{
						{Type: "boolean"},
						{Type: "string"},
					},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Items: &spec.SchemaOrArray{
						Schemas: []spec.Schema{
							*spec.BooleanProperty(),
							*spec.StringProperty(),
						},
					},
				},
			},
		},
		{
			name: "allOf",
			in: &v1beta1.JSONSchemaProps{
				AllOf: []v1beta1.JSONSchemaProps{
					{Type: "boolean"},
					{Type: "string"},
				},
			},
			expected: new(spec.Schema).
				WithAllOf(*spec.BooleanProperty(), *spec.StringProperty()),
		},
		{
			name: "oneOf",
			in: &v1beta1.JSONSchemaProps{
				OneOf: []v1beta1.JSONSchemaProps{
					{Type: "boolean"},
					{Type: "string"},
				},
			},
			expected: new(spec.Schema),
			// expected: &spec.Schema{
			// 	SchemaProps: spec.SchemaProps{
			// 		OneOf: []spec.Schema{
			// 			*spec.BooleanProperty(),
			// 			*spec.StringProperty(),
			// 		},
			// 	},
			// },
		},
		{
			name: "anyOf",
			in: &v1beta1.JSONSchemaProps{
				AnyOf: []v1beta1.JSONSchemaProps{
					{Type: "boolean"},
					{Type: "string"},
				},
			},
			expected: new(spec.Schema),
			// expected: &spec.Schema{
			// 	SchemaProps: spec.SchemaProps{
			// 		AnyOf: []spec.Schema{
			// 			*spec.BooleanProperty(),
			// 			*spec.StringProperty(),
			// 		},
			// 	},
			// },
		},
		{
			name: "not",
			in: &v1beta1.JSONSchemaProps{
				Not: &v1beta1.JSONSchemaProps{
					Type: "boolean",
				},
			},
			expected: new(spec.Schema),
			// expected: &spec.Schema{
			// 	SchemaProps: spec.SchemaProps{
			// 		Not: spec.BooleanProperty(),
			// 	},
			// },
		},
		{
			name: "properties",
			in: &v1beta1.JSONSchemaProps{
				Properties: map[string]v1beta1.JSONSchemaProps{
					testStr: {Type: "boolean"},
				},
			},
			expected: new(spec.Schema).
				SetProperty(testStr, *spec.BooleanProperty()),
		},
		{
			name: "additionalProperties",
			in: &v1beta1.JSONSchemaProps{
				AdditionalProperties: &v1beta1.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: &v1beta1.JSONSchemaProps{Type: "boolean"},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					AdditionalProperties: &spec.SchemaOrBool{
						Allows: true,
						Schema: spec.BooleanProperty(),
					},
				},
			},
		},
		{
			name: "patternProperties",
			in: &v1beta1.JSONSchemaProps{
				PatternProperties: map[string]v1beta1.JSONSchemaProps{
					testStr: {Type: "boolean"},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					PatternProperties: map[string]spec.Schema{
						testStr: *spec.BooleanProperty(),
					},
				},
			},
		},
		{
			name: "dependencies schema",
			in: &v1beta1.JSONSchemaProps{
				Dependencies: v1beta1.JSONSchemaDependencies{
					testStr: v1beta1.JSONSchemaPropsOrStringArray{
						Schema: &v1beta1.JSONSchemaProps{Type: "boolean"},
					},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Dependencies: spec.Dependencies{
						testStr: spec.SchemaOrStringArray{
							Schema: spec.BooleanProperty(),
						},
					},
				},
			},
		},
		{
			name: "dependencies string array",
			in: &v1beta1.JSONSchemaProps{
				Dependencies: v1beta1.JSONSchemaDependencies{
					testStr: v1beta1.JSONSchemaPropsOrStringArray{
						Property: []string{testStr2},
					},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Dependencies: spec.Dependencies{
						testStr: spec.SchemaOrStringArray{
							Property: []string{testStr2},
						},
					},
				},
			},
		},
		{
			name: "additionalItems",
			in: &v1beta1.JSONSchemaProps{
				AdditionalItems: &v1beta1.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: &v1beta1.JSONSchemaProps{Type: "boolean"},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					AdditionalItems: &spec.SchemaOrBool{
						Allows: true,
						Schema: spec.BooleanProperty(),
					},
				},
			},
		},
		{
			name: "definitions",
			in: &v1beta1.JSONSchemaProps{
				Definitions: v1beta1.JSONSchemaDefinitions{
					testStr: v1beta1.JSONSchemaProps{Type: "boolean"},
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Definitions: spec.Definitions{
						testStr: *spec.BooleanProperty(),
					},
				},
			},
		},
		{
			name: "externalDocs",
			in: &v1beta1.JSONSchemaProps{
				ExternalDocs: &v1beta1.ExternalDocumentation{
					Description: testStr,
					URL:         testStr2,
				},
			},
			expected: new(spec.Schema).
				WithExternalDocs(testStr, testStr2),
		},
		{
			name: "example",
			in: &v1beta1.JSONSchemaProps{
				Example: &testApiextensionsJSON,
			},
			expected: new(spec.Schema).
				WithExample(testStr),
		},
	}

	for _, test := range tests {
		out, err := ConvertJSONSchemaPropsToOpenAPIv2Schema(test.in)
		if err != nil {
			t.Errorf("unexpected error in converting openapi schema: %v", test.name)
		}
		if !reflect.DeepEqual(out, test.expected) {
			t.Errorf("result of conversion test '%v' didn't match, want: %v; got: %v", test.name, *test.expected, *out)
		}
	}
}
