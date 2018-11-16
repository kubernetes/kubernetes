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
	"reflect"
	"testing"

	"github.com/go-openapi/spec"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/util/diff"
)

func Test_ConvertJSONSchemaPropsToOpenAPIv2Schema(t *testing.T) {
	testStr := "test"
	testStr2 := "test2"
	testFloat64 := float64(6.4)
	testInt64 := int64(64)
	testApiextensionsJSON := apiextensions.JSON(testStr)

	tests := []struct {
		name     string
		in       *apiextensions.JSONSchemaProps
		expected *spec.Schema
	}{
		{
			name: "id",
			in: &apiextensions.JSONSchemaProps{
				ID: testStr,
			},
			expected: new(spec.Schema).
				WithID(testStr),
		},
		{
			name: "$schema",
			in: &apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				Ref: &testStr,
			},
			expected: spec.RefSchema(testStr),
		},
		{
			name: "description",
			in: &apiextensions.JSONSchemaProps{
				Description: testStr,
			},
			expected: new(spec.Schema).
				WithDescription(testStr),
		},
		{
			name: "type and format",
			in: &apiextensions.JSONSchemaProps{
				Type:   testStr,
				Format: testStr2,
			},
			expected: new(spec.Schema).
				Typed(testStr, testStr2),
		},
		{
			name: "title",
			in: &apiextensions.JSONSchemaProps{
				Title: testStr,
			},
			expected: new(spec.Schema).
				WithTitle(testStr),
		},
		{
			name: "default",
			in: &apiextensions.JSONSchemaProps{
				Default: &testApiextensionsJSON,
			},
			expected: new(spec.Schema).
				WithDefault(testStr),
		},
		{
			name: "maximum and exclusiveMaximum",
			in: &apiextensions.JSONSchemaProps{
				Maximum:          &testFloat64,
				ExclusiveMaximum: true,
			},
			expected: new(spec.Schema).
				WithMaximum(testFloat64, true),
		},
		{
			name: "minimum and exclusiveMinimum",
			in: &apiextensions.JSONSchemaProps{
				Minimum:          &testFloat64,
				ExclusiveMinimum: true,
			},
			expected: new(spec.Schema).
				WithMinimum(testFloat64, true),
		},
		{
			name: "maxLength",
			in: &apiextensions.JSONSchemaProps{
				MaxLength: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxLength(testInt64),
		},
		{
			name: "minLength",
			in: &apiextensions.JSONSchemaProps{
				MinLength: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinLength(testInt64),
		},
		{
			name: "pattern",
			in: &apiextensions.JSONSchemaProps{
				Pattern: testStr,
			},
			expected: new(spec.Schema).
				WithPattern(testStr),
		},
		{
			name: "maxItems",
			in: &apiextensions.JSONSchemaProps{
				MaxItems: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxItems(testInt64),
		},
		{
			name: "minItems",
			in: &apiextensions.JSONSchemaProps{
				MinItems: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinItems(testInt64),
		},
		{
			name: "uniqueItems",
			in: &apiextensions.JSONSchemaProps{
				UniqueItems: true,
			},
			expected: new(spec.Schema).
				UniqueValues(),
		},
		{
			name: "multipleOf",
			in: &apiextensions.JSONSchemaProps{
				MultipleOf: &testFloat64,
			},
			expected: new(spec.Schema).
				WithMultipleOf(testFloat64),
		},
		{
			name: "enum",
			in: &apiextensions.JSONSchemaProps{
				Enum: []apiextensions.JSON{apiextensions.JSON(testStr), apiextensions.JSON(testStr2)},
			},
			expected: new(spec.Schema).
				WithEnum(testStr, testStr2),
		},
		{
			name: "maxProperties",
			in: &apiextensions.JSONSchemaProps{
				MaxProperties: &testInt64,
			},
			expected: new(spec.Schema).
				WithMaxProperties(testInt64),
		},
		{
			name: "minProperties",
			in: &apiextensions.JSONSchemaProps{
				MinProperties: &testInt64,
			},
			expected: new(spec.Schema).
				WithMinProperties(testInt64),
		},
		{
			name: "required",
			in: &apiextensions.JSONSchemaProps{
				Required: []string{testStr, testStr2},
			},
			expected: new(spec.Schema).
				WithRequired(testStr, testStr2),
		},
		{
			name: "items single props",
			in: &apiextensions.JSONSchemaProps{
				Items: &apiextensions.JSONSchemaPropsOrArray{
					Schema: &apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				Items: &apiextensions.JSONSchemaPropsOrArray{
					JSONSchemas: []apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				AllOf: []apiextensions.JSONSchemaProps{
					{Type: "boolean"},
					{Type: "string"},
				},
			},
			expected: new(spec.Schema).
				WithAllOf(*spec.BooleanProperty(), *spec.StringProperty()),
		},
		{
			name: "oneOf",
			in: &apiextensions.JSONSchemaProps{
				OneOf: []apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				AnyOf: []apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				Not: &apiextensions.JSONSchemaProps{
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
			name: "nested logic",
			in: &apiextensions.JSONSchemaProps{
				AllOf: []apiextensions.JSONSchemaProps{
					{
						Not: &apiextensions.JSONSchemaProps{
							Type: "boolean",
						},
					},
					{
						AnyOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{
						OneOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{Type: "string"},
				},
				AnyOf: []apiextensions.JSONSchemaProps{
					{
						Not: &apiextensions.JSONSchemaProps{
							Type: "boolean",
						},
					},
					{
						AnyOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{
						OneOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{Type: "string"},
				},
				OneOf: []apiextensions.JSONSchemaProps{
					{
						Not: &apiextensions.JSONSchemaProps{
							Type: "boolean",
						},
					},
					{
						AnyOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{
						OneOf: []apiextensions.JSONSchemaProps{
							{Type: "boolean"},
							{Type: "string"},
						},
					},
					{Type: "string"},
				},
				Not: &apiextensions.JSONSchemaProps{
					Not: &apiextensions.JSONSchemaProps{
						Type: "boolean",
					},
					AnyOf: []apiextensions.JSONSchemaProps{
						{Type: "boolean"},
						{Type: "string"},
					},
					OneOf: []apiextensions.JSONSchemaProps{
						{Type: "boolean"},
						{Type: "string"},
					},
				},
			},
			expected: new(spec.Schema).
				WithAllOf(spec.Schema{}, spec.Schema{}, spec.Schema{}, *spec.StringProperty()),
		},
		{
			name: "properties",
			in: &apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					testStr: {Type: "boolean"},
				},
			},
			expected: new(spec.Schema).
				SetProperty(testStr, *spec.BooleanProperty()),
		},
		{
			name: "additionalProperties",
			in: &apiextensions.JSONSchemaProps{
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: &apiextensions.JSONSchemaProps{Type: "boolean"},
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
			in: &apiextensions.JSONSchemaProps{
				PatternProperties: map[string]apiextensions.JSONSchemaProps{
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
			in: &apiextensions.JSONSchemaProps{
				Dependencies: apiextensions.JSONSchemaDependencies{
					testStr: apiextensions.JSONSchemaPropsOrStringArray{
						Schema: &apiextensions.JSONSchemaProps{Type: "boolean"},
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
			in: &apiextensions.JSONSchemaProps{
				Dependencies: apiextensions.JSONSchemaDependencies{
					testStr: apiextensions.JSONSchemaPropsOrStringArray{
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
			in: &apiextensions.JSONSchemaProps{
				AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: &apiextensions.JSONSchemaProps{Type: "boolean"},
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
			in: &apiextensions.JSONSchemaProps{
				Definitions: apiextensions.JSONSchemaDefinitions{
					testStr: apiextensions.JSONSchemaProps{Type: "boolean"},
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
			in: &apiextensions.JSONSchemaProps{
				ExternalDocs: &apiextensions.ExternalDocumentation{
					Description: testStr,
					URL:         testStr2,
				},
			},
			expected: new(spec.Schema).
				WithExternalDocs(testStr, testStr2),
		},
		{
			name: "example",
			in: &apiextensions.JSONSchemaProps{
				Example: &testApiextensionsJSON,
			},
			expected: new(spec.Schema).
				WithExample(testStr),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			out, err := ConvertJSONSchemaPropsToOpenAPIv2Schema(test.in)
			if err != nil {
				t.Fatalf("unexpected error in converting openapi schema: %v", err)
			}
			if !reflect.DeepEqual(*out, *test.expected) {
				t.Errorf("unexpected result:\n  want=%v\n   got=%v\n\n%s", *test.expected, *out, diff.ObjectDiff(*test.expected, *out))
			}
		})
	}
}
