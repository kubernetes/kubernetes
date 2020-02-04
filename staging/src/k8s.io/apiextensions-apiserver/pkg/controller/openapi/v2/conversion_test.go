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

package v2

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/go-openapi/spec"
	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	"github.com/googleapis/gnostic/compiler"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"gopkg.in/yaml.v2"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

func Test_ConvertJSONSchemaPropsToOpenAPIv2Schema(t *testing.T) {
	var spec = []byte(`description: Foo CRD for Testing
type: object
properties:
  spec:
    description: Specification of Foo
    type: object
    properties:
      bars:
        description: List of Bars and their specs.
        type: array
        items:
          type: object
          required:
          - name
          properties:
            name:
              description: Name of Bar.
              type: string
            age:
              description: Age of Bar.
              type: string
            bazs:
              description: List of Bazs.
              items:
                type: string
              type: array
  status:
    description: Status of Foo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object
          properties:
            name:
              description: Name of Bar.
              type: string
            available:
              description: Whether the Bar is installed.
              type: boolean
            quxType:
              description: Indicates to external qux type.
              pattern: in-tree|out-of-tree
              type: string`)

	specV1beta1 := apiextensionsv1beta1.JSONSchemaProps{}
	if err := yaml.Unmarshal(spec, &specV1beta1); err != nil {
		t.Fatal(err)
	}

	specInternal := apiextensions.JSONSchemaProps{}
	if err := apiextensionsv1beta1.Convert_v1beta1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(&specV1beta1, &specInternal, nil); err != nil {
		t.Fatal(err)
	}

	ss, err := structuralschema.NewStructural(&specInternal)
	if err != nil {
		t.Fatal(err)
	}

	ssV2 := ToStructuralOpenAPIV2(ss)
	schema := ssV2.ToGoOpenAPI()

	if _, found := schema.Properties["spec"]; !found {
		t.Errorf("spec not found")
	}
	if _, found := schema.Properties["status"]; !found {
		t.Errorf("status not found")
	}
}

func Test_ConvertJSONSchemaPropsToOpenAPIv2SchemaByType(t *testing.T) {
	testStr := "test"
	testStr2 := "test2"
	testFloat64 := float64(6.4)
	testInt64 := int64(64)
	testApiextensionsJSON := apiextensions.JSON(testStr)

	tests := []struct {
		name        string
		in          *apiextensions.JSONSchemaProps
		expected    *spec.Schema
		expectError bool
		expectDiff  bool
	}{
		{
			name: "id",
			in: &apiextensions.JSONSchemaProps{
				ID: testStr,
			},
			expectError: true, // rejected by kube validation and NewStructural
		},
		{
			name: "$schema",
			in: &apiextensions.JSONSchemaProps{
				Schema: "test",
			},
			expectError: true, // rejected by kube validation and NewStructural
		},
		{
			name: "$ref",
			in: &apiextensions.JSONSchemaProps{
				Ref: &testStr,
			},
			expectError: true, // rejected by kube validation and NewStructural
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
			name: "nullable",
			in: &apiextensions.JSONSchemaProps{
				Type:     "object",
				Nullable: true,
			},
			expected: new(spec.Schema),
		},
		{
			name: "nullable required",
			in: &apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"a": {
						Nullable: true,
						Type:     "string",
					},
					"b": {
						Nullable: true,
						Type:     "string",
					},
					"c": {
						Type: "string",
					},
				},
				Required: []string{"a", "c"},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					Properties: map[string]spec.Schema{
						"a": {},
						"b": {},
						"c": {
							SchemaProps: spec.SchemaProps{
								Type: []string{"string"},
							},
						},
					},
					Required: []string{"c"},
				},
			},
		},
		{
			name: "nullable required additionalProperties",
			in: &apiextensions.JSONSchemaProps{
				Type: "object",
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Schema: &apiextensions.JSONSchemaProps{
						Nullable: true,
						Type:     "string",
					},
				},
				Required: []string{"a", "c"},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					AdditionalProperties: &spec.SchemaOrBool{
						Allows: true,
						Schema: &spec.Schema{},
					},
				},
			},
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
			expectError: true, // rejected by kube validation and NewStructural
		},
		{
			name: "allOf",
			in: &apiextensions.JSONSchemaProps{
				AllOf: []apiextensions.JSONSchemaProps{
					{Type: "boolean"},
					{Type: "string"},
				},
			},
			expected: new(spec.Schema),
			// intentionally not exported in v2
			// expected: new(spec.Schema).
			//   WithAllOf(*spec.BooleanProperty(), *spec.StringProperty()),
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
			// not supported by openapi v2
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
			// not supported by openapi v2
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
			// not supported by openapi v2
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
			expected: new(spec.Schema),
			// not supported by OpenAPI v2 + allOf intentionally not exported
			// expected: new(spec.Schema).
			//	WithAllOf(spec.Schema{}, spec.Schema{}, spec.Schema{}, *spec.StringProperty()),
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
			name: "additionalProperties schema",
			in: &apiextensions.JSONSchemaProps{
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Allows: false,
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
			name: "additionalProperties bool",
			in: &apiextensions.JSONSchemaProps{
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: nil,
				},
			},
			expected: &spec.Schema{
				SchemaProps: spec.SchemaProps{
					AdditionalProperties: &spec.SchemaOrBool{
						Allows: true,
						Schema: nil,
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
			expectError: true, // rejected by kube validation and NewStructural
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
			expectError: true, // rejected by kube validation and NewStructural
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
			expectError: true, // rejected by kube validation and NewStructural
		},
		{
			name: "additionalItems",
			in: &apiextensions.JSONSchemaProps{
				AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{
					Allows: true,
					Schema: &apiextensions.JSONSchemaProps{Type: "boolean"},
				},
			},
			expectError: true, // rejected by kube validation and NewStructural
		},
		{
			name: "definitions",
			in: &apiextensions.JSONSchemaProps{
				Definitions: apiextensions.JSONSchemaDefinitions{
					testStr: apiextensions.JSONSchemaProps{Type: "boolean"},
				},
			},
			expectError: true, // rejected by kube validation and NewStructural
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
			expectDiff: true,
		},
		{
			name: "example",
			in: &apiextensions.JSONSchemaProps{
				Example: &testApiextensionsJSON,
			},
			expected: new(spec.Schema).
				WithExample(testStr),
			expectDiff: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ss, err := structuralschema.NewStructural(test.in)
			if err != nil && !test.expectError {
				t.Fatalf("structural schema error: %v", err)
			} else if err == nil && test.expectError {
				t.Fatalf("expected NewStructural error, but didn't get any")
			}

			if !test.expectError {
				out := ToStructuralOpenAPIV2(ss).ToGoOpenAPI()
				if equal := reflect.DeepEqual(*out, *test.expected); !equal && !test.expectDiff {
					t.Errorf("unexpected result:\n  want=%v\n   got=%v\n\n%s", *test.expected, *out, cmp.Diff(*test.expected, *out, cmp.Comparer(refEqual)))
				} else if equal && test.expectDiff {
					t.Errorf("expected diff, but didn't get any")
				}
			}
		})
	}
}

func refEqual(x spec.Ref, y spec.Ref) bool {
	return x.String() == y.String()
}

// TestKubeOpenapiRejectionFiltering tests that the CRD openapi schema filtering leads to a spec that the
// kube-openapi/pkg/util/proto model code support in version used in Kubernetes 1.13.
func TestKubeOpenapiRejectionFiltering(t *testing.T) {
	for i := 0; i < 10000; i++ {
		f := fuzz.New()
		seed := time.Now().UnixNano()
		randSource := rand.New(rand.NewSource(seed))
		f.RandSource(randSource)
		t.Logf("iteration %d with seed %d", i, seed)

		fuzzFuncs(f, func(ref *spec.Ref, c fuzz.Continue, visible bool) {
			var url string
			if c.RandBool() {
				url = fmt.Sprintf("http://%d", c.Intn(100000))
			} else {
				url = "#/definitions/test"
			}
			r, err := spec.NewRef(url)
			if err != nil {
				t.Fatalf("failed to fuzz ref: %v", err)
			}
			*ref = r
		})

		// create go-openapi object and fuzz it (we start here because we have the powerful fuzzer already
		s := &spec.Schema{}
		f.Fuzz(s)

		// convert to apiextensions v1beta1
		bs, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(string(bs))

		var schema *apiextensionsv1beta1.JSONSchemaProps
		if err := json.Unmarshal(bs, &schema); err != nil {
			t.Fatalf("failed to unmarshal JSON into apiextensions/v1beta1: %v", err)
		}

		// convert to internal
		internalSchema := &apiextensions.JSONSchemaProps{}
		if err := apiextensionsv1beta1.Convert_v1beta1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(schema, internalSchema, nil); err != nil {
			t.Fatalf("failed to convert from apiextensions/v1beta1 to internal: %v", err)
		}

		// apply the filter
		ss, err := structuralschema.NewStructural(internalSchema)
		if err != nil {
			t.Fatal(err)
		}
		filtered := ToStructuralOpenAPIV2(ss).ToGoOpenAPI()

		// create a doc out of it
		filteredSwagger := &spec.Swagger{
			SwaggerProps: spec.SwaggerProps{
				Definitions: spec.Definitions{
					"test": *filtered,
				},
				Info: &spec.Info{
					InfoProps: spec.InfoProps{
						Description: "test",
						Version:     "test",
						Title:       "test",
					},
				},
				Swagger: "2.0",
			},
		}

		// convert to JSON
		bs, err = json.Marshal(filteredSwagger)
		if err != nil {
			t.Fatalf("failed to encode filtered to JSON: %v", err)
		}

		// unmarshal as yaml
		var yml yaml.MapSlice
		if err := yaml.Unmarshal(bs, &yml); err != nil {
			t.Fatalf("failed to decode filtered JSON by into memory: %v", err)
		}

		// create gnostic doc
		doc, err := openapi_v2.NewDocument(yml, compiler.NewContext("$root", nil))
		if err != nil {
			t.Fatalf("failed to create gnostic doc: %v", err)
		}

		// load with kube-openapi/pkg/util/proto
		if _, err := proto.NewOpenAPIData(doc); err != nil {
			t.Fatalf("failed to convert to kube-openapi/pkg/util/proto model: %v", err)
		}
	}
}

// fuzzFuncs is copied from kube-openapi/pkg/aggregator. It fuzzes go-openapi/spec schemata.
func fuzzFuncs(f *fuzz.Fuzzer, refFunc func(ref *spec.Ref, c fuzz.Continue, visible bool)) {
	invisible := 0 // == 0 means visible, > 0 means invisible
	depth := 0
	maxDepth := 3
	nilChance := func(depth int) float64 {
		return math.Pow(0.9, math.Max(0.0, float64(maxDepth-depth)))
	}
	updateFuzzer := func(depth int) {
		f.NilChance(nilChance(depth))
		f.NumElements(0, max(0, maxDepth-depth))
	}
	updateFuzzer(depth)
	enter := func(o interface{}, recursive bool, c fuzz.Continue) {
		if recursive {
			depth++
			updateFuzzer(depth)
		}

		invisible++
		c.FuzzNoCustom(o)
		invisible--
	}
	leave := func(recursive bool) {
		if recursive {
			depth--
			updateFuzzer(depth)
		}
	}
	f.Funcs(
		func(ref *spec.Ref, c fuzz.Continue) {
			refFunc(ref, c, invisible == 0)
		},
		func(sa *spec.SchemaOrStringArray, c fuzz.Continue) {
			*sa = spec.SchemaOrStringArray{}
			if c.RandBool() {
				c.Fuzz(&sa.Schema)
			} else {
				c.Fuzz(&sa.Property)
			}
			if sa.Schema == nil && len(sa.Property) == 0 {
				*sa = spec.SchemaOrStringArray{Schema: &spec.Schema{}}
			}
		},
		func(url *spec.SchemaURL, c fuzz.Continue) {
			*url = spec.SchemaURL("http://url")
		},
		func(s *spec.Dependencies, c fuzz.Continue) {
			enter(s, false, c)
			defer leave(false)

			// and nothing with invisible==false
		},
		func(p *spec.SimpleSchema, c fuzz.Continue) {
			// gofuzz is broken and calls this even for *SimpleSchema fields, ignoring NilChance, leading to infinite recursion
			if c.Float64() > nilChance(depth) {
				return
			}

			enter(p, true, c)
			defer leave(true)

			c.FuzzNoCustom(p)

			// reset JSON fields to some correct JSON
			if p.Default != nil {
				p.Default = "42"
			}
			p.Example = nil
		},
		func(s *spec.SwaggerSchemaProps, c fuzz.Continue) {
			// nothing allowed
			*s = spec.SwaggerSchemaProps{}
		},
		func(s *spec.SchemaProps, c fuzz.Continue) {
			// gofuzz is broken and calls this even for *SchemaProps fields, ignoring NilChance, leading to infinite recursion
			if c.Float64() > nilChance(depth) {
				return
			}

			enter(s, true, c)
			defer leave(true)

			c.FuzzNoCustom(s)

			if c.RandBool() {
				types := []string{"object", "array", "boolean", "string", "integer", "number"}
				s.Type = []string{types[c.Intn(len(types))]}
			} else {
				s.Type = nil
			}

			s.ID = ""
			s.Ref = spec.Ref{}
			s.AdditionalItems = nil
			s.Dependencies = nil
			s.Schema = ""
			s.PatternProperties = nil
			s.Definitions = nil

			if len(s.Type) == 1 && s.Type[0] == "array" {
				s.Items = &spec.SchemaOrArray{Schema: &spec.Schema{}}
				c.Fuzz(s.Items.Schema)
			} else {
				s.Items = nil
			}

			// reset JSON fields to some correct JSON
			if s.Default != nil {
				s.Default = "42"
			}
			for i := range s.Enum {
				s.Enum[i] = "42"
			}
		},
		func(i *interface{}, c fuzz.Continue) {
			// do nothing for examples and defaults. These are free form JSON fields.
		},
	)
}

func TestFilterOut(t *testing.T) {
	type Test struct {
		name            string
		input           []string
		x               string
		expected        []string
		expectedChanged bool
	}
	for _, tt := range []Test{
		{"nil", nil, "foo", nil, false},
		{"empty", []string{}, "foo", []string{}, false},
		{"foo", []string{"foo"}, "foo", nil, true},
		{"aaa", []string{"a", "a", "a"}, "a", nil, true},
		{"abc", []string{"a", "b", "c"}, "c", []string{"a", "b"}, true},
		{"abbbcc", []string{"a", "b", "b", "b", "c", "c"}, "b", []string{"a", "c", "c"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, gotChanged := filterOut(tt.input, tt.x)
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("expected slice %v, got %v", tt.expected, got)
			}
			if tt.expectedChanged != gotChanged {
				t.Errorf("expected changed %v, got %v", tt.expected, got)
			}
		})
	}
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}
