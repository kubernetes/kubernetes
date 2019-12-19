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
	"math/rand"
	"testing"

	"github.com/go-openapi/spec"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestRoundTrip checks the conversion to go-openapi types.
// internal -> go-openapi -> JSON -> external -> internal
func TestRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	// add internal and external types to scheme
	if err := apiextensions.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := apiextensionsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	seed := rand.Int63()
	t.Logf("seed: %d", seed)
	fuzzerFuncs := fuzzer.MergeFuzzerFuncs(apiextensionsfuzzer.Funcs)
	f := fuzzer.FuzzerFor(fuzzerFuncs, rand.NewSource(seed), codecs)

	for i := 0; i < 20; i++ {
		// fuzz internal types
		internal := &apiextensions.JSONSchemaProps{}
		f.Fuzz(internal)

		// internal -> go-openapi
		openAPITypes := &spec.Schema{}
		if err := ConvertJSONSchemaProps(internal, openAPITypes); err != nil {
			t.Fatal(err)
		}

		// go-openapi -> JSON
		openAPIJSON, err := json.Marshal(openAPITypes)
		if err != nil {
			t.Fatal(err)
		}

		// JSON -> in-memory JSON => convertNullTypeToNullable => JSON
		var j interface{}
		if err := json.Unmarshal(openAPIJSON, &j); err != nil {
			t.Fatal(err)
		}
		j = stripIntOrStringType(j)
		openAPIJSON, err = json.Marshal(j)
		if err != nil {
			t.Fatal(err)
		}

		// JSON -> external
		external := &apiextensionsv1.JSONSchemaProps{}
		if err := json.Unmarshal(openAPIJSON, external); err != nil {
			t.Fatal(err)
		}

		// external -> internal
		internalRoundTripped := &apiextensions.JSONSchemaProps{}
		if err := scheme.Convert(external, internalRoundTripped, nil); err != nil {
			t.Fatal(err)
		}

		if !apiequality.Semantic.DeepEqual(internal, internalRoundTripped) {
			t.Fatalf("%d: expected\n\t%#v, got \n\t%#v", i, internal, internalRoundTripped)
		}
	}
}

func stripIntOrStringType(x interface{}) interface{} {
	switch x := x.(type) {
	case map[string]interface{}:
		if t, found := x["type"]; found {
			switch t := t.(type) {
			case []interface{}:
				if len(t) == 2 && t[0] == "integer" && t[1] == "string" && x["x-kubernetes-int-or-string"] == true {
					delete(x, "type")
				}
			}
		}
		for k := range x {
			x[k] = stripIntOrStringType(x[k])
		}
		return x
	case []interface{}:
		for i := range x {
			x[i] = stripIntOrStringType(x[i])
		}
		return x
	default:
		return x
	}
}

type failingObject struct {
	object     interface{}
	expectErrs []string
}

func TestValidateCustomResource(t *testing.T) {
	tests := []struct {
		name           string
		schema         apiextensions.JSONSchemaProps
		objects        []interface{}
		failingObjects []failingObject
	}{
		{name: "!nullable",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: false,
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": map[string]interface{}{}},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": "foo"}, expectErrs: []string{`field: Invalid value: "string": field in body must be of type object: "string"`}},
				{object: map[string]interface{}{"field": 42}, expectErrs: []string{`field: Invalid value: "integer": field in body must be of type object: "integer"`}},
				{object: map[string]interface{}{"field": true}, expectErrs: []string{`field: Invalid value: "boolean": field in body must be of type object: "boolean"`}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{`field: Invalid value: "number": field in body must be of type object: "number"`}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{`field: Invalid value: "array": field in body must be of type object: "array"`}},
				{object: map[string]interface{}{"field": nil}, expectErrs: []string{`field: Invalid value: "null": field in body must be of type object: "null"`}},
			},
		},
		{name: "nullable",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: true,
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": nil},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": "foo"}, expectErrs: []string{`field: Invalid value: "string": field in body must be of type object: "string"`}},
				{object: map[string]interface{}{"field": 42}, expectErrs: []string{`field: Invalid value: "integer": field in body must be of type object: "integer"`}},
				{object: map[string]interface{}{"field": true}, expectErrs: []string{`field: Invalid value: "boolean": field in body must be of type object: "boolean"`}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{`field: Invalid value: "number": field in body must be of type object: "number"`}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{`field: Invalid value: "array": field in body must be of type object: "array"`}},
			},
		},
		{name: "nullable and no type",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable: true,
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": nil},
				map[string]interface{}{"field": "foo"},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": []interface{}{}},
			},
		},
		{name: "x-kubernetes-int-or-string",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						XIntOrString: true,
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": "foo"},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": nil}, expectErrs: []string{`field: Invalid value: "null": field in body must be of type integer,string: "null"`}},
				{object: map[string]interface{}{"field": true}, expectErrs: []string{`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{`field: Invalid value: "number": field in body must be of type integer,string: "number"`}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{`field: Invalid value: "object": field in body must be of type integer,string: "object"`}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{`field: Invalid value: "array": field in body must be of type integer,string: "array"`}},
			},
		},
		{name: "nullable and x-kubernetes-int-or-string",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable:     true,
						XIntOrString: true,
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": "foo"},
				map[string]interface{}{"field": nil},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": true}, expectErrs: []string{`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{`field: Invalid value: "number": field in body must be of type integer,string: "number"`}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{`field: Invalid value: "object": field in body must be of type integer,string: "object"`}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{`field: Invalid value: "array": field in body must be of type integer,string: "array"`}},
			},
		},
		{name: "nullable, x-kubernetes-int-or-string and user-provided anyOf",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable:     true,
						XIntOrString: true,
						AnyOf: []apiextensions.JSONSchemaProps{
							{Type: "integer"},
							{Type: "string"},
						},
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": nil},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": "foo"},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": true}, expectErrs: []string{
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`,
					`field: Invalid value: "boolean": field in body must be of type integer: "boolean"`,
				}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "number": field in body must be of type integer,string: "number"`,
					`field: Invalid value: "number": field in body must be of type integer: "number"`,
				}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "object": field in body must be of type integer,string: "object"`,
					`field: Invalid value: "object": field in body must be of type integer: "object"`,
				}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "array": field in body must be of type integer,string: "array"`,
					`field: Invalid value: "array": field in body must be of type integer: "array"`,
				}},
			},
		},
		{name: "nullable, x-kubernetes-int-or-string and user-provider allOf",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable:     true,
						XIntOrString: true,
						AllOf: []apiextensions.JSONSchemaProps{
							{
								AnyOf: []apiextensions.JSONSchemaProps{
									{Type: "integer"},
									{Type: "string"},
								},
							},
						},
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{},
				map[string]interface{}{"field": nil},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": "foo"},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": true}, expectErrs: []string{
					`: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`,
					`field: Invalid value: "boolean": field in body must be of type integer: "boolean"`,
				}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{
					`: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "number": field in body must be of type integer,string: "number"`,
					`field: Invalid value: "number": field in body must be of type integer: "number"`,
				}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{
					`: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "object": field in body must be of type integer,string: "object"`,
					`field: Invalid value: "object": field in body must be of type integer: "object"`,
				}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{
					`: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "array": field in body must be of type integer,string: "array"`,
					`field: Invalid value: "array": field in body must be of type integer: "array"`,
				}},
			},
		},
		{name: "invalid regex",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:    "string",
						Pattern: "+",
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": "foo"}, expectErrs: []string{"field: Invalid value: \"\": field in body should match '+, but pattern is invalid: error parsing regexp: missing argument to repetition operator: `+`'"}},
			},
		},
		{name: "required field",
			schema: apiextensions.JSONSchemaProps{
				Required: []string{"field"},
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Required: []string{"nested"},
						Properties: map[string]apiextensions.JSONSchemaProps{
							"nested": {},
						},
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"test": "a"}, expectErrs: []string{`field: Required value`}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{`field.nested: Required value`}},
			},
		},
		{name: "enum",
			schema: apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Required: []string{"nestedint", "nestedstring"},
						Properties: map[string]apiextensions.JSONSchemaProps{
							"nestedint": {
								Type: "integer",
								Enum: []apiextensions.JSON{1, 2},
							},
							"nestedstring": {
								Type: "string",
								Enum: []apiextensions.JSON{"a", "b"},
							},
						},
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{
					`field.nestedint: Required value`,
					`field.nestedstring: Required value`,
				}},
				{object: map[string]interface{}{"field": map[string]interface{}{"nestedint": "x", "nestedstring": true}}, expectErrs: []string{
					`field.nestedint: Invalid value: "string": field.nestedint in body must be of type integer: "string"`,
					`field.nestedint: Unsupported value: "x": supported values: "1", "2"`,
					`field.nestedstring: Invalid value: "boolean": field.nestedstring in body must be of type string: "boolean"`,
					`field.nestedstring: Unsupported value: true: supported values: "a", "b"`,
				}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator, _, err := NewSchemaValidator(&apiextensions.CustomResourceValidation{OpenAPIV3Schema: &tt.schema})
			if err != nil {
				t.Fatal(err)
			}
			for _, obj := range tt.objects {
				if errs := ValidateCustomResource(nil, obj, validator); len(errs) > 0 {
					t.Errorf("unexpected validation error for %v: %v", obj, errs)
				}
			}
			for i, failingObject := range tt.failingObjects {
				if errs := ValidateCustomResource(nil, failingObject.object, validator); len(errs) == 0 {
					t.Errorf("missing error for %v", failingObject.object)
				} else {
					sawErrors := sets.NewString()
					for _, err := range errs {
						sawErrors.Insert(err.Error())
					}
					expectErrs := sets.NewString(failingObject.expectErrs...)
					for _, unexpectedError := range sawErrors.Difference(expectErrs).List() {
						t.Errorf("%d: unexpected error: %s", i, unexpectedError)
					}
					for _, missingError := range expectErrs.Difference(sawErrors).List() {
						t.Errorf("%d: missing error:    %s", i, missingError)
					}
				}
			}
		})
	}
}

func TestItemsProperty(t *testing.T) {
	type args struct {
		schema apiextensions.JSONSchemaProps
		object interface{}
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"items in object", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"spec": {
						Properties: map[string]apiextensions.JSONSchemaProps{
							"replicas": {
								Type: "integer",
							},
						},
					},
				},
			},
			map[string]interface{}{"spec": map[string]interface{}{"replicas": 1, "items": []string{"1", "2"}}},
		}, false},
		{"items in array", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"secrets": {
						Type: "array",
						Items: &apiextensions.JSONSchemaPropsOrArray{
							Schema: &apiextensions.JSONSchemaProps{
								Type: "string",
							},
						},
					},
				},
			},
			map[string]interface{}{"secrets": []string{"1", "2"}},
		}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator, _, err := NewSchemaValidator(&apiextensions.CustomResourceValidation{OpenAPIV3Schema: &tt.args.schema})
			if err != nil {
				t.Fatal(err)
			}
			if errs := ValidateCustomResource(nil, tt.args.object, validator); (len(errs) > 0) != tt.wantErr {
				if len(errs) == 0 {
					t.Error("expected error, but didn't get one")
				} else {
					t.Errorf("unexpected validation error: %v", errs)
				}
			}
		})
	}
}
