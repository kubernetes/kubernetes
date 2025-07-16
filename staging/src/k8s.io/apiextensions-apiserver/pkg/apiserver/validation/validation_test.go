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
	"context"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	utilpointer "k8s.io/utils/pointer"
	kjson "sigs.k8s.io/json"

	kubeopenapispec "k8s.io/kube-openapi/pkg/validation/spec"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
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

	seed := int64(time.Now().Nanosecond())
	if override := os.Getenv("TEST_RAND_SEED"); len(override) > 0 {
		overrideSeed, err := strconv.Atoi(override)
		if err != nil {
			t.Fatal(err)
		}
		seed = int64(overrideSeed)
		t.Logf("using overridden seed: %d", seed)
	} else {
		t.Logf("seed (override with TEST_RAND_SEED if desired): %d", seed)
	}
	fuzzerFuncs := fuzzer.MergeFuzzerFuncs(apiextensionsfuzzer.Funcs)
	f := fuzzer.FuzzerFor(fuzzerFuncs, rand.NewSource(seed), codecs)

	for i := 0; i < 50; i++ {
		// fuzz internal types
		internal := &apiextensions.JSONSchemaProps{}
		f.Fill(internal)

		// internal -> go-openapi
		openAPITypes := &kubeopenapispec.Schema{}
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
		if strictErrs, err := kjson.UnmarshalStrict(openAPIJSON, &j); err != nil {
			t.Fatal(err)
		} else if len(strictErrs) > 0 {
			t.Fatal(strictErrs)
		}
		j = stripIntOrStringType(j)
		openAPIJSON, err = json.Marshal(j)
		if err != nil {
			t.Fatal(err)
		}

		// JSON -> external
		external := &apiextensionsv1.JSONSchemaProps{}
		if strictErrs, err := kjson.UnmarshalStrict(openAPIJSON, external); err != nil {
			t.Fatal(err)
		} else if len(strictErrs) > 0 {
			t.Fatal(strictErrs)
		}

		// external -> internal
		internalRoundTripped := &apiextensions.JSONSchemaProps{}
		if err := scheme.Convert(external, internalRoundTripped, nil); err != nil {
			t.Fatal(err)
		}

		if !apiequality.Semantic.DeepEqual(internal, internalRoundTripped) {
			t.Log(string(openAPIJSON))
			t.Fatalf("%d: unexpected diff\n\t%s", i, cmp.Diff(internal, internalRoundTripped))
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
	oldObject  interface{}
	expectErrs []string
}

func TestValidateCustomResource(t *testing.T) {
	tests := []struct {
		name           string
		schema         apiextensions.JSONSchemaProps
		objects        []interface{}
		oldObjects     []interface{}
		failingObjects []failingObject
	}{
		{name: "!nullable",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
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
				Type: "object",
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
				Type: "object",
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
				Type: "object",
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
				Type: "object",
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
				Type: "object",
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
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`,
					`field: Invalid value: "boolean": field in body must be of type integer: "boolean"`,
				}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "number": field in body must be of type integer,string: "number"`,
					`field: Invalid value: "number": field in body must be of type integer: "number"`,
				}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "object": field in body must be of type integer,string: "object"`,
					`field: Invalid value: "object": field in body must be of type integer: "object"`,
				}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "array": field in body must be of type integer,string: "array"`,
					`field: Invalid value: "array": field in body must be of type integer: "array"`,
				}},
			},
		},
		{name: "nullable, x-kubernetes-int-or-string and user-provider allOf",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
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
					`<nil>: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "boolean": field in body must be of type integer,string: "boolean"`,
					`field: Invalid value: "boolean": field in body must be of type integer: "boolean"`,
				}},
				{object: map[string]interface{}{"field": 1.2}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "number": field in body must be of type integer,string: "number"`,
					`field: Invalid value: "number": field in body must be of type integer: "number"`,
				}},
				{object: map[string]interface{}{"field": map[string]interface{}{}}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "object": field in body must be of type integer,string: "object"`,
					`field: Invalid value: "object": field in body must be of type integer: "object"`,
				}},
				{object: map[string]interface{}{"field": []interface{}{}}, expectErrs: []string{
					`<nil>: Invalid value: "": "field" must validate all the schemas (allOf). None validated`,
					`<nil>: Invalid value: "": "field" must validate at least one schema (anyOf)`,
					`field: Invalid value: "array": field in body must be of type integer,string: "array"`,
					`field: Invalid value: "array": field in body must be of type integer: "array"`,
				}},
			},
		},
		{name: "invalid regex",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:    "string",
						Pattern: "+",
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"field": "foo"}, expectErrs: []string{"field: Invalid value: \"foo\": field in body should match '+, but pattern is invalid: error parsing regexp: missing argument to repetition operator: `+`'"}},
			},
		},
		{name: "required field",
			schema: apiextensions.JSONSchemaProps{
				Type:     "object",
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
				Type: "object",
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
		{name: "immutability transition rule",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type: "string",
						XValidations: []apiextensions.ValidationRule{
							{
								Rule: "self == oldSelf",
							},
						},
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{"field": "x"},
			},
			oldObjects: []interface{}{
				map[string]interface{}{"field": "x"},
			},
			failingObjects: []failingObject{
				{
					object:    map[string]interface{}{"field": "y"},
					oldObject: map[string]interface{}{"field": "x"},
					expectErrs: []string{
						`field: Invalid value: "string": failed rule: self == oldSelf`,
					}},
			},
		},
		{name: "correlatable transition rule",
			// Ensures a transition rule under a "listMap" is supported.
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:         "array",
						XListType:    &listMapType,
						XListMapKeys: []string{"k1", "k2"},
						Items: &apiextensions.JSONSchemaPropsOrArray{
							Schema: &apiextensions.JSONSchemaProps{
								Type: "object",
								Properties: map[string]apiextensions.JSONSchemaProps{
									"k1": {
										Type: "string",
									},
									"k2": {
										Type: "string",
									},
									"v1": {
										Type: "number",
										XValidations: []apiextensions.ValidationRule{
											{
												Rule: "self >= oldSelf",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{"field": []interface{}{map[string]interface{}{"k1": "a", "k2": "b", "v1": 1.2}}},
			},
			oldObjects: []interface{}{
				map[string]interface{}{"field": []interface{}{map[string]interface{}{"k1": "a", "k2": "b", "v1": 1.0}}},
			},
			failingObjects: []failingObject{
				{
					object:    map[string]interface{}{"field": []interface{}{map[string]interface{}{"k1": "a", "k2": "b", "v1": 0.9}}},
					oldObject: map[string]interface{}{"field": []interface{}{map[string]interface{}{"k1": "a", "k2": "b", "v1": 1.0}}},
					expectErrs: []string{
						`field[0].v1: Invalid value: "number": failed rule: self >= oldSelf`,
					}},
			},
		},
		{name: "validation rule under non-correlatable field",
			// The array makes the rule on the nested string non-correlatable
			// for transition rule purposes. This test ensures that a rule that
			// does NOT use oldSelf (is not a transition rule), still behaves
			// as expected under a non-correlatable field.
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type: "array",
						Items: &apiextensions.JSONSchemaPropsOrArray{
							Schema: &apiextensions.JSONSchemaProps{
								Type: "object",
								Properties: map[string]apiextensions.JSONSchemaProps{
									"x": {
										Type: "string",
										XValidations: []apiextensions.ValidationRule{
											{
												Rule: "self == 'x'",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			objects: []interface{}{
				map[string]interface{}{"field": []interface{}{map[string]interface{}{"x": "x"}}},
			},
			failingObjects: []failingObject{
				{
					object: map[string]interface{}{"field": []interface{}{map[string]interface{}{"x": "y"}}},
					expectErrs: []string{
						`field[0].x: Invalid value: "string": failed rule: self == 'x'`,
					}},
			},
		},
		{name: "maxProperties",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"fieldX": {
						Type:          "object",
						MaxProperties: utilpointer.Int64(2),
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"fieldX": map[string]interface{}{"a": true, "b": true, "c": true}}, expectErrs: []string{
					`fieldX: Too many: 3: must have at most 2 items`,
				}},
			},
		},
		{name: "maxItems",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"fieldX": {
						Type:     "array",
						MaxItems: utilpointer.Int64(2),
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"fieldX": []interface{}{"a", "b", "c"}}, expectErrs: []string{
					`fieldX: Too many: 3: must have at most 2 items`,
				}},
			},
		},
		{name: "maxLength",
			schema: apiextensions.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"fieldX": {
						Type:      "string",
						MaxLength: utilpointer.Int64(2),
					},
				},
			},
			failingObjects: []failingObject{
				{object: map[string]interface{}{"fieldX": "abc"}, expectErrs: []string{
					`fieldX: Too long: may not be more than 2 bytes`,
				}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator, _, err := NewSchemaValidator(&tt.schema)
			if err != nil {
				t.Fatal(err)
			}
			structural, err := structuralschema.NewStructural(&tt.schema)
			if err != nil {
				t.Fatal(err)
			}
			celValidator := cel.NewValidator(structural, false, celconfig.PerCallLimit)
			for i, obj := range tt.objects {
				var oldObject interface{}
				if len(tt.oldObjects) == len(tt.objects) {
					oldObject = tt.oldObjects[i]
				}
				if errs := ValidateCustomResource(nil, obj, validator); len(errs) > 0 {
					t.Errorf("unexpected validation error for %v: %v", obj, errs)
				}
				errs, _ := celValidator.Validate(context.TODO(), nil, structural, obj, oldObject, celconfig.RuntimeCELCostBudget)
				if len(errs) > 0 {
					t.Error(errs.ToAggregate().Error())
				}
			}
			for i, failingObject := range tt.failingObjects {
				errs := ValidateCustomResource(nil, failingObject.object, validator)
				celErrs, _ := celValidator.Validate(context.TODO(), nil, structural, failingObject.object, failingObject.oldObject, celconfig.RuntimeCELCostBudget)
				errs = append(errs, celErrs...)
				if len(errs) == 0 {
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
			validator, _, err := NewSchemaValidator(&tt.args.schema)
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

var listMapType = "map"
