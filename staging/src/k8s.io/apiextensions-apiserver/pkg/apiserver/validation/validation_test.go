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
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"
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
	if err := apiextensionsv1beta1.AddToScheme(scheme); err != nil {
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
		external := &apiextensionsv1beta1.JSONSchemaProps{}
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

func TestValidateCustomResource(t *testing.T) {
	tests := []struct {
		name           string
		schema         apiextensions.JSONSchemaProps
		objects        []interface{}
		failingObjects []interface{}
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": "foo"},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": "foo"},
				map[string]interface{}{"field": 42},
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": nil},
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{
				map[string]interface{}{"field": true},
				map[string]interface{}{"field": 1.2},
				map[string]interface{}{"field": map[string]interface{}{}},
				map[string]interface{}{"field": []interface{}{}},
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
			failingObjects: []interface{}{map[string]interface{}{"field": "foo"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator, _, err := NewSchemaValidator(&apiextensions.CustomResourceValidation{OpenAPIV3Schema: &tt.schema})
			if err != nil {
				t.Fatal(err)
			}
			for _, obj := range tt.objects {
				if err := ValidateCustomResource(obj, validator); err != nil {
					t.Errorf("unexpected validation error for %v: %v", obj, err)
				}
			}
			for _, obj := range tt.failingObjects {
				if err := ValidateCustomResource(obj, validator); err == nil {
					t.Errorf("missing error for %v", obj)
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
			if err := ValidateCustomResource(tt.args.object, validator); (err != nil) != tt.wantErr {
				if err == nil {
					t.Error("expected error, but didn't get one")
				} else {
					t.Errorf("unexpected validation error: %v", err)
				}
			}
		})
	}
}
