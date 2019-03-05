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
		j = convertNullTypeToNullable(j)
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

func convertNullTypeToNullable(x interface{}) interface{} {
	switch x := x.(type) {
	case map[string]interface{}:
		if t, found := x["type"]; found {
			switch t := t.(type) {
			case []interface{}:
				for i, typ := range t {
					if s, ok := typ.(string); !ok || s != "null" {
						continue
					}
					t = append(t[:i], t[i+1:]...)
					switch len(t) {
					case 0:
						delete(x, "type")
					case 1:
						x["type"] = t[0]
					default:
						x["type"] = t
					}
					x["nullable"] = true
					break
				}
			case string:
				if t == "null" {
					delete(x, "type")
					x["nullable"] = true
				}
			}
		}
		for k := range x {
			x[k] = convertNullTypeToNullable(x[k])
		}
		return x
	case []interface{}:
		for i := range x {
			x[i] = convertNullTypeToNullable(x[i])
		}
		return x
	default:
		return x
	}
}

func TestNullable(t *testing.T) {
	type args struct {
		schema apiextensions.JSONSchemaProps
		object interface{}
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"!nullable against non-null", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: false,
					},
				},
			},
			map[string]interface{}{"field": map[string]interface{}{}},
		}, false},
		{"!nullable against null", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: false,
					},
				},
			},
			map[string]interface{}{"field": nil},
		}, true},
		{"!nullable against undefined", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: false,
					},
				},
			},
			map[string]interface{}{},
		}, false},
		{"nullable against non-null", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: true,
					},
				},
			},
			map[string]interface{}{"field": map[string]interface{}{}},
		}, false},
		{"nullable against null", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: true,
					},
				},
			},
			map[string]interface{}{"field": nil},
		}, false},
		{"!nullable against undefined", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Type:     "object",
						Nullable: true,
					},
				},
			},
			map[string]interface{}{},
		}, false},
		{"nullable and no type against non-nil", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable: true,
					},
				},
			},
			map[string]interface{}{"field": 42},
		}, false},
		{"nullable and no type against nil", args{
			apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"field": {
						Nullable: true,
					},
				},
			},
			map[string]interface{}{"field": nil},
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
