/*
Copyright 2020 The Kubernetes Authors.

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

package handler_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kube-openapi/pkg/handler"

	"github.com/go-openapi/spec"
)

func TestDefaultPruning(t *testing.T) {
	def := spec.Definitions{
		"foo": spec.Schema{
			SchemaProps: spec.SchemaProps{
				Default: 0,
				AllOf:   []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}}},
				AnyOf:   []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}}},
				OneOf:   []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}}},
				Not:     &spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}},
				Properties: map[string]spec.Schema{
					"foo": spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}},
				},
				AdditionalProperties: &spec.SchemaOrBool{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}}},
				PatternProperties: map[string]spec.Schema{
					"foo": spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}},
				},
				Dependencies: spec.Dependencies{
					"foo": spec.SchemaOrStringArray{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}}},
				},
				AdditionalItems: &spec.SchemaOrBool{
					Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}},
				},
				Definitions: spec.Definitions{
					"bar": spec.Schema{SchemaProps: spec.SchemaProps{Default: "default-string", Title: "Field"}},
				},
			},
		},
	}
	jsonDef, err := json.Marshal(def)
	if err != nil {
		t.Fatalf("Failed to marshal definition: %v", err)
	}
	wanted := spec.Definitions{
		"foo": spec.Schema{
			SchemaProps: spec.SchemaProps{
				AllOf: []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}}},
				AnyOf: []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}}},
				OneOf: []spec.Schema{spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}}},
				Not:   &spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}},
				Properties: map[string]spec.Schema{
					"foo": spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}},
				},
				AdditionalProperties: &spec.SchemaOrBool{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}}},
				PatternProperties: map[string]spec.Schema{
					"foo": spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}},
				},
				Dependencies: spec.Dependencies{
					"foo": spec.SchemaOrStringArray{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}}},
				},
				AdditionalItems: &spec.SchemaOrBool{
					Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}},
				},
				Definitions: spec.Definitions{
					"bar": spec.Schema{SchemaProps: spec.SchemaProps{Title: "Field"}},
				},
			},
		},
	}

	got := handler.PruneDefaults(def)
	if !reflect.DeepEqual(got, wanted) {
		gotJSON, _ := json.Marshal(got)
		wantedJSON, _ := json.Marshal(wanted)
		t.Fatalf("got: %v\nwanted %v", string(gotJSON), string(wantedJSON))
	}
	// Make sure that def hasn't been changed.
	newDef, _ := json.Marshal(def)
	if string(newDef) != string(jsonDef) {
		t.Fatalf("prune removed defaults from initial config:\nBefore: %v\nAfter: %v", string(jsonDef), string(newDef))
	}
	// Make sure that no-op doesn't change the object.
	if reflect.ValueOf(handler.PruneDefaults(got)).Pointer() != reflect.ValueOf(got).Pointer() {
		t.Fatal("no-op prune returned new object")
	}
}
