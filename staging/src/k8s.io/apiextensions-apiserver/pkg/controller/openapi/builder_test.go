/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestNewBuilder(t *testing.T) {
	type args struct {
	}
	tests := []struct {
		name string

		schema string

		wantedSchema      string
		wantedItemsSchema string
	}{
		{
			"nil",
			"",
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`, `{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
		},
		{"empty",
			"{}",
			`{"properties":{"apiVersion":{},"kind":{},"metadata":{}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
		},
		{"empty properties",
			`{"properties":{"spec":{},"status":{}}}`,
			`{"properties":{"apiVersion":{},"kind":{},"metadata":{},"spec":{},"status":{}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
		},
		{"filled properties",
			`{"properties":{"spec":{"type":"object"},"status":{"type":"object"}}}`,
			`{"properties":{"apiVersion":{},"kind":{},"metadata":{},"spec":{"type":"object"},"status":{"type":"object"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
		},
		{"type",
			`{"type":"object"}`,
			`{"properties":{"apiVersion":{},"kind":{},"metadata":{}},"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var schema *spec.Schema
			if len(tt.schema) > 0 {
				schema = &spec.Schema{}
				if err := json.Unmarshal([]byte(tt.schema), schema); err != nil {
					t.Fatal(err)
				}
			}

			got := newBuilder(&apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "bar.k8s.io",
					Version: "v1",
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "foos",
						Singular: "foo",
						Kind:     "Foo",
						ListKind: "FooList",
					},
					Scope: apiextensions.NamespaceScoped,
				},
			}, "v1", schema)

			var wantedSchema, wantedItemsSchema spec.Schema
			if err := json.Unmarshal([]byte(tt.wantedSchema), &wantedSchema); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tt.wantedItemsSchema), &wantedItemsSchema); err != nil {
				t.Fatal(err)
			}

			gotProperties := properties(got.schema.Properties)
			wantedProperties := properties(wantedSchema.Properties)
			if !gotProperties.Equal(wantedProperties) {
				t.Fatalf("unexpected properties, got: %s, expected: %s", gotProperties.List(), wantedProperties.List())
			}

			// wipe out TypeMeta/ObjectMeta content, with those many lines of descriptions. We trust that they match here.
			if _, found := got.schema.Properties["kind"]; found {
				got.schema.Properties["kind"] = spec.Schema{}
			}
			if _, found := got.schema.Properties["apiVersion"]; found {
				got.schema.Properties["apiVersion"] = spec.Schema{}
			}
			if _, found := got.schema.Properties["metadata"]; found {
				got.schema.Properties["metadata"] = spec.Schema{}
			}

			if !reflect.DeepEqual(&wantedSchema, got.schema) {
				t.Errorf("unexpected schema: %s\nwant = %#v\ngot = %#v", diff.ObjectDiff(&wantedSchema, got.schema), &wantedSchema, got.schema)
			}

			gotListProperties := properties(got.listSchema.Properties)
			if want := sets.NewString("apiVersion", "kind", "metadata", "items"); !gotListProperties.Equal(want) {
				t.Fatalf("unexpected list properties, got: %s, expected: %s", gotListProperties.List(), want.List())
			}

			gotListSchema := got.listSchema.Properties["items"].Items.Schema
			if !reflect.DeepEqual(&wantedItemsSchema, gotListSchema) {
				t.Errorf("unexpected list schema: %s (want/got)", diff.ObjectDiff(&wantedItemsSchema, &gotListSchema))
			}
		})
	}
}

func properties(p map[string]spec.Schema) sets.String {
	ret := sets.NewString()
	for k := range p {
		ret.Insert(k)
	}
	return ret
}
