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

package builder

import (
	"reflect"
	"testing"

	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apiextensionsinternal "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilpointer "k8s.io/utils/pointer"
)

func TestNewBuilder(t *testing.T) {
	tests := []struct {
		name string

		schema string

		wantedSchema      string
		wantedItemsSchema string

		v2 bool // produce OpenAPIv2
	}{
		{
			"nil",
			"",
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`, `{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			true,
		},
		{"with properties",
			`{"type":"object","properties":{"spec":{"type":"object"},"status":{"type":"object"}}}`,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"spec":{"type":"object"},"status":{"type":"object"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			true,
		},
		{"type only",
			`{"type":"object"}`,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			true,
		},
		{"preserve unknown at root v2",
			`{"type":"object","x-kubernetes-preserve-unknown-fields":true}`,
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			true,
		},
		{"preserve unknown at root v3",
			`{"type":"object","x-kubernetes-preserve-unknown-fields":true}`,
			`{"type":"object","x-kubernetes-preserve-unknown-fields":true,"properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			false,
		},
		{"with extensions",
			`
{
  "type":"object", 
  "properties": {
    "int-or-string-1": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ]
    },
    "int-or-string-2": {
      "x-kubernetes-int-or-string": true,
      "allOf": [{
        "anyOf": [
          {"type":"integer"},
          {"type":"string"}
        ]
      }, {
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-3": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ],
      "allOf": [{
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-4": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"minimum": 42.0}
      ]
    },
    "int-or-string-5": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"minimum": 42.0}
      ],
      "allOf": [
        {"minimum": 42.0}
      ]
    },
    "int-or-string-6": {
      "x-kubernetes-int-or-string": true
    },
    "preserve-unknown-fields": {
      "x-kubernetes-preserve-unknown-fields": true
    },
    "embedded-object": {
      "x-kubernetes-embedded-resource": true,
      "x-kubernetes-preserve-unknown-fields": true,
      "type": "object"
    }
  }
}`,
			`
{
  "type":"object",
  "properties": {
    "apiVersion": {"type":"string"},
    "kind": {"type":"string"},
    "metadata": {"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},
    "int-or-string-1": {
      "x-kubernetes-int-or-string": true
    },
    "int-or-string-2": {
      "x-kubernetes-int-or-string": true
    },
    "int-or-string-3": {
      "x-kubernetes-int-or-string": true
    },
    "int-or-string-4": {
      "x-kubernetes-int-or-string": true
    },
    "int-or-string-5": {
      "x-kubernetes-int-or-string": true
    },
    "int-or-string-6": {
      "x-kubernetes-int-or-string": true
    },
    "preserve-unknown-fields": {
      "x-kubernetes-preserve-unknown-fields": true
    },
    "embedded-object": {
      "x-kubernetes-embedded-resource": true,
      "x-kubernetes-preserve-unknown-fields": true,
      "type":"object"
    }
  },
  "x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]
}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			true,
		},
		{"with extensions as v3 schema",
			`
{
  "type":"object", 
  "properties": {
    "int-or-string-1": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ]
    },
    "int-or-string-2": {
      "x-kubernetes-int-or-string": true,
      "allOf": [{
        "anyOf": [
          {"type":"integer"},
          {"type":"string"}
        ]
      }, {
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-3": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ],
      "allOf": [{
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-4": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"minimum": 42.0}
      ]
    },
    "int-or-string-5": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"minimum": 42.0}
      ],
      "allOf": [
        {"minimum": 42.0}
      ]
    },
    "int-or-string-6": {
      "x-kubernetes-int-or-string": true
    },
    "preserve-unknown-fields": {
      "x-kubernetes-preserve-unknown-fields": true
    },
    "embedded-object": {
      "x-kubernetes-embedded-resource": true,
      "x-kubernetes-preserve-unknown-fields": true,
      "type": "object"
    }
  }
}`,
			`
{
  "type":"object",
  "properties": {
    "apiVersion": {"type":"string"},
    "kind": {"type":"string"},
    "metadata": {"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},
    "int-or-string-1": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ]
    },
    "int-or-string-2": {
      "x-kubernetes-int-or-string": true,
      "allOf": [{
        "anyOf": [
          {"type":"integer"},
          {"type":"string"}
        ]
      }, {
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-3": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ],
      "allOf": [{
        "anyOf": [
          {"minimum": 42.0}
        ]
      }]
    },
    "int-or-string-4": {
      "x-kubernetes-int-or-string": true,
      "allOf": [{
        "anyOf": [
          {"type":"integer"},
          {"type":"string"}
        ]
      }],
      "anyOf": [
        {"minimum": 42.0}
      ]
    },
    "int-or-string-5": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"minimum": 42.0}
      ],
      "allOf": [{
        "anyOf": [
          {"type":"integer"},
          {"type":"string"}
        ]
      }, {
        "minimum": 42.0
      }]
    },
    "int-or-string-6": {
      "x-kubernetes-int-or-string": true,
      "anyOf": [
        {"type":"integer"},
        {"type":"string"}
      ]
    },
    "preserve-unknown-fields": {
      "x-kubernetes-preserve-unknown-fields": true
    },
    "embedded-object": {
      "x-kubernetes-embedded-resource": true,
      "x-kubernetes-preserve-unknown-fields": true,
      "type": "object",
      "required":["kind","apiVersion"],
      "properties":{
        "apiVersion":{
          "description":"apiVersion defines the versioned schema of this representation of an object. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources",
          "type":"string"
        },
        "kind":{
          "description":"kind is a string value representing the type of this object. In CamelCase. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds",
          "type":"string"
        },
        "metadata":{
          "description":"Standard object's metadata. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata",
          "$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"
        }
      }
    }
  },
  "x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]
}`,
			`{"$ref":"#/definitions/io.k8s.bar.v1.Foo"}`,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var schema *structuralschema.Structural
			if len(tt.schema) > 0 {
				v1beta1Schema := &apiextensionsv1.JSONSchemaProps{}
				if err := json.Unmarshal([]byte(tt.schema), &v1beta1Schema); err != nil {
					t.Fatal(err)
				}
				internalSchema := &apiextensionsinternal.JSONSchemaProps{}
				apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v1beta1Schema, internalSchema, nil)
				var err error
				schema, err = structuralschema.NewStructural(internalSchema)
				if err != nil {
					t.Fatalf("structural schema error: %v", err)
				}
				if errs := structuralschema.ValidateStructural(nil, schema); len(errs) > 0 {
					t.Fatalf("structural schema validation error: %v", errs.ToAggregate())
				}
				schema = schema.Unfold()
			}

			got := newBuilder(&apiextensionsv1.CustomResourceDefinition{
				Spec: apiextensionsv1.CustomResourceDefinitionSpec{
					Group: "bar.k8s.io",
					Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
						{
							Name: "v1",
						},
					},
					Names: apiextensionsv1.CustomResourceDefinitionNames{
						Plural:   "foos",
						Singular: "foo",
						Kind:     "Foo",
						ListKind: "FooList",
					},
					Scope: apiextensionsv1.NamespaceScoped,
				},
			}, "v1", schema, tt.v2)

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
			for _, metaField := range []string{"kind", "apiVersion", "metadata"} {
				if _, found := got.schema.Properties["kind"]; found {
					prop := got.schema.Properties[metaField]
					prop.Description = ""
					got.schema.Properties[metaField] = prop
				}
			}

			if !reflect.DeepEqual(&wantedSchema, got.schema) {
				t.Errorf("unexpected schema: %s\nwant = %#v\ngot = %#v", schemaDiff(&wantedSchema, got.schema), &wantedSchema, got.schema)
			}

			gotListProperties := properties(got.listSchema.Properties)
			if want := sets.NewString("apiVersion", "kind", "metadata", "items"); !gotListProperties.Equal(want) {
				t.Fatalf("unexpected list properties, got: %s, expected: %s", gotListProperties.List(), want.List())
			}

			gotListSchema := got.listSchema.Properties["items"].Items.Schema
			if !reflect.DeepEqual(&wantedItemsSchema, gotListSchema) {
				t.Errorf("unexpected list schema: %s (want/got)", schemaDiff(&wantedItemsSchema, gotListSchema))
			}
		})
	}
}

func TestCRDRouteParameterBuilder(t *testing.T) {
	testCRDKind := "Foo"
	testCRDGroup := "foo-group"
	testCRDVersion := "foo-version"
	testCRDResourceName := "foos"

	testCases := []struct {
		scope apiextensionsv1.ResourceScope
		paths map[string]struct {
			expectNamespaceParam bool
			expectNameParam      bool
			expectedActions      sets.String
		}
	}{
		{
			scope: apiextensionsv1.NamespaceScoped,
			paths: map[string]struct {
				expectNamespaceParam bool
				expectNameParam      bool
				expectedActions      sets.String
			}{
				"/apis/foo-group/foo-version/foos":                                      {expectNamespaceParam: false, expectNameParam: false, expectedActions: sets.NewString("list")},
				"/apis/foo-group/foo-version/namespaces/{namespace}/foos":               {expectNamespaceParam: true, expectNameParam: false, expectedActions: sets.NewString("post", "list", "deletecollection")},
				"/apis/foo-group/foo-version/namespaces/{namespace}/foos/{name}":        {expectNamespaceParam: true, expectNameParam: true, expectedActions: sets.NewString("get", "put", "patch", "delete")},
				"/apis/foo-group/foo-version/namespaces/{namespace}/foos/{name}/scale":  {expectNamespaceParam: true, expectNameParam: true, expectedActions: sets.NewString("get", "patch", "put")},
				"/apis/foo-group/foo-version/namespaces/{namespace}/foos/{name}/status": {expectNamespaceParam: true, expectNameParam: true, expectedActions: sets.NewString("get", "patch", "put")},
			},
		},
		{
			scope: apiextensionsv1.ClusterScoped,
			paths: map[string]struct {
				expectNamespaceParam bool
				expectNameParam      bool
				expectedActions      sets.String
			}{
				"/apis/foo-group/foo-version/foos":               {expectNamespaceParam: false, expectNameParam: false, expectedActions: sets.NewString("post", "list", "deletecollection")},
				"/apis/foo-group/foo-version/foos/{name}":        {expectNamespaceParam: false, expectNameParam: true, expectedActions: sets.NewString("get", "put", "patch", "delete")},
				"/apis/foo-group/foo-version/foos/{name}/scale":  {expectNamespaceParam: false, expectNameParam: true, expectedActions: sets.NewString("get", "patch", "put")},
				"/apis/foo-group/foo-version/foos/{name}/status": {expectNamespaceParam: false, expectNameParam: true, expectedActions: sets.NewString("get", "patch", "put")},
			},
		},
	}

	for _, testCase := range testCases {
		testNamespacedCRD := &apiextensionsv1.CustomResourceDefinition{
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Scope: testCase.scope,
				Group: testCRDGroup,
				Names: apiextensionsv1.CustomResourceDefinitionNames{
					Kind:   testCRDKind,
					Plural: testCRDResourceName,
				},
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
					{
						Name: testCRDVersion,
						Subresources: &apiextensionsv1.CustomResourceSubresources{
							Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
							Scale:  &apiextensionsv1.CustomResourceSubresourceScale{},
						},
					},
				},
			},
		}
		swagger, err := BuildSwagger(testNamespacedCRD, testCRDVersion, Options{V2: true, StripDefaults: true})
		require.NoError(t, err)
		require.Equal(t, len(testCase.paths), len(swagger.Paths.Paths), testCase.scope)
		for path, expected := range testCase.paths {
			t.Run(path, func(t *testing.T) {
				path, ok := swagger.Paths.Paths[path]
				if !ok {
					t.Errorf("unexpected path %v", path)
				}

				hasNamespaceParam := false
				hasNameParam := false
				for _, param := range path.Parameters {
					if param.In == "path" && param.Name == "namespace" {
						hasNamespaceParam = true
					}
					if param.In == "path" && param.Name == "name" {
						hasNameParam = true
					}
				}
				assert.Equal(t, expected.expectNamespaceParam, hasNamespaceParam)
				assert.Equal(t, expected.expectNameParam, hasNameParam)

				actions := sets.NewString()
				for _, operation := range []*spec.Operation{path.Get, path.Post, path.Put, path.Patch, path.Delete} {
					if operation != nil {
						action, ok := operation.VendorExtensible.Extensions.GetString(endpoints.ROUTE_META_ACTION)
						if ok {
							actions.Insert(action)
						}
						if action == "patch" {
							expected := []string{"application/json-patch+json", "application/merge-patch+json"}
							if utilfeature.DefaultFeatureGate.Enabled(features.ServerSideApply) {
								expected = append(expected, "application/apply-patch+yaml")
							}
							assert.Equal(t, operation.Consumes, expected)
						} else {
							assert.Equal(t, operation.Consumes, []string{"application/json", "application/yaml"})
						}
					}
				}
				assert.Equal(t, expected.expectedActions, actions)
			})
		}
	}
}

func properties(p map[string]spec.Schema) sets.String {
	ret := sets.NewString()
	for k := range p {
		ret.Insert(k)
	}
	return ret
}

func schemaDiff(a, b *spec.Schema) string {
	as, err := json.Marshal(a)
	if err != nil {
		panic(err)
	}
	bs, err := json.Marshal(b)
	if err != nil {
		panic(err)
	}
	return diff.StringDiff(string(as), string(bs))
}

func TestBuildSwagger(t *testing.T) {
	tests := []struct {
		name                  string
		schema                string
		preserveUnknownFields *bool
		wantedSchema          string
		opts                  Options
	}{
		{
			"nil",
			"",
			nil,
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with properties",
			`{"type":"object","properties":{"spec":{"type":"object"},"status":{"type":"object"}}}`,
			nil,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"spec":{"type":"object"},"status":{"type":"object"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with invalid-typed properties",
			`{"type":"object","properties":{"spec":{"type":"bug"},"status":{"type":"object"}}}`,
			nil,
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with non-structural schema",
			`{"type":"object","properties":{"foo":{"type":"array"}}}`,
			nil,
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with spec.preseveUnknownFields=true",
			`{"type":"object","properties":{"foo":{"type":"string"}}}`,
			utilpointer.BoolPtr(true),
			`{"type":"object","x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with stripped defaults",
			`{"type":"object","properties":{"foo":{"type":"string","default":"bar"}}}`,
			nil,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"foo":{"type":"string"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"with stripped defaults",
			`{"type":"object","properties":{"foo":{"type":"string","default":"bar"}}}`,
			nil,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"foo":{"type":"string"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"v2",
			`{"type":"object","properties":{"foo":{"type":"string","oneOf":[{"pattern":"a"},{"pattern":"b"}]}}}`,
			nil,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"foo":{"type":"string"}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: true, StripDefaults: true},
		},
		{
			"v3",
			`{"type":"object","properties":{"foo":{"type":"string","oneOf":[{"pattern":"a"},{"pattern":"b"}]}}}`,
			nil,
			`{"type":"object","properties":{"apiVersion":{"type":"string"},"kind":{"type":"string"},"metadata":{"$ref":"#/definitions/io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"},"foo":{"type":"string","oneOf":[{"pattern":"a"},{"pattern":"b"}]}},"x-kubernetes-group-version-kind":[{"group":"bar.k8s.io","kind":"Foo","version":"v1"}]}`,
			Options{V2: false, StripDefaults: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var validation *apiextensionsv1.CustomResourceValidation
			if len(tt.schema) > 0 {
				v1Schema := &apiextensionsv1.JSONSchemaProps{}
				if err := json.Unmarshal([]byte(tt.schema), &v1Schema); err != nil {
					t.Fatal(err)
				}
				validation = &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: v1Schema,
				}
			}
			if tt.preserveUnknownFields != nil && *tt.preserveUnknownFields {
				validation.OpenAPIV3Schema.XPreserveUnknownFields = utilpointer.BoolPtr(true)
			}

			// TODO: mostly copied from the test above. reuse code to cleanup
			got, err := BuildSwagger(&apiextensionsv1.CustomResourceDefinition{
				Spec: apiextensionsv1.CustomResourceDefinitionSpec{
					Group: "bar.k8s.io",
					Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
						{
							Name:   "v1",
							Schema: validation,
						},
					},
					Names: apiextensionsv1.CustomResourceDefinitionNames{
						Plural:   "foos",
						Singular: "foo",
						Kind:     "Foo",
						ListKind: "FooList",
					},
					Scope: apiextensionsv1.NamespaceScoped,
				},
			}, "v1", tt.opts)
			if err != nil {
				t.Fatal(err)
			}

			var wantedSchema spec.Schema
			if err := json.Unmarshal([]byte(tt.wantedSchema), &wantedSchema); err != nil {
				t.Fatal(err)
			}

			gotSchema := got.Definitions["io.k8s.bar.v1.Foo"]
			gotProperties := properties(gotSchema.Properties)
			wantedProperties := properties(wantedSchema.Properties)
			if !gotProperties.Equal(wantedProperties) {
				t.Fatalf("unexpected properties, got: %s, expected: %s", gotProperties.List(), wantedProperties.List())
			}

			// wipe out TypeMeta/ObjectMeta content, with those many lines of descriptions. We trust that they match here.
			for _, metaField := range []string{"kind", "apiVersion", "metadata"} {
				if _, found := gotSchema.Properties["kind"]; found {
					prop := gotSchema.Properties[metaField]
					prop.Description = ""
					gotSchema.Properties[metaField] = prop
				}
			}

			if !reflect.DeepEqual(&wantedSchema, &gotSchema) {
				t.Errorf("unexpected schema: %s\nwant = %#v\ngot = %#v", schemaDiff(&wantedSchema, &gotSchema), &wantedSchema, &gotSchema)
			}
		})
	}
}
