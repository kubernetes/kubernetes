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

package integration

import (
	"testing"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/yaml"
)

var immutableFixture = &apiextensionsv1beta1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "foos.tests.example.com"},
	Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
		Group:   "tests.example.com",
		Version: "v1beta1",
		Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
			Plural:   "foos",
			Singular: "foo",
			Kind:     "Foo",
			ListKind: "FooList",
		},
		Scope: apiextensionsv1beta1.ClusterScoped,
		Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
			Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
		},
	},
}

const (
	immutableFooSchema = `
type: object
properties:
  foo:
    type: string
    x-kubernetes-immutability: immutable
  bar:
    type: string
    x-kubernetes-immutability: immutable
  mutable:
    type: string
`

	immutableNestedFooSchema = `
type: object
properties:
  foo:
    type: object
    properties:
      nested:
        x-kubernetes-immutability: immutable
        type: string
      mutable:
        type: string
  bar:
    type: string
    x-kubernetes-immutability: immutable
  mutable:
    type: string
`
	addOnlyNestedFooSchema = `
type: object
properties:
  foo:
    type: object
    properties:
      nested:
        x-kubernetes-immutability: addOnly
        type: string
      mutable:
        type: string
  mutable:
    type: string
`
	removeOnlyNestedFooSchema = `
type: object
properties:
  foo:
    type: object
    properties:
      nested:
        x-kubernetes-immutability: removeOnly
        type: string
      mutable:
        type: string
  mutable:
    type: string
`
	mapImmutableValuesFooSchema = `
type: object
properties:
  foo:
    type: object
    additionalProperties:
      type: string
      x-kubernetes-immutability: immutable
  mutable:
    type: string
`
	mapImmutableKeysFooSchema = `
type: object
properties:
  foo:
    type: object
    x-kubernetes-key-immutability: immutable
    additionalProperties:
      type: string
  mutable:
    type: string
`

	immutableMapFooInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
metadata:
  name: foo
foo:
  a: "v1"
  b: "v2"
mutable: "a"
`

	immutableFooInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
metadata:
  name: foo
foo: "immutable"
bar: "immutable"
mutable: "a"
`
	immutableNestedFooInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
metadata:
  name: foo
foo:
  nested: "immutable"
bar: "immutable"
mutable: "a"
`
	immutableNestedNonSetFooInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
metadata:
  name: foo
foo:
  mutable: ""
mutable: "a"
`
)

func TestImmutableUpdate(t *testing.T) {
	testCases := []struct {
		testCase          string
		immutableSchema   string
		immutableInstance string
		mutations         func(obj map[string]interface{})
		error             bool
	}{
		{
			testCase:          "immutableString",
			immutableSchema:   immutableFooSchema,
			immutableInstance: immutableFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo")
				unstructured.SetNestedField(obj, "mutating", "bar")
			},
			error: true,
		},
		{
			testCase:          "immutableStringNested",
			immutableSchema:   immutableNestedFooSchema,
			immutableInstance: immutableNestedFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo", "nested")
			},
			error: true,
		},
		{
			testCase:          "immutableNestedNonSet",
			immutableSchema:   immutableNestedFooSchema,
			immutableInstance: immutableNestedNonSetFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo", "nested")
			},
			error: true,
		},
		{
			testCase:          "addOnlyNested",
			immutableSchema:   addOnlyNestedFooSchema,
			immutableInstance: immutableNestedFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo", "nested")
			},
			error: true,
		},
		{
			testCase:          "addOnlyNested (positive)",
			immutableSchema:   addOnlyNestedFooSchema,
			immutableInstance: immutableNestedNonSetFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo", "nested")
			},
			error: false,
		},
		{
			testCase:          "removeOnlyNested",
			immutableSchema:   removeOnlyNestedFooSchema,
			immutableInstance: immutableNestedFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutating", "foo", "nested")
			},
			error: true,
		},
		{
			testCase:          "removeOnlyNested (positive)",
			immutableSchema:   removeOnlyNestedFooSchema,
			immutableInstance: immutableNestedFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.RemoveNestedField(obj, "foo", "nested")
			},
			error: false,
		},
		{
			testCase:          "mapImmutableValues",
			immutableSchema:   mapImmutableValuesFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutated", "foo", "a")
			},
			error: true,
		},
		{
			testCase:          "mapImmutableValues (add Key-value)",
			immutableSchema:   mapImmutableValuesFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutated", "foo", "c")
			},
			error: false,
		},
		{
			testCase:          "mapImmutableValues (remove Key-value)",
			immutableSchema:   mapImmutableValuesFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.RemoveNestedField(obj, "foo", "b")
			},
			error: false,
		},
		{
			testCase:          "mapImmutableKeys (different value for same key)",
			immutableSchema:   mapImmutableKeysFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutated", "foo", "b")
			},
			error: false,
		},
		{
			testCase:          "mapImmutableKeys (remove key)",
			immutableSchema:   mapImmutableKeysFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.RemoveNestedField(obj, "foo", "b")
			},
			error: true,
		},
		{
			testCase:          "mapImmutableKeys (add key)",
			immutableSchema:   mapImmutableKeysFooSchema,
			immutableInstance: immutableMapFooInstance,
			mutations: func(obj map[string]interface{}) {
				unstructured.SetNestedField(obj, "mutated", "foo", "c")
			},
			error: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testCase, func(t *testing.T) {
			tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
			if err != nil {
				t.Fatal(err)
			}
			defer tearDownFn()

			crd := immutableFixture.DeepCopy()
			crd.Spec.Validation = &apiextensionsv1beta1.CustomResourceValidation{}
			if err := yaml.Unmarshal([]byte(tc.immutableSchema), &crd.Spec.Validation.OpenAPIV3Schema); err != nil {
				t.Fatal(err)
			}

			crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
			if err != nil {
				t.Fatal(err)
			}

			t.Logf("Creating CR")
			fooClient := dynamicClient.Resource(schema.GroupVersionResource{crd.Spec.Group, crd.Spec.Version, crd.Spec.Names.Plural})
			foo := &unstructured.Unstructured{}
			if err := yaml.Unmarshal([]byte(tc.immutableInstance), &foo.Object); err != nil {
				t.Fatal(err)
			}

			foo, err = fooClient.Create(foo, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Unable to create CR: %v", err)
			}

			tc.mutations(foo.Object)
			foo, err = fooClient.Update(foo, metav1.UpdateOptions{})
			if (err != nil) != tc.error {
				t.Fatalf("Updated: %#v. Got: %v. Expected error: %v", foo.UnstructuredContent(), err, tc.error)
			}
			t.Logf("Got: %v. Expected error: %v", err, tc.error)
		})
	}
}
