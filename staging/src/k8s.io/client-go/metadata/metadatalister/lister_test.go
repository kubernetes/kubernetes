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

package metadatalister

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
)

func TestNamespaceGetMethod(t *testing.T) {
	tests := []struct {
		name            string
		existingObjects []runtime.Object
		namespaceToSync string
		gvrToSync       schema.GroupVersionResource
		objectToGet     string
		expectedObject  *metav1.PartialObjectMetadata
		expectError     bool
	}{
		{
			name: "scenario 1: gets name-foo1 resource from the indexer from ns-foo namespace",
			existingObjects: []runtime.Object{
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-bar", "name-bar"),
			},
			namespaceToSync: "ns-foo",
			gvrToSync:       schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"},
			objectToGet:     "name-foo1",
			expectedObject:  newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
		},
		{
			name: "scenario 2: gets name-foo-non-existing resource from the indexer from ns-foo namespace",
			existingObjects: []runtime.Object{
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-bar", "name-bar"),
			},
			namespaceToSync: "ns-foo",
			gvrToSync:       schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"},
			objectToGet:     "name-foo-non-existing",
			expectError:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// test data
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			for _, obj := range test.existingObjects {
				err := indexer.Add(obj)
				if err != nil {
					t.Fatal(err)
				}
			}
			// act
			target := New(indexer, test.gvrToSync).Namespace(test.namespaceToSync)
			actualObject, err := target.Get(test.objectToGet)

			// validate
			if test.expectError {
				if err == nil {
					t.Fatal("expected to get an error but non was returned")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(test.expectedObject, actualObject) {
				t.Fatalf("unexpected object has been returned expected = %v actual = %v, diff = %v", test.expectedObject, actualObject, cmp.Diff(test.expectedObject, actualObject))
			}
		})
	}
}

func TestNamespaceListMethod(t *testing.T) {
	// test data
	objs := []runtime.Object{
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
		newPartialObjectMetadata("group/version", "TheKind", "ns-bar", "name-bar"),
	}
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, obj := range objs {
		err := indexer.Add(obj)
		if err != nil {
			t.Fatal(err)
		}
	}
	expectedOutput := []*metav1.PartialObjectMetadata{
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
	}
	namespaceToList := "ns-foo"

	// act
	target := New(indexer, schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"}).Namespace(namespaceToList)
	actualOutput, err := target.List(labels.Everything())

	// validate
	if err != nil {
		t.Fatal(err)
	}
	assertListOrDie(expectedOutput, actualOutput, t)
}

func TestListerGetMethod(t *testing.T) {
	tests := []struct {
		name            string
		existingObjects []runtime.Object
		namespaceToSync string
		gvrToSync       schema.GroupVersionResource
		objectToGet     string
		expectedObject  *metav1.PartialObjectMetadata
		expectError     bool
	}{
		{
			name: "scenario 1: gets name-foo1 resource from the indexer",
			existingObjects: []runtime.Object{
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
				newPartialObjectMetadata("group/version", "TheKind", "", "name-foo1"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-bar", "name-bar"),
			},
			namespaceToSync: "",
			gvrToSync:       schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"},
			objectToGet:     "name-foo1",
			expectedObject:  newPartialObjectMetadata("group/version", "TheKind", "", "name-foo1"),
		},
		{
			name: "scenario 2: doesn't get name-foo resource from the indexer from ns-foo namespace",
			existingObjects: []runtime.Object{
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo1"),
				newPartialObjectMetadata("group/version", "TheKind", "ns-bar", "name-bar"),
			},
			namespaceToSync: "ns-foo",
			gvrToSync:       schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"},
			objectToGet:     "name-foo",
			expectError:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// test data
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			for _, obj := range test.existingObjects {
				err := indexer.Add(obj)
				if err != nil {
					t.Fatal(err)
				}
			}
			// act
			target := New(indexer, test.gvrToSync)
			actualObject, err := target.Get(test.objectToGet)

			// validate
			if test.expectError {
				if err == nil {
					t.Fatal("expected to get an error but non was returned")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(test.expectedObject, actualObject) {
				t.Fatalf("unexpected object has been returned expected = %v actual = %v, diff = %v", test.expectedObject, actualObject, cmp.Diff(test.expectedObject, actualObject))
			}
		})
	}
}

func TestListerListMethod(t *testing.T) {
	// test data
	objs := []runtime.Object{
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-bar"),
	}
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, obj := range objs {
		err := indexer.Add(obj)
		if err != nil {
			t.Fatal(err)
		}
	}
	expectedOutput := []*metav1.PartialObjectMetadata{
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-foo"),
		newPartialObjectMetadata("group/version", "TheKind", "ns-foo", "name-bar"),
	}

	// act
	target := New(indexer, schema.GroupVersionResource{Group: "group", Version: "version", Resource: "TheKinds"})
	actualOutput, err := target.List(labels.Everything())

	// validate
	if err != nil {
		t.Fatal(err)
	}
	assertListOrDie(expectedOutput, actualOutput, t)
}

func assertListOrDie(expected, actual []*metav1.PartialObjectMetadata, t *testing.T) {
	if len(actual) != len(expected) {
		t.Fatalf("unexpected number of items returned, expected = %d, actual = %d", len(expected), len(actual))
	}
	for _, expectedObject := range expected {
		found := false
		for _, actualObject := range actual {
			if actualObject.GetName() == expectedObject.GetName() {
				if !reflect.DeepEqual(expectedObject, actualObject) {
					t.Fatalf("unexpected object has been returned expected = %v actual = %v, diff = %v", expectedObject, actualObject, cmp.Diff(expectedObject, actualObject))
				}
				found = true
			}
		}
		if !found {
			t.Fatalf("the resource with the name = %s was not found in the returned output", expectedObject.GetName())
		}
	}
}

func newPartialObjectMetadata(apiVersion, kind, namespace, name string) *metav1.PartialObjectMetadata {
	return &metav1.PartialObjectMetadata{
		TypeMeta: metav1.TypeMeta{
			APIVersion: apiVersion,
			Kind:       kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
	}
}
