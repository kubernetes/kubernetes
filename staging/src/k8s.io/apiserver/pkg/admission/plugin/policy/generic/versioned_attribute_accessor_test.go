/*
Copyright 2026 The Kubernetes Authors.

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

package generic

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

func TestVersionedAttributeAccessorRetainsIndependentKinds(t *testing.T) {
	accessor, inputGVK, firstGVK, secondGVK := newTestVersionedAttributeAccessor(t)
	first, err := accessor.VersionedAttribute(firstGVK)
	require.NoError(t, err)
	second, err := accessor.VersionedAttribute(secondGVK)
	require.NoError(t, err)
	require.NotSame(t, first, second)

	firstObject := first.VersionedObject.Object().(*unstructured.Unstructured).DeepCopy()
	require.NoError(t, unstructured.SetNestedField(firstObject.Object, "first", "metadata", "labels", "marker"))
	first.UpdateObject(firstObject)

	secondLabels, _, err := unstructured.NestedStringMap(second.VersionedObject.Object().(*unstructured.Unstructured).Object, "metadata", "labels")
	require.NoError(t, err)
	_, exists := secondLabels["marker"]
	require.False(t, exists, "validation snapshots must remain independent")
	input, err := accessor.VersionedAttribute(inputGVK)
	require.NoError(t, err)
	inputLabels, _, err := unstructured.NestedStringMap(input.VersionedObject.Object().(*unstructured.Unstructured).Object, "metadata", "labels")
	require.NoError(t, err)
	_, exists = inputLabels["marker"]
	require.False(t, exists)
}

func TestVersionedAttributeAccessorConcurrentKinds(t *testing.T) {
	accessor, _, firstGVK, secondGVK := newTestVersionedAttributeAccessor(t)
	_, err := accessor.VersionedAttribute(firstGVK)
	require.NoError(t, err)
	_, err = accessor.VersionedAttribute(secondGVK)
	require.NoError(t, err)
	type result struct {
		attr *admission.VersionedAttributes
		err  error
	}
	results := make(chan result, 2)
	var wg sync.WaitGroup
	for _, gvk := range []schema.GroupVersionKind{firstGVK, secondGVK} {
		wg.Add(1)
		go func(gvk schema.GroupVersionKind) {
			defer wg.Done()
			attr, err := accessor.VersionedAttribute(gvk)
			results <- result{attr: attr, err: err}
		}(gvk)
	}
	wg.Wait()
	close(results)
	for result := range results {
		require.NoError(t, result.err)
		require.NotNil(t, result.attr)
		require.NotNil(t, result.attr.VersionedObject.Object())
	}
}

func newTestVersionedAttributeAccessor(t *testing.T) (*versionedAttributeAccessor, schema.GroupVersionKind, schema.GroupVersionKind, schema.GroupVersionKind) {
	t.Helper()
	inputGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	firstGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1alpha1", Kind: "Widget"}
	secondGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1beta1", Kind: "Widget"}
	scheme := runtime.NewScheme()
	for _, gvk := range []schema.GroupVersionKind{inputGVK, firstGVK, secondGVK} {
		scheme.AddKnownTypeWithName(gvk, &unstructured.Unstructured{})
	}
	object := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": inputGVK.GroupVersion().String(),
		"kind":       inputGVK.Kind,
		"metadata":   map[string]interface{}{"name": "demo", "labels": map[string]interface{}{}},
	}}
	object.SetGroupVersionKind(inputGVK)
	attrs := admission.NewAttributesRecord(object, nil, inputGVK, "", "demo",
		schema.GroupVersionResource{Group: inputGVK.Group, Version: inputGVK.Version, Resource: "widgets"},
		"", admission.Create, &metav1.CreateOptions{}, false, nil)
	return &versionedAttributeAccessor{versionedAttrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{}, attr: attrs, objectInterfaces: admission.NewObjectInterfacesFromScheme(scheme)}, inputGVK, firstGVK, secondGVK
}
