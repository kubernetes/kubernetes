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

package unstructured_test

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/equality"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestNilUnstructuredContent(t *testing.T) {
	var u unstructured.Unstructured
	uCopy := u.DeepCopy()
	content := u.UnstructuredContent()
	expContent := make(map[string]interface{})
	assert.EqualValues(t, expContent, content)
	assert.Equal(t, uCopy, &u)
}

// TestUnstructuredMetadataRoundTrip checks that metadata accessors
// correctly set the metadata for unstructured objects.
// First, it fuzzes an empty ObjectMeta and sets this value as the metadata for an unstructured object.
// Next, it uses metadata accessor methods to set these fuzzed values to another unstructured object.
// Finally, it checks that both the unstructured objects are equal.
func TestUnstructuredMetadataRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

	N := 1000
	for i := 0; i < N; i++ {
		u := &unstructured.Unstructured{Object: map[string]interface{}{}}
		uCopy := u.DeepCopy()
		metadata := &metav1.ObjectMeta{}
		fuzzer.Fuzz(metadata)

		if err := setObjectMeta(u, metadata); err != nil {
			t.Fatalf("unexpected error setting fuzzed ObjectMeta: %v", err)
		}
		setObjectMetaUsingAccessors(u, uCopy)

		if !equality.Semantic.DeepEqual(u, uCopy) {
			t.Errorf("diff: %v", diff.ObjectReflectDiff(u, uCopy))
		}
	}
}

// TestUnstructuredMetadataOmitempty checks that ObjectMeta omitempty
// semantics are enforced for unstructured objects.
// The fuzzing test above should catch these cases but this is here just to be safe.
// Example: the metadata.clusterName field has the omitempty json tag
// so if it is set to it's zero value (""), it should be removed from the metadata map.
func TestUnstructuredMetadataOmitempty(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

	// fuzz to make sure we don't miss any function calls below
	u := &unstructured.Unstructured{Object: map[string]interface{}{}}
	metadata := &metav1.ObjectMeta{}
	fuzzer.Fuzz(metadata)
	if err := setObjectMeta(u, metadata); err != nil {
		t.Fatalf("unexpected error setting fuzzed ObjectMeta: %v", err)
	}

	// set zero values for all fields in metadata explicitly
	// to check that omitempty fields having zero values are never set
	u.SetName("")
	u.SetGenerateName("")
	u.SetNamespace("")
	u.SetSelfLink("")
	u.SetUID("")
	u.SetResourceVersion("")
	u.SetGeneration(0)
	u.SetCreationTimestamp(metav1.Time{})
	u.SetDeletionTimestamp(nil)
	u.SetDeletionGracePeriodSeconds(nil)
	u.SetLabels(nil)
	u.SetAnnotations(nil)
	u.SetOwnerReferences(nil)
	u.SetFinalizers(nil)
	u.SetClusterName("")
	u.SetManagedFields(nil)

	gotMetadata, _, err := unstructured.NestedFieldNoCopy(u.UnstructuredContent(), "metadata")
	if err != nil {
		t.Error(err)
	}
	emptyMetadata := make(map[string]interface{})

	if !reflect.DeepEqual(gotMetadata, emptyMetadata) {
		t.Errorf("expected %v, got %v", emptyMetadata, gotMetadata)
	}
}

func setObjectMeta(u *unstructured.Unstructured, objectMeta *metav1.ObjectMeta) error {
	if objectMeta == nil {
		unstructured.RemoveNestedField(u.UnstructuredContent(), "metadata")
		return nil
	}
	metadata, err := runtime.DefaultUnstructuredConverter.ToUnstructured(objectMeta)
	if err != nil {
		return err
	}
	u.UnstructuredContent()["metadata"] = metadata
	return nil
}

func setObjectMetaUsingAccessors(u, uCopy *unstructured.Unstructured) {
	uCopy.SetName(u.GetName())
	uCopy.SetGenerateName(u.GetGenerateName())
	uCopy.SetNamespace(u.GetNamespace())
	uCopy.SetSelfLink(u.GetSelfLink())
	uCopy.SetUID(u.GetUID())
	uCopy.SetResourceVersion(u.GetResourceVersion())
	uCopy.SetGeneration(u.GetGeneration())
	uCopy.SetCreationTimestamp(u.GetCreationTimestamp())
	uCopy.SetDeletionTimestamp(u.GetDeletionTimestamp())
	uCopy.SetDeletionGracePeriodSeconds(u.GetDeletionGracePeriodSeconds())
	uCopy.SetLabels(u.GetLabels())
	uCopy.SetAnnotations(u.GetAnnotations())
	uCopy.SetOwnerReferences(u.GetOwnerReferences())
	uCopy.SetFinalizers(u.GetFinalizers())
	uCopy.SetClusterName(u.GetClusterName())
	uCopy.SetManagedFields(u.GetManagedFields())
}
