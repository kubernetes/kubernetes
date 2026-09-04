/*
Copyright The Kubernetes Authors.

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

package cacher

import (
	"io"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/storage"
)

func testPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "pod",
			Namespace:       "ns",
			ResourceVersion: "100",
			Labels:          map[string]string{"app": "test"},
			Annotations:     map[string]string{"key": "value"},
			ManagedFields:   []metav1.ManagedFieldsEntry{{Manager: "kubelet"}},
		},
		Spec: corev1.PodSpec{
			NodeName:   "node",
			Containers: []corev1.Container{{Name: "c", Image: "image"}},
		},
	}
}

func TestGetMutableObject(t *testing.T) {
	pod := testPod()
	mutable := getMutableObject(pod)
	if mutable == runtime.Object(pod) {
		t.Fatalf("getMutableObject returned the original object")
	}
	if diff := cmp.Diff(pod, mutable); diff != "" {
		t.Errorf("unexpected difference between original and copy (-want +got):\n%s", diff)
	}

	// Writing the fields the watch path is allowed to write must not be
	// visible through the original object.
	mutable.GetObjectKind().SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("Pod"))
	versioner := storage.APIObjectVersioner{}
	if err := versioner.UpdateObject(mutable, 200); err != nil {
		t.Fatal(err)
	}
	empty := metav1.TypeMeta{}
	if pod.TypeMeta != empty {
		t.Errorf("original object TypeMeta was modified: %#v", pod.TypeMeta)
	}
	if got := pod.ResourceVersion; got != "100" {
		t.Errorf("original object resourceVersion was modified: %q", got)
	}
}

func TestGetMutableObjectCachingObject(t *testing.T) {
	cached, err := newCachingObject(testPod())
	if err != nil {
		t.Fatal(err)
	}
	if mutable := getMutableObject(cached); mutable != runtime.Object(cached) {
		t.Errorf("expected the cachingObject to be returned as is, got %T", mutable)
	}
}

func TestGetMutableObjectUnstructured(t *testing.T) {
	object := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "example.com/v1",
		"kind":       "Example",
		"metadata":   map[string]interface{}{"name": "example"},
	}}
	mutable := getMutableObject(object)
	mutable.GetObjectKind().SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("Pod"))
	if got := object.GetObjectKind().GroupVersionKind().String(); got != "example.com/v1, Kind=Example" {
		t.Errorf("original unstructured object was modified, got %q", got)
	}
}

// TestGetMutableObjectConcurrentEncode is the reason getMutableObject copies at
// all: the same object is served to every watcher, and encoding it writes
// TypeMeta. Run with -race.
func TestGetMutableObjectConcurrentEncode(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	metav1.AddToGroupVersion(scheme, corev1.SchemeGroupVersion)
	codecs := serializer.NewCodecFactory(scheme)

	shared := testPod()
	expected := shared.DeepCopy()

	for _, info := range codecs.SupportedMediaTypes() {
		var wg sync.WaitGroup
		for range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				encoder := codecs.EncoderForVersion(info.Serializer, corev1.SchemeGroupVersion)
				for range 100 {
					if err := encoder.Encode(getMutableObject(shared), io.Discard); err != nil {
						t.Errorf("failed to encode: %v", err)
						return
					}
				}
			}()
		}
		wg.Wait()
	}
	if diff := cmp.Diff(expected, shared); diff != "" {
		t.Errorf("shared object was modified by encoding (-want +got):\n%s", diff)
	}
}

func BenchmarkGetMutableObject(b *testing.B) {
	pod := testPod()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = getMutableObject(pod)
	}
}
