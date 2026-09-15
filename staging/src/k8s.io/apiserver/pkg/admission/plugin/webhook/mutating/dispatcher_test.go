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

package mutating

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	jsonpatch "gopkg.in/evanphx/json-patch.v4"

	admissionv1 "k8s.io/api/admission/v1"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serializerjson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/admission"
)

type countingObject struct {
	value        string
	marshalCount *int
}

func (o *countingObject) MarshalJSON() ([]byte, error) {
	(*o.marshalCount)++
	return []byte(fmt.Sprintf(`{"value":%q}`, o.value)), nil
}

func (o *countingObject) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func (o *countingObject) DeepCopyObject() runtime.Object {
	copy := *o
	return &copy
}

func TestSerializedObjectsAreCachedUntilObjectUpdate(t *testing.T) {
	objectMarshalCount := 0
	oldObjectMarshalCount := 0
	accessor := &versionedAttributeAccessor{
		versionedAttr: &admission.VersionedAttributes{
			VersionedObject:    admission.NewLazyObject(&countingObject{value: "object", marshalCount: &objectMarshalCount}),
			VersionedOldObject: admission.NewLazyObject(&countingObject{value: "old-object", marshalCount: &oldObjectMarshalCount}),
		},
	}

	objectJSON, oldObjectJSON, err := accessor.serializedObjects()
	if err != nil {
		t.Fatalf("unexpected serialization error: %v", err)
	}
	secondObjectJSON, secondOldObjectJSON, err := accessor.serializedObjects()
	if err != nil {
		t.Fatalf("unexpected serialization error: %v", err)
	}
	if objectMarshalCount != 1 || oldObjectMarshalCount != 1 {
		t.Fatalf("expected each object to be serialized once, got object=%d oldObject=%d", objectMarshalCount, oldObjectMarshalCount)
	}
	if string(objectJSON) != string(secondObjectJSON) || string(oldObjectJSON) != string(secondOldObjectJSON) {
		t.Fatal("cached serialization changed between reads")
	}

	updatedObjectMarshalCount := 0
	accessor.updateObject(&countingObject{value: "updated", marshalCount: &updatedObjectMarshalCount})
	updatedObjectJSON, _, err := accessor.serializedObjects()
	if err != nil {
		t.Fatalf("unexpected serialization error after update: %v", err)
	}
	if updatedObjectMarshalCount != 1 {
		t.Fatalf("expected updated object to be serialized once, got %d", updatedObjectMarshalCount)
	}
	if oldObjectMarshalCount != 1 {
		t.Fatalf("expected immutable old object serialization to remain cached, got %d", oldObjectMarshalCount)
	}
	if got, want := string(updatedObjectJSON), `{"value":"updated"}`; got != want {
		t.Fatalf("unexpected updated object JSON: got %s, want %s", got, want)
	}
}

func TestSerializedObjectJSONMatchesWebhookPatchEncoding(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to register core types: %v", err)
	}
	serializer := serializerjson.NewSerializerWithOptions(serializerjson.DefaultMetaFactory, scheme, scheme, serializerjson.SerializerOptions{})

	tests := []struct {
		name   string
		object runtime.Object
	}{
		{
			name: "typed",
			object: &corev1.Pod{
				TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
				ObjectMeta: metav1.ObjectMeta{Name: "test", Labels: map[string]string{"a": "b"}},
			},
		},
		{
			name: "unstructured",
			object: &unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "example.com/v1",
				"kind":       "Example",
				"metadata": map[string]any{
					"name":   "test",
					"labels": map[string]any{"a": "b"},
				},
			}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			accessor := &versionedAttributeAccessor{versionedAttr: &admission.VersionedAttributes{
				VersionedObject: admission.NewLazyObject(test.object),
			}}
			got, _, err := accessor.serializedObjects()
			if err != nil {
				t.Fatalf("unexpected serialization error: %v", err)
			}
			want, err := runtime.Encode(serializer, test.object)
			if err != nil {
				t.Fatalf("unexpected webhook patch encoding error: %v", err)
			}
			assert.JSONEq(t, string(want), string(got))
		})
	}
}

func TestSetAdmissionReviewObjects(t *testing.T) {
	objectJSON := []byte(`{"object":"cached"}`)
	oldObjectJSON := []byte(`{"oldObject":"cached"}`)

	tests := []struct {
		name    string
		request runtime.Object
		verify  func(t *testing.T, request runtime.Object)
	}{
		{
			name: "v1",
			request: &admissionv1.AdmissionReview{Request: &admissionv1.AdmissionRequest{
				Object:    runtime.RawExtension{Object: &corev1.Pod{}},
				OldObject: runtime.RawExtension{Object: &corev1.Pod{}},
			}},
			verify: func(t *testing.T, request runtime.Object) {
				review := request.(*admissionv1.AdmissionReview)
				assert.Equal(t, objectJSON, review.Request.Object.Raw)
				assert.Nil(t, review.Request.Object.Object)
				assert.Equal(t, oldObjectJSON, review.Request.OldObject.Raw)
				assert.Nil(t, review.Request.OldObject.Object)
			},
		},
		{
			name: "v1beta1",
			request: &admissionv1beta1.AdmissionReview{Request: &admissionv1beta1.AdmissionRequest{
				Object:    runtime.RawExtension{Object: &corev1.Pod{}},
				OldObject: runtime.RawExtension{Object: &corev1.Pod{}},
			}},
			verify: func(t *testing.T, request runtime.Object) {
				review := request.(*admissionv1beta1.AdmissionReview)
				assert.Equal(t, objectJSON, review.Request.Object.Raw)
				assert.Nil(t, review.Request.Object.Object)
				assert.Equal(t, oldObjectJSON, review.Request.OldObject.Raw)
				assert.Nil(t, review.Request.OldObject.Object)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := setAdmissionReviewObjects(test.request, objectJSON, oldObjectJSON); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			test.verify(t, test.request)
		})
	}

	if err := setAdmissionReviewObjects(&metav1.Status{}, objectJSON, oldObjectJSON); err == nil {
		t.Fatal("expected an error for an unsupported admission review type")
	}
}

func BenchmarkAdmissionReviewObjectSerialization(b *testing.B) {
	labels := make(map[string]string, 1000)
	for i := 0; i < 1000; i++ {
		labels[fmt.Sprintf("label-%04d", i)] = fmt.Sprintf("value-%04d", i)
	}
	object := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "benchmark", Labels: labels}}
	oldObject := object.DeepCopy()

	b.Run("ten-webhooks/uncached", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for range 10 {
				review := &admissionv1.AdmissionReview{Request: &admissionv1.AdmissionRequest{
					Object:    runtime.RawExtension{Object: object},
					OldObject: runtime.RawExtension{Object: oldObject},
				}}
				if _, err := utiljson.Marshal(review); err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("ten-webhooks/cached", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			objectJSON, err := utiljson.Marshal(object)
			if err != nil {
				b.Fatal(err)
			}
			oldObjectJSON, err := utiljson.Marshal(oldObject)
			if err != nil {
				b.Fatal(err)
			}
			for range 10 {
				review := &admissionv1.AdmissionReview{Request: &admissionv1.AdmissionRequest{
					Object:    runtime.RawExtension{Raw: objectJSON},
					OldObject: runtime.RawExtension{Raw: oldObjectJSON},
				}}
				if _, err := utiljson.Marshal(review); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func TestMutationAnnotationValue(t *testing.T) {
	tcs := []struct {
		config   string
		webhook  string
		mutated  bool
		expected string
	}{
		{
			config:   "test-config",
			webhook:  "test-webhook",
			mutated:  true,
			expected: `{"configuration":"test-config","webhook":"test-webhook","mutated":true}`,
		},
		{
			config:   "test-config",
			webhook:  "test-webhook",
			mutated:  false,
			expected: `{"configuration":"test-config","webhook":"test-webhook","mutated":false}`,
		},
	}

	for _, tc := range tcs {
		actual, err := mutationAnnotationValue(tc.config, tc.webhook, tc.mutated)
		assert.NoError(t, err, "unexpected error")
		if actual != tc.expected {
			t.Errorf("composed mutation annotation value doesn't match, want: %s, got: %s", tc.expected, actual)
		}
	}
}

func TestJSONPatchAnnotationValue(t *testing.T) {
	tcs := []struct {
		name     string
		config   string
		webhook  string
		patch    []byte
		expected string
	}{
		{
			name:     "valid patch annotation",
			config:   "test-config",
			webhook:  "test-webhook",
			patch:    []byte(`[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			expected: `{"configuration":"test-config","webhook":"test-webhook","patch":[{"op":"add","path":"/metadata/labels/a","value":"true"}],"patchType":"JSONPatch"}`,
		},
		{
			name:     "empty configuration",
			config:   "",
			webhook:  "test-webhook",
			patch:    []byte(`[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			expected: `{"configuration":"","webhook":"test-webhook","patch":[{"op":"add","path":"/metadata/labels/a","value":"true"}],"patchType":"JSONPatch"}`,
		},
		{
			name:     "empty webhook",
			config:   "test-config",
			webhook:  "",
			patch:    []byte(`[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			expected: `{"configuration":"test-config","webhook":"","patch":[{"op":"add","path":"/metadata/labels/a","value":"true"}],"patchType":"JSONPatch"}`,
		},
		{
			name:     "valid JSON patch empty operation",
			config:   "test-config",
			webhook:  "test-webhook",
			patch:    []byte("[{}]"),
			expected: `{"configuration":"test-config","webhook":"test-webhook","patch":[{}],"patchType":"JSONPatch"}`,
		},
		{
			name:     "empty slice patch",
			config:   "test-config",
			webhook:  "test-webhook",
			patch:    []byte("[]"),
			expected: `{"configuration":"test-config","webhook":"test-webhook","patch":[],"patchType":"JSONPatch"}`,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			jsonPatch, err := jsonpatch.DecodePatch(tc.patch)
			assert.NoError(t, err, "unexpected error decode patch")
			actual, err := jsonPatchAnnotationValue(tc.config, tc.webhook, jsonPatch)
			assert.NoError(t, err, "unexpected error getting json patch annotation")
			if actual != tc.expected {
				t.Errorf("composed patch annotation value doesn't match, want: %s, got: %s", tc.expected, actual)
			}

			var p map[string]interface{}
			if err := json.Unmarshal([]byte(actual), &p); err != nil {
				t.Errorf("unexpected error unmarshaling patch annotation: %v", err)
			}
			if p["configuration"] != tc.config {
				t.Errorf("unmarshaled configuration doesn't match, want: %s, got: %v", tc.config, p["configuration"])
			}
			if p["webhook"] != tc.webhook {
				t.Errorf("unmarshaled webhook doesn't match, want: %s, got: %v", tc.webhook, p["webhook"])
			}
			var expectedPatch interface{}
			err = json.Unmarshal(tc.patch, &expectedPatch)
			if err != nil {
				t.Errorf("unexpected error unmarshaling patch: %v, %v", tc.patch, err)
			}
			if !reflect.DeepEqual(expectedPatch, p["patch"]) {
				t.Errorf("unmarshaled patch doesn't match, want: %v, got: %v", expectedPatch, p["patch"])
			}
		})
	}
}
