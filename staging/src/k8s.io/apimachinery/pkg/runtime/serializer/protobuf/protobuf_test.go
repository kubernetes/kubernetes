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

package protobuf

import (
	"bytes"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
)

func TestCacheableObject(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "group", Version: "version", Kind: "MockCacheableObject"}
	creater := &mockCreater{obj: &runtimetesting.MockCacheableObject{}}
	typer := &mockTyper{gvk: &gvk}

	encoders := []runtime.Encoder{
		NewSerializer(creater, typer),
		NewRawSerializer(creater, typer),
	}

	for _, encoder := range encoders {
		runtimetesting.CacheableObjectTest(t, encoder)
	}
}

type mockCreater struct {
	apiVersion string
	kind       string
	err        error
	obj        runtime.Object
}

func (c *mockCreater) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	c.apiVersion, c.kind = kind.GroupVersion().String(), kind.Kind
	return c.obj, c.err
}

type mockTyper struct {
	gvk *schema.GroupVersionKind
	err error
}

func (t *mockTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	if t.gvk == nil {
		return nil, false, t.err
	}
	return []schema.GroupVersionKind{*t.gvk}, false, t.err
}

func (t *mockTyper) Recognizes(_ schema.GroupVersionKind) bool {
	return false
}

func TestSerializerEncodeWithAllocator(t *testing.T) {
	testCases := []struct {
		name string
		obj  runtime.Object
	}{
		{
			name: "encode a bufferedMarshaller obj",
			obj: &testapigroupv1.Carp{
				TypeMeta: metav1.TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "name",
					Namespace: "namespace",
				},
				Spec: testapigroupv1.CarpSpec{
					Subdomain: "carp.k8s.io",
				},
			},
		},

		{
			name: "encode a runtime.Unknown obj",
			obj:  &runtime.Unknown{TypeMeta: runtime.TypeMeta{APIVersion: "group/version", Kind: "Unknown"}, Raw: []byte("hello world")},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			target := NewSerializer(nil, nil)

			writer := &bytes.Buffer{}
			if err := target.Encode(tc.obj, writer); err != nil {
				t.Fatal(err)
			}

			writer2 := &bytes.Buffer{}
			alloc := &testAllocator{}
			if err := target.EncodeWithAllocator(tc.obj, writer2, alloc); err != nil {
				t.Fatal(err)
			}
			if alloc.allocateCount != 1 {
				t.Fatalf("expected the Allocate method to be called exactly 1 but it was executed: %v times ", alloc.allocateCount)
			}

			// to ensure compatibility of the new method with the old one, serialized data must be equal
			// also we are not testing decoding since "roundtripping" is tested elsewhere for all known types
			if !reflect.DeepEqual(writer.Bytes(), writer2.Bytes()) {
				t.Fatal("data mismatch, data serialized with the Encode method is different than serialized with the EncodeWithAllocator method")
			}
		})
	}
}

func TestRawSerializerEncodeWithAllocator(t *testing.T) {
	testCases := []struct {
		name string
		obj  runtime.Object
	}{
		{
			name: "encode a bufferedReverseMarshaller obj",
			obj: &testapigroupv1.Carp{
				TypeMeta: metav1.TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "name",
					Namespace: "namespace",
				},
				Spec: testapigroupv1.CarpSpec{
					Subdomain: "carp.k8s.io",
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			writer := &bytes.Buffer{}
			target := NewRawSerializer(nil, nil)

			if err := target.Encode(tc.obj, writer); err != nil {
				t.Fatal(err)
			}

			writer2 := &bytes.Buffer{}
			alloc := &testAllocator{}
			if err := target.EncodeWithAllocator(tc.obj, writer2, alloc); err != nil {
				t.Fatal(err)
			}
			if alloc.allocateCount != 1 {
				t.Fatalf("expected the Allocate method to be called exactly 1 but it was executed: %v times ", alloc.allocateCount)
			}

			// to ensure compatibility of the new method with the old one, serialized data must be equal
			// also we are not testing decoding since "roundtripping" is tested elsewhere for all known types
			if !reflect.DeepEqual(writer.Bytes(), writer2.Bytes()) {
				t.Fatal("data mismatch, data serialized with the Encode method is different than serialized with the EncodeWithAllocator method")
			}
		})
	}
}

type testAllocator struct {
	buf           []byte
	allocateCount int
}

func (ta *testAllocator) Allocate(n uint64) []byte {
	ta.buf = make([]byte, n)
	ta.allocateCount++
	return ta.buf
}
