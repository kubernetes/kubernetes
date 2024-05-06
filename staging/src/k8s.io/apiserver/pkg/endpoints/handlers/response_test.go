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

package handlers

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

var _ runtime.CacheableObject = &mockCacheableObject{}

type mockCacheableObject struct {
	gvk schema.GroupVersionKind
	obj runtime.Object
}

// DeepCopyObject implements runtime.Object interface.
func (m *mockCacheableObject) DeepCopyObject() runtime.Object {
	panic("DeepCopy unimplemented for mockCacheableObject")
}

// GetObjectKind implements runtime.Object interface.
func (m *mockCacheableObject) GetObjectKind() schema.ObjectKind {
	return m
}

// GroupVersionKind implements schema.ObjectKind interface.
func (m *mockCacheableObject) GroupVersionKind() schema.GroupVersionKind {
	return m.gvk
}

// SetGroupVersionKind implements schema.ObjectKind interface.
func (m *mockCacheableObject) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	m.gvk = gvk
}

// CacheEncode implements runtime.CacheableObject interface.
func (m *mockCacheableObject) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	return encode(m.obj.DeepCopyObject(), w)
}

// GetObject implements runtime.CacheableObject interface.
func (m *mockCacheableObject) GetObject() runtime.Object {
	return m.obj
}

type mockNamer struct{}

func (*mockNamer) Namespace(_ *http.Request) (string, error)           { return "", nil }
func (*mockNamer) Name(_ *http.Request) (string, string, error)        { return "", "", nil }
func (*mockNamer) ObjectName(_ runtime.Object) (string, string, error) { return "", "", nil }

type mockEncoder struct {
	obj runtime.Object
}

func (e *mockEncoder) Encode(obj runtime.Object, _ io.Writer) error {
	e.obj = obj
	return nil
}

func (e *mockEncoder) Identifier() runtime.Identifier {
	return runtime.Identifier("")
}

func TestCacheableObject(t *testing.T) {
	pomGVK := metav1.SchemeGroupVersion.WithKind("PartialObjectMetadata")
	tableGVK := metav1.SchemeGroupVersion.WithKind("Table")

	status := &metav1.Status{Status: "status"}
	pod := &examplev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "name",
			Namespace: "namespace",
		},
	}
	podMeta := &metav1.PartialObjectMetadata{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "name",
			Namespace: "namespace",
		},
	}
	podMeta.GetObjectKind().SetGroupVersionKind(pomGVK)
	podTable := &metav1.Table{
		Rows: []metav1.TableRow{
			{
				Cells: []interface{}{pod.Name, pod.CreationTimestamp.Time.UTC().Format(time.RFC3339)},
			},
		},
	}

	tableConvertor := rest.NewDefaultTableConvertor(examplev1.Resource("Pod"))

	testCases := []struct {
		desc   string
		object runtime.Object
		opts   *metav1beta1.TableOptions
		target *schema.GroupVersionKind

		expectedUnwrap bool
		expectedObj    runtime.Object
		expectedErr    error
	}{
		{
			desc:        "metav1.Status",
			object:      status,
			expectedObj: status,
			expectedErr: nil,
		},
		{
			desc:        "cacheableObject nil convert",
			object:      &mockCacheableObject{obj: pod},
			target:      nil,
			expectedObj: pod,
			expectedErr: nil,
		},
		{
			desc:        "cacheableObject as PartialObjectMeta",
			object:      &mockCacheableObject{obj: pod},
			target:      &pomGVK,
			expectedObj: podMeta,
			expectedErr: nil,
		},
		{
			desc:        "cacheableObject as Table",
			object:      &mockCacheableObject{obj: pod},
			opts:        &metav1beta1.TableOptions{NoHeaders: true, IncludeObject: metav1.IncludeNone},
			target:      &tableGVK,
			expectedObj: podTable,
			expectedErr: nil,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			internalEncoder := &mockEncoder{}
			watchEncoder := newWatchEmbeddedEncoder(
				request.WithRequestInfo(context.TODO(), &request.RequestInfo{}),
				internalEncoder, test.target, test.opts,
				&RequestScope{
					Namer:          &mockNamer{},
					TableConvertor: tableConvertor,
				},
			)

			err := watchEncoder.Encode(test.object, nil)
			if err != test.expectedErr {
				t.Errorf("unexpected error: %v, expected: %v", err, test.expectedErr)
			}
			if a, e := internalEncoder.obj, test.expectedObj; !reflect.DeepEqual(a, e) {
				t.Errorf("unexpected result: %#v, expected: %#v", a, e)
			}
		})
	}
}

func TestAsPartialObjectMetadataList(t *testing.T) {
	var remainingItemCount int64 = 10
	pods := &examplev1.PodList{
		ListMeta: metav1.ListMeta{
			ResourceVersion:    "10",
			Continue:           "continuetoken",
			RemainingItemCount: &remainingItemCount,
		},
	}

	pomGVs := []schema.GroupVersion{metav1beta1.SchemeGroupVersion, metav1.SchemeGroupVersion}
	for _, gv := range pomGVs {
		t.Run(fmt.Sprintf("as %s PartialObjectMetadataList", gv), func(t *testing.T) {
			list, err := asPartialObjectMetadataList(pods, gv)
			if err != nil {
				t.Fatalf("failed to transform object: %v", err)
			}

			var listMeta metav1.ListMeta
			switch gv {
			case metav1beta1.SchemeGroupVersion:
				listMeta = list.(*metav1beta1.PartialObjectMetadataList).ListMeta
			case metav1.SchemeGroupVersion:
				listMeta = list.(*metav1.PartialObjectMetadataList).ListMeta
			}
			if !reflect.DeepEqual(pods.ListMeta, listMeta) {
				t.Errorf("unexpected list metadata: %v, expected: %v", listMeta, pods.ListMeta)
			}
		})
	}
}

func TestWatchEncoderIdentifier(t *testing.T) {
	eventFields := reflect.VisibleFields(reflect.TypeOf(metav1.WatchEvent{}))
	if len(eventFields) != 2 {
		t.Error("New field was added to metav1.WatchEvent.")
		t.Error("  Ensure that the following places are updated accordingly:")
		t.Error("  - watchEncoder::doEncode method when creating outEvent")
		t.Error("  - watchEncoder::typeIdentifier to capture all relevant fields in identifier")
	}
}
