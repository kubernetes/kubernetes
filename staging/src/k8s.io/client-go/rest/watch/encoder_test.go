/*
Copyright 2014 The Kubernetes Authors.

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

package watch

import (
	"bytes"
	"io"
	"testing"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
)

// getEncoder mimics how k8s.io/client-go/rest.createSerializers creates a encoder
func getEncoder() runtime.Encoder {
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme.Scheme, scheme.Scheme, runtimejson.SerializerOptions{})
	directCodecFactory := scheme.Codecs.WithoutConversion()
	return directCodecFactory.EncoderForVersion(jsonSerializer, v1.SchemeGroupVersion)
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	testCases := []struct {
		Type   watch.EventType
		Object runtime.Object
	}{
		{
			watch.Added,
			&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			watch.Modified,
			&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			watch.Deleted,
			&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			watch.Bookmark,
			&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
	}
	for i, testCase := range testCases {
		buf := &bytes.Buffer{}

		encoder := NewEncoder(streaming.NewEncoder(buf, getEncoder()), getEncoder())
		if err := encoder.Encode(&watch.Event{Type: testCase.Type, Object: testCase.Object}); err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}

		rc := io.NopCloser(buf)
		decoder := NewDecoder(streaming.NewDecoder(rc, getDecoder()), getDecoder())
		event, obj, err := decoder.Decode()
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if !apiequality.Semantic.DeepDerivative(testCase.Object, obj) {
			t.Errorf("%d: expected %#v, got %#v", i, testCase.Object, obj)
		}
		if event != testCase.Type {
			t.Errorf("%d: unexpected type: %#v", i, event)
		}
	}
}
