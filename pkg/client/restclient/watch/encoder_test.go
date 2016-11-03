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

package versioned_test

import (
	"bytes"
	"io/ioutil"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/streaming"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	testCases := []struct {
		Type   watch.EventType
		Object runtime.Object
		Codec  runtime.Codec
	}{
		{
			watch.Added,
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			testapi.Default.Codec(),
		},
		{
			watch.Modified,
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			testapi.Default.Codec(),
		},
		{
			watch.Deleted,
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			testapi.Default.Codec(),
		},
	}
	for i, testCase := range testCases {
		buf := &bytes.Buffer{}

		codec := testCase.Codec
		encoder := versioned.NewEncoder(streaming.NewEncoder(buf, codec), codec)
		if err := encoder.Encode(&watch.Event{Type: testCase.Type, Object: testCase.Object}); err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}

		rc := ioutil.NopCloser(buf)
		decoder := versioned.NewDecoder(streaming.NewDecoder(rc, codec), codec)
		event, obj, err := decoder.Decode()
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if !api.Semantic.DeepDerivative(testCase.Object, obj) {
			t.Errorf("%d: expected %#v, got %#v", i, testCase.Object, obj)
		}
		if event != testCase.Type {
			t.Errorf("%d: unexpected type: %#v", i, event)
		}
	}
}
