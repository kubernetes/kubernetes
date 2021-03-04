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

package testing

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// nonCacheableTestObject implements json.Marshaler and proto.Marshaler interfaces
// for mocking purpose.
// +k8s:deepcopy-gen=false
type noncacheableTestObject struct {
	gvk schema.GroupVersionKind
}

// MarshalJSON implements json.Marshaler interface.
func (*noncacheableTestObject) MarshalJSON() ([]byte, error) {
	return []byte("\"json-result\""), nil
}

// Marshal implements proto.Marshaler interface.
func (*noncacheableTestObject) Marshal() ([]byte, error) {
	return []byte("\"proto-result\""), nil
}

// DeepCopyObject implements runtime.Object interface.
func (*noncacheableTestObject) DeepCopyObject() runtime.Object {
	panic("DeepCopy unimplemented for noncacheableTestObject")
}

// GetObjectKind implements runtime.Object interface.
func (o *noncacheableTestObject) GetObjectKind() schema.ObjectKind {
	return o
}

// GroupVersionKind implements schema.ObjectKind interface.
func (o *noncacheableTestObject) GroupVersionKind() schema.GroupVersionKind {
	return o.gvk
}

// SetGroupVersionKind implements schema.ObjectKind interface.
func (o *noncacheableTestObject) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	o.gvk = gvk
}

var _ runtime.CacheableObject = &MockCacheableObject{}

// MochCacheableObject is used to test CacheableObject interface.
// +k8s:deepcopy-gen=false
type MockCacheableObject struct {
	gvk schema.GroupVersionKind

	t *testing.T

	runEncode      bool
	returnSelf     bool
	expectedResult string
	expectedError  error

	intercepted []runtime.Identifier
}

// DeepCopyObject implements runtime.Object interface.
func (m *MockCacheableObject) DeepCopyObject() runtime.Object {
	panic("DeepCopy unimplemented for MockCacheableObject")
}

// GetObjectKind implements runtime.Object interface.
func (m *MockCacheableObject) GetObjectKind() schema.ObjectKind {
	return m
}

// GroupVersionKind implements schema.ObjectKind interface.
func (m *MockCacheableObject) GroupVersionKind() schema.GroupVersionKind {
	return m.gvk
}

// SetGroupVersionKind implements schema.ObjectKind interface.
func (m *MockCacheableObject) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	m.gvk = gvk
}

// Marshal implements proto.Marshaler interface.
// This is implemented to avoid errors from protobuf serializer.
func (*MockCacheableObject) Marshal() ([]byte, error) {
	return []byte("\"proto-result\""), nil
}

// CacheEncode implements runtime.CacheableObject interface.
func (m *MockCacheableObject) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	m.intercepted = append(m.intercepted, id)
	if m.runEncode {
		return encode(m.GetObject(), w)
	}
	if _, err := w.Write([]byte(m.expectedResult)); err != nil {
		m.t.Errorf("couldn't write to io.Writer: %v", err)
	}
	return m.expectedError
}

// GetObject implements runtime.CacheableObject interface.
func (m *MockCacheableObject) GetObject() runtime.Object {
	if m.returnSelf {
		return m
	}
	gvk := schema.GroupVersionKind{Group: "group", Version: "version", Kind: "noncacheableTestObject"}
	return &noncacheableTestObject{gvk: gvk}
}

func (m *MockCacheableObject) interceptedCalls() []runtime.Identifier {
	return m.intercepted
}

type testBuffer struct {
	writer io.Writer
	t      *testing.T
	object *MockCacheableObject
}

// Write implements io.Writer interface.
func (b *testBuffer) Write(p []byte) (int, error) {
	// Before writing any byte, check if <object> has already
	// intercepted any CacheEncode operation.
	if len(b.object.interceptedCalls()) == 0 {
		b.t.Errorf("writing to buffer without handling MockCacheableObject")
	}
	return b.writer.Write(p)
}

// CacheableObjectTest implements a test that should be run for every
// runtime.Encoder interface implementation.
// It checks whether CacheableObject is properly supported by it.
func CacheableObjectTest(t *testing.T, e runtime.Encoder) {
	gvk1 := schema.GroupVersionKind{Group: "group", Version: "version1", Kind: "MockCacheableObject"}

	testCases := []struct {
		desc           string
		runEncode      bool
		returnSelf     bool
		expectedResult string
		expectedError  error
	}{
		{
			desc:      "delegate",
			runEncode: true,
		},
		{
			desc:       "delegate return self",
			runEncode:  true,
			returnSelf: true,
		},
		{
			desc:           "cached success",
			runEncode:      false,
			expectedResult: "result",
			expectedError:  nil,
		},
		{
			desc:           "cached failure",
			runEncode:      false,
			expectedResult: "",
			expectedError:  fmt.Errorf("encoding error"),
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			obj := &MockCacheableObject{
				gvk:            gvk1,
				t:              t,
				runEncode:      test.runEncode,
				returnSelf:     test.returnSelf,
				expectedResult: test.expectedResult,
				expectedError:  test.expectedError,
			}
			buffer := bytes.NewBuffer(nil)
			w := &testBuffer{
				writer: buffer,
				t:      t,
				object: obj,
			}

			if err := e.Encode(obj, w); err != test.expectedError {
				t.Errorf("unexpected error: %v, expected: %v", err, test.expectedError)
			}
			if !test.runEncode {
				if result := buffer.String(); result != test.expectedResult {
					t.Errorf("unexpected result: %s, expected: %s", result, test.expectedResult)
				}
			}
			intercepted := obj.interceptedCalls()
			if len(intercepted) != 1 {
				t.Fatalf("unexpected number of intercepted calls: %v", intercepted)
			}
			if intercepted[0] != e.Identifier() {
				t.Errorf("unexpected intercepted call: %v, expected: %v", intercepted, e.Identifier())
			}
		})
	}
}
