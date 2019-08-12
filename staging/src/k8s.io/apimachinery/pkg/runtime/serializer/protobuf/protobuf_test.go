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
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serializertesting "k8s.io/apimachinery/pkg/runtime/serializer/testing"
)

func TestInterceptEncode(t *testing.T) {
	testCases := []struct {
		desc           string
		expectedResult string
		expectedError  error
	}{
		{
			desc:           "success",
			expectedResult: "{}",
			expectedError:  nil,
		},
		{
			desc:           "failure",
			expectedResult: "",
			expectedError:  fmt.Errorf("failure"),
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			gvk := schema.GroupVersionKind{Group: "group", Version: "version", Kind: "MockCustomEncoder"}
			creater := &mockCreater{obj: &serializertesting.MockCustomEncoder{}}
			typer := &mockTyper{gvk: &gvk}

			serializers := []runtime.Encoder{
				NewSerializer(creater, typer),
				NewRawSerializer(creater, typer),
			}
			for _, serializer := range serializers {
				writer := bytes.NewBuffer(nil)

				obj := &serializertesting.MockCustomEncoder{
					GVK:            gvk,
					ExpectedResult: test.expectedResult,
					ExpectedError:  test.expectedError,
				}
				if err := serializer.Encode(obj, writer); err != test.expectedError {
					t.Errorf("unexpected error: %v, expected: %v", err, test.expectedError)
				}
				if result := writer.String(); result != test.expectedResult {
					t.Errorf("unexpected result: %v, expected: %v", result, test.expectedResult)
				}
				intercepted := obj.InterceptedCalls()
				if len(intercepted) != 1 {
					t.Fatalf("unexpected number of intercepted calls: %v", intercepted)
				}
				if intercepted[0].Encoder != serializer ||
					intercepted[0].ObjectConvertor != nil ||
					intercepted[0].ObjectTyper != nil ||
					intercepted[0].Version != nil {
					t.Errorf("unexpected intercepted encoder: %v", intercepted[0])
				}
			}
		})
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
