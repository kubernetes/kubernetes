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
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// MockCustomEncoder is used to test CustomEncoder interface.
// +k8s:deepcopy-gen=false
type MockCustomEncoder struct {
	GVK            schema.GroupVersionKind
	ExpectedResult string
	ExpectedError  error

	interceptedCalls []runtime.WithVersionEncoder
}

func (m *MockCustomEncoder) DeepCopyObject() runtime.Object {
	panic("DeepCopyObject unimplemented for mockCustomEncoder")
}

func (m *MockCustomEncoder) GetObjectKind() schema.ObjectKind {
	return m
}

func (m *MockCustomEncoder) GroupVersionKind() schema.GroupVersionKind {
	return m.GVK
}

func (m *MockCustomEncoder) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	m.GVK = gvk
}

func (m *MockCustomEncoder) InterceptEncode(encoder runtime.WithVersionEncoder, w io.Writer) error {
	m.interceptedCalls = append(m.interceptedCalls, encoder)
	w.Write([]byte(m.ExpectedResult))
	return m.ExpectedError
}

func (m *MockCustomEncoder) InterceptedCalls() []runtime.WithVersionEncoder {
	return m.interceptedCalls
}
