/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"errors"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type verifierCall struct {
	GVK schema.GroupVersionKind
	Err error
}

type mockVerifier struct {
	t             *testing.T
	expectedCalls map[schema.GroupVersionKind]error
}

func (m *mockVerifier) CheckExpectations() {
	if len(m.expectedCalls) != 0 {
		m.t.Errorf("Expected calls remaining: %v", m.expectedCalls)
	}
}

func (m *mockVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	returnErr, ok := m.expectedCalls[gvk]
	if !ok {
		m.t.Errorf("Unexpected HasSupport call with GVK=%v", gvk)
	}
	delete(m.expectedCalls, gvk)
	return returnErr
}

func TestCachingVerifier(t *testing.T) {
	gvk1 := schema.GroupVersionKind{
		Group:   "group",
		Version: "version",
		Kind:    "kind",
	}
	gvk2 := schema.GroupVersionKind{
		Group:   "group2",
		Version: "version2",
		Kind:    "kind2",
	}

	err1 := errors.New("some error")

	testCases := []struct {
		name                    string
		calls                   []verifierCall
		expectedUnderlyingCalls map[schema.GroupVersionKind]error
	}{
		{
			name: "return value is cached",
			calls: []verifierCall{
				{GVK: gvk1, Err: nil},
				{GVK: gvk1, Err: nil},
				{GVK: gvk1, Err: nil},
				{GVK: gvk2, Err: err1},
				{GVK: gvk2, Err: err1},
				{GVK: gvk2, Err: err1},
			},
			expectedUnderlyingCalls: map[schema.GroupVersionKind]error{
				gvk1: nil,
				gvk2: err1,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := &mockVerifier{
				t:             t,
				expectedCalls: tc.expectedUnderlyingCalls,
			}
			verifier := newCachingVerifier(m)

			for _, call := range tc.calls {
				err := verifier.HasSupport(call.GVK)
				if !errors.Is(err, call.Err) {
					t.Errorf("Expected error: %v, got: %v", call.Err, err)
				}
			}

			m.CheckExpectations()
		})
	}
}
