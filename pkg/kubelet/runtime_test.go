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

package kubelet

import (
	"reflect"
	"testing"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestRuntimeStateSetRuntimeHandlersSortsAndCopies(t *testing.T) {
	testCases := []struct {
		name     string
		handlers []kubecontainer.RuntimeHandler
		expected []string
	}{
		{
			name: "unsortedWithDefault",
			handlers: []kubecontainer.RuntimeHandler{
				{Name: "runc"},
				{Name: ""},
				{Name: "crun"},
			},
			expected: []string{"", "crun", "runc"},
		},
		{
			name: "alreadySorted",
			handlers: []kubecontainer.RuntimeHandler{
				{Name: ""},
				{Name: "crun"},
				{Name: "runc"},
			},
			expected: []string{"", "crun", "runc"},
		},
		{
			name:     "emptyHandlers",
			handlers: nil,
			expected: []string{},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			state := &runtimeState{}
			input := append([]kubecontainer.RuntimeHandler(nil), tt.handlers...)
			original := append([]kubecontainer.RuntimeHandler(nil), input...)

			state.setRuntimeHandlers(input)

			if !reflect.DeepEqual(input, original) {
				t.Fatalf("setRuntimeHandlers mutated input slice: got %#v, want %#v", input, original)
			}

			got := state.runtimeHandlers()
			if len(got) != len(tt.expected) {
				t.Fatalf("unexpected handler count: got %d, want %d", len(got), len(tt.expected))
			}
			for i, name := range tt.expected {
				if got[i].Name != name {
					t.Errorf("unexpected handler order at %d: got %q, want %q", i, got[i].Name, name)
				}
			}
		})
	}
}
