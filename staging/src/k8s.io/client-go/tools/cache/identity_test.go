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

// Package cache is a client-side caching mechanism. It is useful for
// reducing the number of server calls you'd otherwise need to make.
// Reflector watches a server and updates a Store. Two stores are provided;
// one that simply caches objects (for example, to allow a scheduler to
// list currently available nodes), and one that additionally acts as
// a FIFO queue (for example, to allow a scheduler to process incoming
// pods).
package cache

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestIdentifierUniqueness(t *testing.T) {
	tests := []struct {
		name       string
		setup      func() // create identifiers before the test
		idName     string
		obj        runtime.Object
		wantUnique bool
		wantErr    bool
	}{
		{
			name:       "empty name returns error",
			setup:      func() {},
			idName:     "",
			obj:        &v1.Pod{},
			wantUnique: false,
			wantErr:    true,
		},
		{
			name:       "first identifier with name is unique",
			setup:      func() {},
			idName:     "my-fifo",
			obj:        &v1.Pod{},
			wantUnique: true,
			wantErr:    false,
		},
		{
			name: "same name different itemType is unique",
			setup: func() {
				_, _ = NewIdentifier("my-fifo", &v1.Pod{})
			},
			idName:     "my-fifo",
			obj:        &v1.ConfigMap{},
			wantUnique: true,
			wantErr:    false,
		},
		{
			name: "different name same itemType is unique",
			setup: func() {
				_, _ = NewIdentifier("fifo-1", &v1.Pod{})
			},
			idName:     "fifo-2",
			obj:        &v1.Pod{},
			wantUnique: true,
			wantErr:    false,
		},
		{
			name: "duplicate name+itemType returns error",
			setup: func() {
				_, _ = NewIdentifier("my-fifo", &v1.Pod{})
			},
			idName:     "my-fifo",
			obj:        &v1.Pod{},
			wantUnique: false,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetIdentity()
			tt.setup()

			id, err := NewIdentifier(tt.idName, tt.obj)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewIdentifier() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got := id.IsUnique(); got != tt.wantUnique {
				t.Errorf("IsUnique() = %v, want %v", got, tt.wantUnique)
			}
		})
	}
}
