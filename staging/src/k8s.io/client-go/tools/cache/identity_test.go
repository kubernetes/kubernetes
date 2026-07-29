/*
Copyright The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	podsGVR       = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	configMapsGVR = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}
)

func TestNewInformerName(t *testing.T) {
	tests := []struct {
		name    string
		setup   func()
		inName  string
		wantErr bool
	}{
		{
			name:    "first name is unique",
			setup:   func() {},
			inName:  "my-informer",
			wantErr: false,
		},
		{
			name: "duplicate name returns error",
			setup: func() {
				_, _ = NewInformerName("my-informer")
			},
			inName:  "my-informer",
			wantErr: true,
		},
		{
			name:    "empty name returns error",
			setup:   func() {},
			inName:  "",
			wantErr: true,
		},
		{
			name: "different name is unique",
			setup: func() {
				_, _ = NewInformerName("informer-1")
			},
			inName:  "informer-2",
			wantErr: false,
		},
		{
			name: "released name can be reused",
			setup: func() {
				n, _ := NewInformerName("my-informer")
				n.Release()
			},
			inName:  "my-informer",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ResetInformerNamesForTesting()
			tt.setup()

			_, err := NewInformerName(tt.inName)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewInformerName() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestWithResource(t *testing.T) {
	tests := []struct {
		name         string
		setup        func() *InformerName
		gvr          schema.GroupVersionResource
		wantReserved bool
	}{
		{
			name: "first GVR is reserved",
			setup: func() *InformerName {
				n, _ := NewInformerName("my-informer")
				return n
			},
			gvr:          podsGVR,
			wantReserved: true,
		},
		{
			name: "same GVR second time is not reserved",
			setup: func() *InformerName {
				n, _ := NewInformerName("my-informer")
				_ = n.WithResource(podsGVR)
				return n
			},
			gvr:          podsGVR,
			wantReserved: false,
		},
		{
			name: "different GVR is reserved",
			setup: func() *InformerName {
				n, _ := NewInformerName("my-informer")
				_ = n.WithResource(podsGVR)
				return n
			},
			gvr:          configMapsGVR,
			wantReserved: true,
		},
		{
			name: "nil InformerName returns not reserved",
			setup: func() *InformerName {
				return nil
			},
			gvr:          podsGVR,
			wantReserved: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ResetInformerNamesForTesting()
			n := tt.setup()

			id := n.WithResource(tt.gvr)
			if got := id.Reserved(); got != tt.wantReserved {
				t.Errorf("Reserved() = %v, want %v", got, tt.wantReserved)
			}
		})
	}
}

func TestRelease(t *testing.T) {
	ResetInformerNamesForTesting()

	n, err := NewInformerName("my-informer")
	if err != nil {
		t.Fatalf("NewInformerName() error = %v", err)
	}

	// Get a reserved identifier
	id := n.WithResource(podsGVR)
	if !id.Reserved() {
		t.Error("Expected Reserved() = true before Release()")
	}

	// Release the name
	n.Release()

	// The identifier should no longer be reserved
	if id.Reserved() {
		t.Error("Expected Reserved() = false after Release()")
	}

	// Should be able to reuse the name
	n2, err := NewInformerName("my-informer")
	if err != nil {
		t.Errorf("NewInformerName() after Release() error = %v", err)
	}

	// New identifier from new name should be reserved
	id2 := n2.WithResource(podsGVR)
	if !id2.Reserved() {
		t.Error("Expected Reserved() = true for new InformerName after Release()")
	}
}
