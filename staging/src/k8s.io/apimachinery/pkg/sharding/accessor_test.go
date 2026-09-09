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

package sharding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestResolveFieldValue(t *testing.T) {
	obj := &testObject{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID("test-uid-123"),
			Name:      "test-name",
			Namespace: "test-namespace",
		},
	}

	tests := []struct {
		fieldPath string
		want      string
		wantErr   bool
	}{
		{"object.metadata.uid", "test-uid-123", false},
		{"object.metadata.namespace", "test-namespace", false},
		{"object.metadata.name", "", true},
		{"object.metadata.labels", "", true},
		{"invalid.path", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.fieldPath, func(t *testing.T) {
			got, err := ResolveFieldValue(obj, tt.fieldPath)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error for fieldPath %q", tt.fieldPath)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("ResolveFieldValue(%q) = %q, want %q", tt.fieldPath, got, tt.want)
			}
		})
	}
}
