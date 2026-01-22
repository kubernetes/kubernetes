/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNormalizeConfigMap(t *testing.T) {
	tests := []struct {
		name     string
		data     runtime.Object
		expected runtime.Object
	}{
		{
			name:     "nil ConfigMap",
			expected: nil,
		},
		{
			name: "strips managed fields",
			data: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:          "test",
					Namespace:     "default",
					ManagedFields: []metav1.ManagedFieldsEntry{{Manager: "fake"}},
				},
				Data: map[string]string{"key": "value"},
			},
			expected: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Data: map[string]string{"key": "value"},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			normalized := StripMetadata[runtime.Object](tt.data)

			if !reflect.DeepEqual(normalized, tt.expected) {
				t.Errorf("StripMetadata() = %v, want %v", normalized, tt.expected)
			}
		})
	}
}
