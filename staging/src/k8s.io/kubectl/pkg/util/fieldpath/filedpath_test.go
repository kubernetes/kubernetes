/*
Copyright 2022 The Kubernetes Authors.

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

package fieldpath

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestExtractFieldPathAsString(t *testing.T) {
	tests := []struct {
		name         string
		obj          interface{}
		fieldPath    string
		expectResult string
		expectErr    bool
	}{
		{
			name:         "Invalid type",
			obj:          &corev1.PodList{},
			expectResult: "",
			expectErr:    false,
		},
		{
			name: "Invalid fieldPath",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"myKey": "myValue"},
				},
			},
			fieldPath: "metadata.annotation['myKey']",
			expectErr: true,
		},
		{
			name: "Extract annotations field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"myKey": "myValue"},
				},
			},
			fieldPath:    "metadata.annotations",
			expectResult: "myKey=\"myValue\"",
			expectErr:    false,
		},
		{
			name: "Extract labels field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"myKey": "myValue"},
				},
			},
			fieldPath:    "metadata.labels",
			expectResult: "myKey=\"myValue\"",
			expectErr:    false,
		},
		{
			name: "Extract name field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "name",
				},
			},
			fieldPath:    "metadata.name",
			expectResult: "name",
			expectErr:    false,
		},
		{
			name: "Extract namespace field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "namespace",
				},
			},
			fieldPath:    "metadata.namespace",
			expectResult: "namespace",
			expectErr:    false,
		},
		{
			name: "Extract uid field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: "uid",
				},
			},
			fieldPath:    "metadata.uid",
			expectResult: "uid",
			expectErr:    false,
		},
		{
			name:      "Invalid fieldPath",
			obj:       &corev1.Pod{},
			fieldPath: "metadata.ui",
			expectErr: true,
		},
		{
			name: "Extract annotations field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"myKey": "myValue"},
				},
			},
			fieldPath:    "metadata.annotations['myKey']",
			expectResult: "myValue",
			expectErr:    false,
		},
		{
			name: "Extract label field as string",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"myLabel": "myValue"},
				},
			},
			fieldPath:    "metadata.labels['myLabel']",
			expectResult: "myValue",
			expectErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractFieldPathAsString(tt.obj, tt.fieldPath)
			if tt.expectErr && err == nil {
				t.Error("expected return error,but got nil")
			}
			if !tt.expectErr && err != nil {
				t.Error("expected return error is nil ,but got not nil")
			}
			if tt.expectResult != got {
				t.Errorf("%v: expected %v; got %v", tt.name, tt.expectResult, got)
			}
		})
	}
}
