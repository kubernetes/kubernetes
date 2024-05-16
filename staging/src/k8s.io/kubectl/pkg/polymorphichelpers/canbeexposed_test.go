/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestCanBeExposed(t *testing.T) {
	tests := []struct {
		kind      schema.GroupKind
		expectErr bool
	}{
		{
			kind:      corev1.SchemeGroupVersion.WithKind("ReplicationController").GroupKind(),
			expectErr: false,
		},
		{
			kind:      corev1.SchemeGroupVersion.WithKind("Service").GroupKind(),
			expectErr: false,
		},
		{
			kind:      corev1.SchemeGroupVersion.WithKind("Pod").GroupKind(),
			expectErr: false,
		},
		{
			kind:      appsv1.SchemeGroupVersion.WithKind("Deployment").GroupKind(),
			expectErr: false,
		},
		{
			kind:      extensionsv1beta1.SchemeGroupVersion.WithKind("ReplicaSet").GroupKind(),
			expectErr: false,
		},
		{
			kind:      corev1.SchemeGroupVersion.WithKind("Node").GroupKind(),
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := canBeExposed(test.kind)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}
