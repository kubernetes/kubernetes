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
	"bytes"
	"testing"

	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDefaultObjectPauser(t *testing.T) {
	tests := []struct {
		object    runtime.Object
		expect    []byte
		expectErr bool
	}{
		{
			object: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Paused: false,
				},
			},
			expect:    []byte(`paused":true`),
			expectErr: false,
		},
		{
			object: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Paused: true,
				},
			},
			expectErr: true,
		},
		{
			object:    &extensionsv1beta1.ReplicaSet{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		actual, err := defaultObjectPauser(test.object)
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !bytes.Contains(actual, test.expect) {
			t.Errorf("expected %s, but got %s", test.expect, actual)
		}
	}
}
