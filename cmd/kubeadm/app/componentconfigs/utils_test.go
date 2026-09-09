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

package componentconfigs

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestUnsupportedConfigVersionsError(t *testing.T) {
	tests := []struct {
		name         string
		errs         UnsupportedConfigVersionsErrorMap
		wantError    string
		wantErrorMap string
	}{
		{
			name: "only one error",
			errs: UnsupportedConfigVersionsErrorMap{
				"foo": &UnsupportedConfigVersionError{
					OldVersion: schema.GroupVersion{
						Group:   "admissionregistration.k8s.io",
						Version: "v1beta1",
					},
					CurrentVersion: schema.GroupVersion{
						Group:   "admissionregistration.k8s.io",
						Version: "v1",
					},
				},
			},
			wantError:    "unsupported apiVersion \"admissionregistration.k8s.io/v1beta1\", you may have to do manual conversion to \"admissionregistration.k8s.io/v1\" and run kubeadm again",
			wantErrorMap: "multiple unsupported config version errors encountered:\n\t- unsupported apiVersion \"admissionregistration.k8s.io/v1beta1\", you may have to do manual conversion to \"admissionregistration.k8s.io/v1\" and run kubeadm again",
		},
		{
			name: "multiple errors",
			errs: UnsupportedConfigVersionsErrorMap{
				"err1": &UnsupportedConfigVersionError{
					OldVersion: schema.GroupVersion{
						Group:   "admissionregistration.k8s.io",
						Version: "v1beta1",
					},
					CurrentVersion: schema.GroupVersion{
						Group:   "admissionregistration.k8s.io",
						Version: "v1",
					},
				},
				"err2": &UnsupportedConfigVersionError{
					OldVersion: schema.GroupVersion{
						Group:   "node.k8s.io",
						Version: "v1beta1",
					},
					CurrentVersion: schema.GroupVersion{
						Group:   "node.k8s.io",
						Version: "v1",
					},
				},
			},
			wantErrorMap: "multiple unsupported config version errors encountered:" +
				"\n\t- unsupported apiVersion \"admissionregistration.k8s.io/v1beta1\", you may have to do manual conversion to \"admissionregistration.k8s.io/v1\" and run kubeadm again" +
				"\n\t- unsupported apiVersion \"node.k8s.io/v1beta1\", you may have to do manual conversion to \"node.k8s.io/v1\" and run kubeadm again",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.errs.Error(); got != tt.wantErrorMap {
				t.Errorf("UnsupportedConfigVersionsErrorMap.Error() = %v\n, want %v", got, tt.wantErrorMap)
			}
			if tt.wantError != "" {
				for _, err := range tt.errs {
					if got := err.Error(); got != tt.wantError {
						t.Errorf("UnsupportedConfigVersionError.Error() = %v\n, want %v", got, tt.wantError)
						break
					}
				}
			}
		})
	}
}
