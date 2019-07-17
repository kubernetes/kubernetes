/*
Copyright 2019 The Kubernetes Authors.

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

package v1beta1

import (
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestInternalToVersionedInitConfigurationConversion(t *testing.T) {
	testcases := map[string]struct {
		in            kubeadm.InitConfiguration
		expectedError bool
	}{
		"conversion succeeds": {
			in:            kubeadm.InitConfiguration{},
			expectedError: false,
		},
		"certificateKey set causes an error": {
			in: kubeadm.InitConfiguration{
				CertificateKey: "secret",
			},
			expectedError: true,
		},
		"ignorePreflightErrors set causes an error": {
			in: kubeadm.InitConfiguration{
				NodeRegistration: kubeadm.NodeRegistrationOptions{
					IgnorePreflightErrors: []string{"SomeUndesirableError"},
				},
			},
			expectedError: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			versioned := &InitConfiguration{}
			err := Convert_kubeadm_InitConfiguration_To_v1beta1_InitConfiguration(&tc.in, versioned, nil)
			if err == nil && tc.expectedError {
				t.Error("unexpected success")
			} else if err != nil && !tc.expectedError {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestInternalToVersionedJoinConfigurationConversion(t *testing.T) {
	testcases := map[string]struct {
		in            kubeadm.JoinConfiguration
		expectedError bool
	}{
		"conversion succeeds": {
			in:            kubeadm.JoinConfiguration{},
			expectedError: false,
		},
		"ignorePreflightErrors set causes an error": {
			in: kubeadm.JoinConfiguration{
				NodeRegistration: kubeadm.NodeRegistrationOptions{
					IgnorePreflightErrors: []string{"SomeUndesirableError"},
				},
			},
			expectedError: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			versioned := &JoinConfiguration{}
			err := Convert_kubeadm_JoinConfiguration_To_v1beta1_JoinConfiguration(&tc.in, versioned, nil)
			if err == nil && tc.expectedError {
				t.Error("unexpected success")
			} else if err != nil && !tc.expectedError {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestInternalToVersionedNodeRegistrationOptionsConversion(t *testing.T) {
	testcases := map[string]struct {
		in            kubeadm.NodeRegistrationOptions
		expectedError bool
	}{
		"conversion succeeds": {
			in:            kubeadm.NodeRegistrationOptions{},
			expectedError: false,
		},
		"ignorePreflightErrors set causes an error": {
			in: kubeadm.NodeRegistrationOptions{
				IgnorePreflightErrors: []string{"SomeUndesirableError"},
			},
			expectedError: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			versioned := &NodeRegistrationOptions{}
			err := Convert_kubeadm_NodeRegistrationOptions_To_v1beta1_NodeRegistrationOptions(&tc.in, versioned, nil)
			if err == nil && tc.expectedError {
				t.Error("unexpected success")
			} else if err != nil && !tc.expectedError {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestInternalToVersionedJoinControlPlaneConversion(t *testing.T) {
	testcases := map[string]struct {
		in            kubeadm.JoinControlPlane
		expectedError bool
	}{
		"conversion succeeds": {
			in:            kubeadm.JoinControlPlane{},
			expectedError: false,
		},
		"certificateKey set causes an error": {
			in: kubeadm.JoinControlPlane{
				CertificateKey: "secret",
			},
			expectedError: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			versioned := &JoinControlPlane{}
			err := Convert_kubeadm_JoinControlPlane_To_v1beta1_JoinControlPlane(&tc.in, versioned, nil)
			if err == nil && tc.expectedError {
				t.Error("unexpected success")
			} else if err != nil && !tc.expectedError {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
