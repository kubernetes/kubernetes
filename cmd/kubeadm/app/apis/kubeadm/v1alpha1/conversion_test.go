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

package v1alpha1_test

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
)

func TestUpgradeBootstrapTokens(t *testing.T) {
	testcases := []struct {
		name        string
		in          *v1alpha1.MasterConfiguration
		expectedOut *kubeadm.MasterConfiguration
		expectError bool
	}{
		{
			name: "empty configs should create at least one token",
			in:   &v1alpha1.MasterConfiguration{},
			expectedOut: &kubeadm.MasterConfiguration{
				BootstrapTokens: []kubeadm.BootstrapToken{
					{
						Token: nil,
					},
				},
			},
			expectError: false,
		},
		{
			name: "fail at parsing incoming token",
			in: &v1alpha1.MasterConfiguration{
				Token: "some fake token",
			},
			expectError: true,
		},
		{
			name: "input has values",
			in: &v1alpha1.MasterConfiguration{
				Token: "abcdef.abcdefghijklmnop",
				TokenTTL: &metav1.Duration{
					Duration: time.Duration(10 * time.Hour),
				},
				TokenUsages: []string{"action"},
				TokenGroups: []string{"group", "group2"},
			},
			expectedOut: &kubeadm.MasterConfiguration{
				BootstrapTokens: []kubeadm.BootstrapToken{
					{
						Token: &kubeadm.BootstrapTokenString{
							ID:     "abcdef",
							Secret: "abcdefghijklmnop",
						},
						TTL: &metav1.Duration{
							Duration: time.Duration(10 * time.Hour),
						},
						Usages: []string{"action"},
						Groups: []string{"group", "group2"},
					},
				},
			},
			expectError: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			out := &kubeadm.MasterConfiguration{}
			err := v1alpha1.UpgradeBootstrapTokens(tc.in, out)

			if tc.expectError {
				if err == nil {
					t.Fatal("expected an error but did not get one.")
				}
				// do not continue if we got an expected error
				return
			}

			if !reflect.DeepEqual(out.BootstrapTokens, tc.expectedOut.BootstrapTokens) {
				t.Fatalf("\nexpected: %v\ngot: %v", tc.expectedOut.BootstrapTokens, out.BootstrapTokens)
			}
		})
	}

}
