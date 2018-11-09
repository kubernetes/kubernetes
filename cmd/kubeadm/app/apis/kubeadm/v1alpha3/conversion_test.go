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

package v1alpha3_test

import (
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestJoinConfigurationConversion(t *testing.T) {
	testcases := map[string]struct {
		old         *v1alpha3.JoinConfiguration
		expectedErr string
	}{
		"conversion succeeds": {
			old:         &v1alpha3.JoinConfiguration{},
			expectedErr: "",
		},
		"feature gates fails to be converted": {
			old: &v1alpha3.JoinConfiguration{
				FeatureGates: map[string]bool{
					"someGate": true,
				},
			},
			expectedErr: "featureGates has been removed from JoinConfiguration and featureGates from ClusterConfiguration will be used instead. Please cleanup JoinConfiguration.FeatureGates fields",
		},
	}
	for _, tc := range testcases {
		internal := &kubeadm.JoinConfiguration{}
		err := scheme.Scheme.Convert(tc.old, internal, nil)
		if len(tc.expectedErr) != 0 {
			testutil.AssertError(t, err, tc.expectedErr)
		} else if err != nil {
			t.Errorf("no error was expected but '%s' was found", err)
		}
	}
}
