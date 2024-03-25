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

package apiclient

import (
	"reflect"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestGetControlPlaneComponents(t *testing.T) {
	testcases := []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected []controlPlaneComponent
	}{
		{
			name: "port values from config",
			cfg: &kubeadmapi.ClusterConfiguration{
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "secure-port", Value: "1111"},
						},
					},
				},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "2222"},
					},
				},
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "3333"},
					},
				},
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: "https://127.0.0.1:1111/healthz"},
				{name: "kube-controller-manager", url: "https://127.0.0.1:2222/healthz"},
				{name: "kube-scheduler", url: "https://127.0.0.1:3333/healthz"},
			},
		},
		{
			name: "default ports",
			cfg:  &kubeadmapi.ClusterConfiguration{},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: "https://127.0.0.1:6443/healthz"},
				{name: "kube-controller-manager", url: "https://127.0.0.1:10257/healthz"},
				{name: "kube-scheduler", url: "https://127.0.0.1:10259/healthz"},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getControlPlaneComponents(tc.cfg)
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Fatalf("expected result: %+v, got: %+v", tc.expected, actual)
			}
		})
	}
}
