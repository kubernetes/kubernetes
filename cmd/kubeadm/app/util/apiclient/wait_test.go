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
	"fmt"
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
			name: "port and addresses from config",
			cfg: &kubeadmapi.ClusterConfiguration{
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "secure-port", Value: "1111"},
							{Name: "advertise-address", Value: "fd00:1::"},
						},
					},
				},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "2222"},
						{Name: "bind-address", Value: "127.0.0.1"},
					},
				},
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "secure-port", Value: "3333"},
						{Name: "bind-address", Value: "127.0.0.1"},
					},
				},
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: fmt.Sprintf("https://[fd00:1::]:1111/%s", endpointLivez)},
				{name: "kube-controller-manager", url: fmt.Sprintf("https://127.0.0.1:2222/%s", endpointHealthz)},
				{name: "kube-scheduler", url: fmt.Sprintf("https://127.0.0.1:3333/%s", endpointLivez)},
			},
		},
		{
			name: "default ports and addresses",
			cfg:  &kubeadmapi.ClusterConfiguration{},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", url: fmt.Sprintf("https://192.168.0.1:6443/%s", endpointLivez)},
				{name: "kube-controller-manager", url: fmt.Sprintf("https://127.0.0.1:10257/%s", endpointHealthz)},
				{name: "kube-scheduler", url: fmt.Sprintf("https://127.0.0.1:10259/%s", endpointLivez)},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getControlPlaneComponents(tc.cfg, "192.168.0.1")
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Fatalf("expected result: %+v, got: %+v", tc.expected, actual)
			}
		})
	}
}
