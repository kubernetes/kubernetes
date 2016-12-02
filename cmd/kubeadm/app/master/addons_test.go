/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestCreateKubeProxyPodSpec(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			cfg:      &kubeadmapi.MasterConfiguration{},
			expected: true,
		},
	}

	for _, rt := range tests {
		actual := createKubeProxyPodSpec(rt.cfg)
		if (actual.Containers[0].Name != "") != rt.expected {
			t.Errorf(
				"failed createKubeProxyPodSpec:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual.Containers[0].Name != ""),
			)
		}
	}
}

func TestCreateKubeDNSPodSpec(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{DNSDomain: "localhost"},
			},
			expected: "--domain=localhost",
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{DNSDomain: "foo"},
			},
			expected: "--domain=foo",
		},
	}

	for _, rt := range tests {
		actual := createKubeDNSPodSpec(rt.cfg)
		if actual.Containers[0].Args[0] != rt.expected {
			t.Errorf(
				"failed createKubeDNSPodSpec:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Containers[0].Args[0],
			)
		}
	}
}

func TestCreateKubeDNSServiceSpec(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "foo"},
			},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/1"},
			},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/24"},
			},
			expected: true,
		},
	}

	for _, rt := range tests {
		_, actual := createKubeDNSServiceSpec(rt.cfg)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed createKubeDNSServiceSpec:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
