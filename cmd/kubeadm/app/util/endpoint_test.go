/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestGetMasterEndpoint(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.MasterConfiguration
		endpoint string
		expected bool
	}{
		{
			name: "valid IPv4 endpoint",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			},
			endpoint: "https://1.2.3.4:1234",
			expected: true,
		},
		{
			name: "valid IPv6 endpoint",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "2001:db8::1",
					BindPort:         4321,
				},
			},
			endpoint: "https://[2001:db8::1]:4321",
			expected: true,
		},
		{
			name: "invalid IPv4 endpoint",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			},
			endpoint: "https://[1.2.3.4]:1234",
			expected: false,
		},
		{
			name: "invalid IPv6 endpoint",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "2001:db8::1",
					BindPort:         4321,
				},
			},
			endpoint: "https://2001:db8::1:4321",
			expected: false,
		},
		{
			name: "invalid IPv4 AdvertiseAddress",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.34",
					BindPort:         1234,
				},
			},
			endpoint: "https://1.2.3.4:1234",
			expected: false,
		},
		{
			name: "invalid IPv6 AdvertiseAddress",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "2001::db8::1",
					BindPort:         4321,
				},
			},
			endpoint: "https://[2001:db8::1]:4321",
			expected: false,
		},
	}
	for _, rt := range tests {
		actual, err := GetMasterEndpoint(rt.cfg)
		if err != nil && rt.expected {
			t.Error(err)
		}
		if actual != rt.endpoint && rt.expected {
			t.Errorf(
				"%s test case failed:\n\texpected: %s\n\t actual: %s",
				rt.name,
				rt.endpoint,
				(actual),
			)
		}
	}
}

func TestGetMasterHostPort(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.MasterConfiguration
		hostPort string
		expected bool
	}{
		{
			name: "valid IPv4 master host and port",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			},
			hostPort: "1.2.3.4:1234",
			expected: true,
		},
		{
			name: "valid IPv6 master host port",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "2001:db8::1",
					BindPort:         4321,
				},
			},
			hostPort: "[2001:db8::1]:4321",
			expected: true,
		},
		{
			name: "invalid IPv4 address",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.34",
					BindPort:         1234,
				},
			},
			hostPort: "1.2.3.4:1234",
			expected: false,
		},
		{
			name: "invalid IPv6 address",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "2001::db8::1",
					BindPort:         4321,
				},
			},
			hostPort: "[2001:db8::1]:4321",
			expected: false,
		},
		{
			name: "invalid TCP port number",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         987654321,
				},
			},
			hostPort: "1.2.3.4:987654321",
			expected: false,
		},
		{
			name: "invalid negative TCP port number",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         -987654321,
				},
			},
			hostPort: "1.2.3.4:-987654321",
			expected: false,
		},
		{
			name: "unspecified IPv4 TCP port",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1.2.3.4",
				},
			},
			hostPort: "1.2.3.4:0",
			expected: true,
		},
		{
			name: "unspecified IPv6 TCP port",
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					AdvertiseAddress: "1:2:3::4",
				},
			},
			hostPort: "[1:2:3::4]:0",
			expected: true,
		},
	}
	for _, rt := range tests {
		actual, err := GetMasterHostPort(rt.cfg)
		if err != nil && rt.expected {
			t.Error(err)
		}
		if actual != rt.hostPort && rt.expected {
			t.Errorf(
				"%s test case failed:\n\texpected: %s\n\t actual: %s",
				rt.name,
				rt.hostPort,
				(actual),
			)
		}
	}
}
