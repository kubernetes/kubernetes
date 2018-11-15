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
		name             string
		cfg              *kubeadmapi.InitConfiguration
		expectedEndpoint string
		expectedError    bool
	}{
		{
			name: "use ControlPlaneEndpoint (dns) if fully defined",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "cp.k8s.io:1234",
				},
			},
			expectedEndpoint: "https://cp.k8s.io:1234",
		},
		{
			name: "use ControlPlaneEndpoint (ipv4) if fully defined",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "1.2.3.4:1234",
				},
			},
			expectedEndpoint: "https://1.2.3.4:1234",
		},
		{
			name: "use ControlPlaneEndpoint (ipv6) if fully defined",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "[2001:db8::1]:1234",
				},
			},
			expectedEndpoint: "https://[2001:db8::1]:1234",
		},
		{
			name: "use ControlPlaneEndpoint (dns) + BindPort if ControlPlaneEndpoint defined without port",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{

					ControlPlaneEndpoint: "cp.k8s.io",
				},
			},
			expectedEndpoint: "https://cp.k8s.io:4567",
		},
		{
			name: "use ControlPlaneEndpoint (ipv4) + BindPort if ControlPlaneEndpoint defined without port",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "1.2.3.4",
				},
			},
			expectedEndpoint: "https://1.2.3.4:4567",
		},
		{
			name: "use ControlPlaneEndpoint (ipv6) + BindPort if ControlPlaneEndpoint defined without port",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{

					ControlPlaneEndpoint: "2001:db8::1",
				},
			},
			expectedEndpoint: "https://[2001:db8::1]:4567",
		},
		{
			name: "use AdvertiseAddress (ipv4) + BindPort if ControlPlaneEndpoint is not defined",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "4.5.6.7",
				},
			},
			expectedEndpoint: "https://4.5.6.7:4567",
		},
		{
			name: "use AdvertiseAddress (ipv6) + BindPort if ControlPlaneEndpoint is not defined",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort:         4567,
					AdvertiseAddress: "2001:db8::1",
				},
			},
			expectedEndpoint: "https://[2001:db8::1]:4567",
		},
		{
			name: "fail if invalid BindPort",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					BindPort: 0,
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid ControlPlaneEndpoint (dns)",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "bad!!.cp.k8s.io",
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid ControlPlaneEndpoint (ip4)",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "1..0",
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid ControlPlaneEndpoint (ip6)",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "1200::AB00:1234::2552:7777:1313",
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid ControlPlaneEndpoint (port)",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "cp.k8s.io:0",
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid AdvertiseAddress (ip4)",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1..0",
					BindPort:         4567,
				},
			},
			expectedError: true,
		},
		{
			name: "fail if invalid AdvertiseAddress (ip6)",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1200::AB00:1234::2552:7777:1313",
					BindPort:         4567,
				},
			},
			expectedError: true,
		},
	}

	for _, rt := range tests {
		actualEndpoint, actualError := GetMasterEndpoint(rt.cfg)

		if (actualError != nil) && !rt.expectedError {
			t.Errorf("%s unexpected failure: %v", rt.name, actualError)
			continue
		} else if (actualError == nil) && rt.expectedError {
			t.Errorf("%s passed when expected to fail", rt.name)
			continue
		}

		if actualEndpoint != rt.expectedEndpoint {
			t.Errorf("%s returned invalid endpoint %s, expected %s", rt.name, actualEndpoint, rt.expectedEndpoint)
		}
	}
}

func TestParseHostPort(t *testing.T) {

	var tests = []struct {
		name          string
		hostport      string
		expectedHost  string
		expectedPort  string
		expectedError bool
	}{
		{
			name:         "valid dns",
			hostport:     "cp.k8s.io",
			expectedHost: "cp.k8s.io",
			expectedPort: "",
		},
		{
			name:         "valid dns:port",
			hostport:     "cp.k8s.io:1234",
			expectedHost: "cp.k8s.io",
			expectedPort: "1234",
		},
		{
			name:         "valid ip4",
			hostport:     "1.2.3.4",
			expectedHost: "1.2.3.4",
			expectedPort: "",
		},
		{
			name:         "valid ipv4:port",
			hostport:     "1.2.3.4:1234",
			expectedHost: "1.2.3.4",
			expectedPort: "1234",
		},
		{
			name:         "valid ipv6",
			hostport:     "2001:db8::1",
			expectedHost: "2001:db8::1",
			expectedPort: "",
		},
		{
			name:         "valid ipv6:port",
			hostport:     "[2001:db8::1]:1234",
			expectedHost: "2001:db8::1",
			expectedPort: "1234",
		},
		{
			name:          "invalid port(not a number)",
			hostport:      "cp.k8s.io:aaa",
			expectedError: true,
		},
		{
			name:          "invalid port(out of range, positive port number)",
			hostport:      "cp.k8s.io:987654321",
			expectedError: true,
		},
		{
			name:          "invalid port(out of range, negative port number)",
			hostport:      "cp.k8s.io:-987654321",
			expectedError: true,
		},
		{
			name:          "invalid port(out of range, negative port number)",
			hostport:      "cp.k8s.io:123:123",
			expectedError: true,
		},
		{
			name:          "invalid dns",
			hostport:      "bad!!cp.k8s.io",
			expectedError: true,
		},
		{
			name:          "invalid valid dns:port",
			hostport:      "bad!!cp.k8s.io:1234",
			expectedError: true,
		},
		{
			name:         "invalid ip4, but valid DNS",
			hostport:     "259.2.3.4",
			expectedHost: "259.2.3.4",
		},
		{
			name:          "invalid ip4",
			hostport:      "1..3.4",
			expectedError: true,
		},
		{
			name:          "invalid ip4(2):port",
			hostport:      "1..3.4:1234",
			expectedError: true,
		},
		{
			name:          "invalid ipv6",
			hostport:      "1200::AB00:1234::2552:7777:1313",
			expectedError: true,
		},
		{
			name:          "invalid ipv6:port",
			hostport:      "[1200::AB00:1234::2552:7777:1313]:1234",
			expectedError: true,
		},
	}

	for _, rt := range tests {
		actualHost, actualPort, actualError := ParseHostPort(rt.hostport)

		if (actualError != nil) && !rt.expectedError {
			t.Errorf("%s unexpected failure: %v", rt.name, actualError)
			continue
		} else if (actualError == nil) && rt.expectedError {
			t.Errorf("%s passed when expected to fail", rt.name)
			continue
		}

		if actualHost != rt.expectedHost {
			t.Errorf("%s returned invalid host %s, expected %s", rt.name, actualHost, rt.expectedHost)
			continue
		}

		if actualPort != rt.expectedPort {
			t.Errorf("%s returned invalid port %s, expected %s", rt.name, actualPort, rt.expectedPort)
		}
	}
}

func TestParsePort(t *testing.T) {

	var tests = []struct {
		name          string
		port          string
		expectedPort  int
		expectedError bool
	}{
		{
			name:         "valid port",
			port:         "1234",
			expectedPort: 1234,
		},
		{
			name:          "invalid port (not a number)",
			port:          "a",
			expectedError: true,
		},
		{
			name:          "invalid port (<1)",
			port:          "-10",
			expectedError: true,
		},
		{
			name:          "invalid port (>65535)",
			port:          "66535",
			expectedError: true,
		},
	}

	for _, rt := range tests {
		actualPort, actualError := ParsePort(rt.port)

		if (actualError != nil) && !rt.expectedError {
			t.Errorf("%s unexpected failure: %v", rt.name, actualError)
			continue
		} else if (actualError == nil) && rt.expectedError {
			t.Errorf("%s passed when expected to fail", rt.name)
			continue
		}

		if actualPort != rt.expectedPort {
			t.Errorf("%s returned invalid port %d, expected %d", rt.name, actualPort, rt.expectedPort)
		}
	}
}
