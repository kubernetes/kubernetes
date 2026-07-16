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

package constants

import (
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestGetStaticPodDirectory(t *testing.T) {
	expected := filepath.FromSlash("/etc/kubernetes/manifests")
	actual := GetStaticPodDirectory()

	if actual != expected {
		t.Errorf(
			"failed GetStaticPodDirectory:\n\texpected: %s\n\t  actual: %s",
			expected,
			actual,
		)
	}
}

func TestGetAdminKubeConfigPath(t *testing.T) {
	expected := filepath.Join(KubernetesDir, AdminKubeConfigFileName)
	actual := GetAdminKubeConfigPath()

	if actual != expected {
		t.Errorf(
			"failed GetAdminKubeConfigPath:\n\texpected: %s\n\t  actual: %s",
			expected,
			actual,
		)
	}
}

func TestGetKubeletKubeConfigPath(t *testing.T) {
	expected := filepath.FromSlash("/etc/kubernetes/kubelet.conf")
	actual := GetKubeletKubeConfigPath()

	if actual != expected {
		t.Errorf(
			"failed GetKubeletKubeConfigPath:\n\texpected: %s\n\t  actual: %s",
			expected,
			actual,
		)
	}
}

func TestGetStaticPodFilepath(t *testing.T) {
	var tests = []struct {
		componentName, manifestsDir, expected string
	}{
		{
			componentName: "kube-apiserver",
			manifestsDir:  "/etc/kubernetes/manifests",
			expected:      "/etc/kubernetes/manifests/kube-apiserver.yaml",
		},
		{
			componentName: "kube-controller-manager",
			manifestsDir:  "/etc/kubernetes/manifests/",
			expected:      "/etc/kubernetes/manifests/kube-controller-manager.yaml",
		},
		{
			componentName: "foo",
			manifestsDir:  "/etc/bar/",
			expected:      "/etc/bar/foo.yaml",
		},
	}
	for _, rt := range tests {
		t.Run(rt.componentName, func(t *testing.T) {
			actual := GetStaticPodFilepath(rt.componentName, rt.manifestsDir)
			expected := filepath.FromSlash(rt.expected)
			if actual != expected {
				t.Errorf(
					"failed GetStaticPodFilepath:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestEtcdSupportedVersionLength(t *testing.T) {
	const max = 3
	if len(SupportedEtcdVersion) != max {
		t.Fatalf("SupportedEtcdVersion must include exactly %d versions", max)
	}
}

func TestEtcdSupportedVersion(t *testing.T) {
	var supportedEtcdVersion = map[uint8]string{
		17: "3.3.17-0",
		18: "3.4.2-0",
		19: "3.4.3-0",
	}
	var tests = []struct {
		kubernetesVersion string
		expectedVersion   *version.Version
		expectedWarning   bool
		expectedError     bool
	}{
		{
			kubernetesVersion: "1.x.1",
			expectedVersion:   nil,
			expectedWarning:   false,
			expectedError:     true,
		},
		{
			kubernetesVersion: "1.10.1",
			expectedVersion:   version.MustParseSemantic("3.3.17-0"),
			expectedWarning:   true,
			expectedError:     false,
		},
		{
			kubernetesVersion: "1.99.0",
			expectedVersion:   version.MustParseSemantic("3.4.3-0"),
			expectedWarning:   true,
			expectedError:     false,
		},
		{
			kubernetesVersion: "1.17.2",
			expectedVersion:   version.MustParseSemantic("3.3.17-0"),
			expectedWarning:   false,
			expectedError:     false,
		},
		{
			kubernetesVersion: "1.18.1",
			expectedVersion:   version.MustParseSemantic("3.4.2-0"),
			expectedWarning:   false,
			expectedError:     false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.kubernetesVersion, func(t *testing.T) {
			actualVersion, actualWarning, actualError := EtcdSupportedVersion(supportedEtcdVersion, rt.kubernetesVersion)
			if (actualError != nil) != rt.expectedError {
				t.Fatalf("expected error %v, got %v", rt.expectedError, actualError != nil)
			}
			if (actualWarning != nil) != rt.expectedWarning {
				t.Fatalf("expected warning %v, got %v", rt.expectedWarning, actualWarning != nil)
			}
			if actualError == nil && actualVersion.String() != rt.expectedVersion.String() {
				t.Errorf("expected version %s, got %s", rt.expectedVersion.String(), actualVersion.String())
			}
		})
	}
}

func TestGetKubernetesServiceCIDR(t *testing.T) {
	var tests = []struct {
		svcSubnetList string
		isDualStack   bool
		expected      string
		expectedError bool
		name          string
	}{
		{
			svcSubnetList: "192.168.10.0/24",
			expected:      "192.168.10.0/24",
			expectedError: false,
			name:          "valid: valid IPv4 range from single-stack",
		},
		{
			svcSubnetList: "fd03::/112",
			expected:      "fd03::/112",
			expectedError: false,
			name:          "valid: valid IPv6 range from single-stack",
		},
		{
			svcSubnetList: "192.168.10.0/24,fd03::/112",
			expected:      "192.168.10.0/24",
			expectedError: false,
			name:          "valid: valid <IPv4,IPv6> ranges from dual-stack",
		},
		{
			svcSubnetList: "fd03::/112,192.168.10.0/24",
			expected:      "fd03::/112",
			expectedError: false,
			name:          "valid: valid <IPv6,IPv4> ranges from dual-stack",
		},
		{
			svcSubnetList: "192.168.10.0/24,fd03:x::/112",
			expected:      "",
			expectedError: true,
			name:          "invalid: failed to parse subnet range for dual-stack",
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual, actualError := GetKubernetesServiceCIDR(rt.svcSubnetList)
			if rt.expectedError {
				if actualError == nil {
					t.Errorf("failed GetKubernetesServiceCIDR:\n\texpected error, but got no error")
				}
			} else if !rt.expectedError && actualError != nil {
				t.Errorf("failed GetKubernetesServiceCIDR:\n\texpected no error, but got: %v", actualError)
			} else {
				if actual.String() != rt.expected {
					t.Errorf(
						"failed GetKubernetesServiceCIDR:\n\texpected: %s\n\t  actual: %s",
						rt.expected,
						actual.String(),
					)
				}
			}
		})
	}
}

func TestGetSkewedKubernetesVersionImpl(t *testing.T) {
	tests := []struct {
		name           string
		versionInfo    *apimachineryversion.Info
		n              int
		expectedResult *version.Version
	}{
		{
			name:           "invalid versionInfo; placeholder version is returned",
			versionInfo:    &apimachineryversion.Info{},
			expectedResult: DefaultKubernetesPlaceholderVersion,
		},
		{
			name:           "valid skew of -1",
			versionInfo:    &apimachineryversion.Info{Major: "1", GitVersion: "v1.23.0"},
			n:              -1,
			expectedResult: version.MustParseSemantic("v1.22.0"),
		},
		{
			name:           "valid skew of 0",
			versionInfo:    &apimachineryversion.Info{Major: "1", GitVersion: "v1.23.0"},
			n:              0,
			expectedResult: version.MustParseSemantic("v1.23.0"),
		},
		{
			name:           "valid skew of +1",
			versionInfo:    &apimachineryversion.Info{Major: "1", GitVersion: "v1.23.0"},
			n:              1,
			expectedResult: version.MustParseSemantic("v1.24.0"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := getSkewedKubernetesVersionImpl(tc.versionInfo, tc.n)
			if cmp, _ := result.Compare(tc.expectedResult.String()); cmp != 0 {
				t.Errorf("expected result: %v, got %v", tc.expectedResult, result)
			}
		})
	}
}

func TestGetAPIServerVirtualIP(t *testing.T) {
	var tests = []struct {
		name, svcSubnet, expectedIP string
		expectedErr                 bool
	}{
		{
			name:        "subnet mask 24",
			svcSubnet:   "10.96.0.12/24",
			expectedIP:  "10.96.0.1",
			expectedErr: false,
		},
		{
			name:        "subnet mask 12",
			svcSubnet:   "10.96.0.0/12",
			expectedIP:  "10.96.0.1",
			expectedErr: false,
		},
		{
			name:        "subnet mask 26",
			svcSubnet:   "10.87.116.64/26",
			expectedIP:  "10.87.116.65",
			expectedErr: false,
		},
		{
			name:        "dual-stack ipv4 primary, subnet mask 26",
			svcSubnet:   "10.87.116.64/26,fd03::/112",
			expectedIP:  "10.87.116.65",
			expectedErr: false,
		},
		{
			name:        "dual-stack, subnet mask 26 , missing first ip segment",
			svcSubnet:   ",10.87.116.64/26",
			expectedErr: true,
		},
		{
			name:        "dual-stack ipv4 primary, subnet mask 26, missing second ip segment",
			svcSubnet:   "10.87.116.64/26,",
			expectedErr: true,
		},
		{
			name:        "dual-stack ipv6 primary, subnet mask 112",
			svcSubnet:   "fd03::/112,10.87.116.64/26",
			expectedIP:  "fd03::1",
			expectedErr: false,
		},
		{
			name:        "dual-stack, subnet mask 26, missing first ip segment",
			svcSubnet:   ",fd03::/112",
			expectedErr: true,
		},
		{
			name:        "dual-stack, subnet mask 26, missing second ip segment",
			svcSubnet:   "fd03::/112,",
			expectedErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			virtualIP, err := GetAPIServerVirtualIP(rt.svcSubnet)
			if (err != nil) != rt.expectedErr {
				t.Errorf("failed APIServerVirtualIP:\n\texpectedErr: %v, got: %v", rt.expectedErr, err)
			} else if !rt.expectedErr {
				if virtualIP.String() != rt.expectedIP {
					t.Errorf(
						"failed APIServerVirtualIP:\n\texpected: %s\n\t  actual: %s",
						rt.expectedIP,
						virtualIP.String(),
					)
				}
			}
		})
	}
}

func TestGetDNSIP(t *testing.T) {
	tests := []struct {
		name          string
		svcSubnetList string
		expected      string
		expectedError bool
	}{
		{
			name:          "valid IPv4 range from single-stack",
			svcSubnetList: "192.168.10.0/24",
			expected:      "192.168.10.10",
			expectedError: false,
		},
		{
			name:          "valid IPv6 range from single-stack",
			svcSubnetList: "fd03::/112",
			expected:      "fd03::a",
			expectedError: false,
		},
		{
			name:          "valid <IPv4,IPv6> ranges from dual-stack",
			svcSubnetList: "192.168.10.0/24,fd03::/112",
			expected:      "192.168.10.10",
			expectedError: false,
		},
		{
			name:          "valid <IPv6,IPv4> ranges from dual-stack",
			svcSubnetList: "fd03::/112,192.168.10.0/24",
			expected:      "fd03::a",
			expectedError: false,
		},
		{
			name:          "invalid subnet range from dual-stack",
			svcSubnetList: "192.168.10.0/24,fd03:x::/112",
			expected:      "",
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual, actualError := GetDNSIP(tt.svcSubnetList)
			if tt.expectedError {
				if actualError == nil {
					t.Errorf("failed GetDNSIP:\n\texpected error, but got no error")
				}
			} else if !tt.expectedError && actualError != nil {
				t.Errorf("failed GetDNSIP:\n\texpected no error, but got: %v", actualError)
			} else {
				if actual.String() != tt.expected {
					t.Errorf(
						"failed GetDNSIP:\n\texpected: %s\n\t  actual: %s",
						tt.expected,
						actual.String(),
					)
				}
			}
		})
	}
}
