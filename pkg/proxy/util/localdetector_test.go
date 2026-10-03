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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/nodemanager"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNoOpLocalDetector(t *testing.T) {
	localDetector := NewNoOpLocalDetector()
	if localDetector.IsImplemented() {
		t.Error("NoOpLocalDetector returns true for IsImplemented")
	}

	ifLocal := localDetector.IfLocal()
	if len(ifLocal) != 0 {
		t.Errorf("NoOpLocalDetector returns %v for IsLocal (expected nil)", ifLocal)
	}

	ifNotLocal := localDetector.IfNotLocal()
	if len(ifNotLocal) != 0 {
		t.Errorf("NoOpLocalDetector returns %v for IsNotLocal (expected nil)", ifNotLocal)
	}
}

func TestDetectLocalByCIDR(t *testing.T) {
	cases := []struct {
		cidr                     string
		expectedIfLocalOutput    []string
		expectedIfNotLocalOutput []string
	}{
		{
			cidr:                     "10.0.0.0/14",
			expectedIfLocalOutput:    []string{"-s", "10.0.0.0/14"},
			expectedIfNotLocalOutput: []string{"!", "-s", "10.0.0.0/14"},
		},
		{
			cidr:                     "2002:0:0:1234::/64",
			expectedIfLocalOutput:    []string{"-s", "2002:0:0:1234::/64"},
			expectedIfNotLocalOutput: []string{"!", "-s", "2002:0:0:1234::/64"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByCIDR(c.cidr)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByCIDR returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedIfLocalOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedIfLocalOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedIfNotLocalOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedIfNotLocalOutput, ifNotLocal)
		}
	}
}

func TestDetectLocalByBridgeInterface(t *testing.T) {
	cases := []struct {
		ifaceName               string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifaceName:               "eth0",
			expectedJumpIfOutput:    []string{"-i", "eth0"},
			expectedJumpIfNotOutput: []string{"!", "-i", "eth0"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByBridgeInterface(c.ifaceName)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByBridgeInterface returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}

func TestDetectLocalNFTByBridgeInterface(t *testing.T) {
	cases := []struct {
		ifaceName               string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifaceName:               "eth0",
			expectedJumpIfOutput:    []string{"iifname", "eth0"},
			expectedJumpIfNotOutput: []string{"iifname", "!=", "eth0"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByBridgeInterface(c.ifaceName)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByBridgeInterface returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocalNFT()
		ifNotLocal := localDetector.IfNotLocalNFT()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocalNFT, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocalNFT, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}

func TestDetectLocalByInterfaceNamePrefix(t *testing.T) {
	cases := []struct {
		ifacePrefix             string
		chain                   string
		args                    []string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifacePrefix:             "eth0",
			expectedJumpIfOutput:    []string{"-i", "eth0+"},
			expectedJumpIfNotOutput: []string{"!", "-i", "eth0+"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByInterfaceNamePrefix(c.ifacePrefix)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByInterfaceNamePrefix returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocal()
		ifNotLocal := localDetector.IfNotLocal()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocal, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocal, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}

func TestDetectLocalNFTByInterfaceNamePrefix(t *testing.T) {
	cases := []struct {
		ifacePrefix             string
		chain                   string
		args                    []string
		expectedJumpIfOutput    []string
		expectedJumpIfNotOutput []string
	}{
		{
			ifacePrefix:             "eth",
			expectedJumpIfOutput:    []string{"iifname", "eth*"},
			expectedJumpIfNotOutput: []string{"iifname", "!=", "eth*"},
		},
	}
	for _, c := range cases {
		localDetector := NewDetectLocalByInterfaceNamePrefix(c.ifacePrefix)
		if !localDetector.IsImplemented() {
			t.Error("DetectLocalByInterfaceNamePrefix returns false for IsImplemented")
		}

		ifLocal := localDetector.IfLocalNFT()
		ifNotLocal := localDetector.IfNotLocalNFT()

		if !reflect.DeepEqual(ifLocal, c.expectedJumpIfOutput) {
			t.Errorf("IfLocalNFT, expected: '%v', but got: '%v'", c.expectedJumpIfOutput, ifLocal)
		}

		if !reflect.DeepEqual(ifNotLocal, c.expectedJumpIfNotOutput) {
			t.Errorf("IfNotLocalNFT, expected: '%v', but got: '%v'", c.expectedJumpIfNotOutput, ifNotLocal)
		}
	}
}

type testNodeManager struct {
	nodemanager.Dummy

	primaryIPFamily v1.IPFamily
	podCIDRs        []string
}

func (n *testNodeManager) PrimaryIPFamily() v1.IPFamily {
	return n.primaryIPFamily
}

func (n *testNodeManager) PodCIDRs() []string {
	return n.podCIDRs
}

func TestGetLocalTrafficDetectors(t *testing.T) {
	cases := []struct {
		name            string
		config          *kubeproxyconfig.KubeProxyConfiguration
		primaryIPFamily v1.IPFamily
		nodePodCIDRs    []string
		expected        map[v1.IPFamily]LocalTrafficDetector
	}{
		// LocalModeClusterCIDR
		{
			name: "LocalModeClusterCIDR, single-stack IPv4 cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv6 cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv6 cluster with single-stack IPv4 config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			// This will output a warning that there is no IPv6 CIDR but it
			// will still use the provided IPv4 CIDR for IPv4.
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv4 cluster with single-stack IPv6 config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			// This will output a warning that there is no IPv4 CIDR but it
			// will still use the provided IPv6 CIDR for IPv6.
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, dual-stack IPv4-primary cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, dual-stack IPv6-primary cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, IPv4-primary kube-proxy / IPv6-primary config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, no ClusterCIDR",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{""},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		// LocalModeNodeCIDR
		{
			name: "LocalModeNodeCIDR, single-stack IPv4 cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24"},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv6 cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv6 cluster with single-stack IPv4 config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24"},
			// This will output a warning that there is no IPv6 CIDR but it
			// will still use the provided IPv4 CIDR for IPv4.
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv4 cluster with single-stack IPv6 config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96"},
			// This will output a warning that there is no IPv4 CIDR but it
			// will still use the provided IPv6 CIDR for IPv6.
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, dual-stack IPv4-primary cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, dual-stack IPv6-primary cluster",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96", "10.0.0.0/24"},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, IPv6-primary kube-proxy / IPv4-primary config",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, no PodCIDRs",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{""},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{},
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		// unknown mode
		{
			name: "unknown LocalMode",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalMode("abcd"),
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewNoOpLocalDetector(),
				v1.IPv6Protocol: NewNoOpLocalDetector(),
			},
		},
		// LocalModeBridgeInterface
		{
			name: "LocalModeBridgeInterface",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeBridgeInterface,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					BridgeInterface: "eth",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByBridgeInterface("eth"),
				v1.IPv6Protocol: NewDetectLocalByBridgeInterface("eth"),
			},
		},
		{
			name: "LocalModeBridgeInterface, strange bridge name",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeBridgeInterface,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					BridgeInterface: "1234567890123456789",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByBridgeInterface("1234567890123456789"),
				v1.IPv6Protocol: NewDetectLocalByBridgeInterface("1234567890123456789"),
			},
		},
		// LocalModeInterfaceNamePrefix
		{
			name: "LocalModeInterfaceNamePrefix",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "eth",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByInterfaceNamePrefix("eth"),
				v1.IPv6Protocol: NewDetectLocalByInterfaceNamePrefix("eth"),
			},
		},
		{
			name: "LocalModeInterfaceNamePrefix, strange interface name",
			config: &kubeproxyconfig.KubeProxyConfiguration{
				DetectLocalMode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "1234567890123456789",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]LocalTrafficDetector{
				v1.IPv4Protocol: NewDetectLocalByInterfaceNamePrefix("1234567890123456789"),
				v1.IPv6Protocol: NewDetectLocalByInterfaceNamePrefix("1234567890123456789"),
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			nodeManager := &testNodeManager{
				primaryIPFamily: c.primaryIPFamily,
				podCIDRs:        c.nodePodCIDRs,
			}
			r := GetLocalTrafficDetectors(ctx, c.config, nodeManager)
			if !reflect.DeepEqual(r, c.expected) {
				t.Errorf("Unexpected detect-local implementation, expected: %q, got: %q", c.expected, r)
			}
		})
	}
}
