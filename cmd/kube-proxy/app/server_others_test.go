//go:build !windows
// +build !windows

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

package app

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	netutils "k8s.io/utils/net"

	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"

	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utiliptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
)

func Test_getDetectLocalMode(t *testing.T) {
	cases := []struct {
		detectLocal string
		expected    proxyconfigapi.LocalMode
		errExpected bool
	}{
		{
			detectLocal: "",
			expected:    proxyconfigapi.LocalModeClusterCIDR,
			errExpected: false,
		},
		{
			detectLocal: string(proxyconfigapi.LocalModeClusterCIDR),
			expected:    proxyconfigapi.LocalModeClusterCIDR,
			errExpected: false,
		},
		{
			detectLocal: string(proxyconfigapi.LocalModeInterfaceNamePrefix),
			expected:    proxyconfigapi.LocalModeInterfaceNamePrefix,
			errExpected: false,
		},
		{
			detectLocal: string(proxyconfigapi.LocalModeBridgeInterface),
			expected:    proxyconfigapi.LocalModeBridgeInterface,
			errExpected: false,
		},
		{
			detectLocal: "abcd",
			expected:    proxyconfigapi.LocalMode("abcd"),
			errExpected: true,
		},
	}
	for i, c := range cases {
		proxyConfig := &proxyconfigapi.KubeProxyConfiguration{DetectLocalMode: proxyconfigapi.LocalMode(c.detectLocal)}
		r, err := getDetectLocalMode(proxyConfig)
		if c.errExpected {
			if err == nil {
				t.Errorf("Expected error, but did not fail for mode %v", c.detectLocal)
			}
			continue
		}
		if err != nil {
			t.Errorf("Got error parsing mode: %v", err)
			continue
		}
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q got %q", i, c.expected, r)
		}
	}
}

func Test_detectNodeIP(t *testing.T) {
	cases := []struct {
		name        string
		nodeInfo    *v1.Node
		hostname    string
		bindAddress string
		expectedIP  net.IP
	}{
		{
			name:        "Bind address IPv4 unicast address and no Node object",
			nodeInfo:    makeNodeWithAddresses("", "", ""),
			hostname:    "fakeHost",
			bindAddress: "10.0.0.1",
			expectedIP:  netutils.ParseIPSloppy("10.0.0.1"),
		},
		{
			name:        "Bind address IPv6 unicast address and no Node object",
			nodeInfo:    makeNodeWithAddresses("", "", ""),
			hostname:    "fakeHost",
			bindAddress: "fd00:4321::2",
			expectedIP:  netutils.ParseIPSloppy("fd00:4321::2"),
		},
		{
			name:        "No Valid IP found",
			nodeInfo:    makeNodeWithAddresses("", "", ""),
			hostname:    "fakeHost",
			bindAddress: "",
			expectedIP:  netutils.ParseIPSloppy("127.0.0.1"),
		},
		// Disabled because the GetNodeIP method has a backoff retry mechanism
		// and the test takes more than 30 seconds
		// ok  	k8s.io/kubernetes/cmd/kube-proxy/app	34.136s
		// {
		//	name:        "No Valid IP found and unspecified bind address",
		//	nodeInfo:    makeNodeWithAddresses("", "", ""),
		//	hostname:    "fakeHost",
		//	bindAddress: "0.0.0.0",
		//	expectedIP:  net.IP{127,0,0,1),
		// },
		{
			name:        "Bind address 0.0.0.0 and node with IPv4 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "192.168.1.1", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  netutils.ParseIPSloppy("192.168.1.1"),
		},
		{
			name:        "Bind address :: and node with IPv4 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "192.168.1.1", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  netutils.ParseIPSloppy("192.168.1.1"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with IPv6 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "fd00:1234::1", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  netutils.ParseIPSloppy("fd00:1234::1"),
		},
		{
			name:        "Bind address :: and node with IPv6 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "fd00:1234::1", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  netutils.ParseIPSloppy("fd00:1234::1"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with only IPv4 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  netutils.ParseIPSloppy("90.90.90.90"),
		},
		{
			name:        "Bind address :: and node with only IPv4 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  netutils.ParseIPSloppy("90.90.90.90"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with only IPv6 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  netutils.ParseIPSloppy("2001:db8::2"),
		},
		{
			name:        "Bind address :: and node with only IPv6 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  netutils.ParseIPSloppy("2001:db8::2"),
		},
	}
	for _, c := range cases {
		client := clientsetfake.NewSimpleClientset(c.nodeInfo)
		ip := detectNodeIP(client, c.hostname, c.bindAddress)
		if !ip.Equal(c.expectedIP) {
			t.Errorf("Case[%s] Expected IP %q got %q", c.name, c.expectedIP, ip)
		}
	}
}

func Test_getLocalDetector(t *testing.T) {
	cases := []struct {
		mode        proxyconfigapi.LocalMode
		config      *proxyconfigapi.KubeProxyConfiguration
		ipt         utiliptables.Interface
		expected    proxyutiliptables.LocalTrafficDetector
		nodeInfo    *v1.Node
		errExpected bool
	}{
		// LocalModeClusterCIDR, nodeInfo would be nil for these cases
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:         utiliptablestest.NewFake(),
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14", utiliptablestest.NewFake())),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/64", utiliptablestest.NewIPv6Fake())),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0"},
			ipt:         utiliptablestest.NewFake(),
			expected:    nil,
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    nil,
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    nil,
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:         utiliptablestest.NewFake(),
			expected:    nil,
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         utiliptablestest.NewFake(),
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		// LocalModeNodeCIDR
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:         utiliptablestest.NewFake(),
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24", utiliptablestest.NewFake())),
			nodeInfo:    makeNodeWithPodCIDRs("10.0.0.0/24"),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/96", utiliptablestest.NewIPv6Fake())),
			nodeInfo:    makeNodeWithPodCIDRs("2002::1234:abcd:ffff:c0a8:101/96"),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0"},
			ipt:         utiliptablestest.NewFake(),
			expected:    nil,
			nodeInfo:    makeNodeWithPodCIDRs("10.0.0.0"),
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    nil,
			nodeInfo:    makeNodeWithPodCIDRs("2002::1234:abcd:ffff:c0a8:101"),
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:         utiliptablestest.NewIPv6Fake(),
			expected:    nil,
			nodeInfo:    makeNodeWithPodCIDRs("10.0.0.0/24"),
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:         utiliptablestest.NewFake(),
			expected:    nil,
			nodeInfo:    makeNodeWithPodCIDRs("2002::1234:abcd:ffff:c0a8:101/96"),
			errExpected: true,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         utiliptablestest.NewFake(),
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			nodeInfo:    makeNodeWithPodCIDRs(),
			errExpected: false,
		},
		// unknown mode, nodeInfo would be nil for these cases
		{
			mode:        proxyconfigapi.LocalMode("abcd"),
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:         utiliptablestest.NewFake(),
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		// LocalModeBridgeInterface, nodeInfo and ipt are not needed for these cases
		{
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "eth"},
			},
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByBridgeInterface("eth")),
			errExpected: false,
		},
		{
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: ""},
			},
			errExpected: true,
		},
		{
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "1234567890123456789"},
			},
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByBridgeInterface("1234567890123456789")),
			errExpected: false,
		},
		// LocalModeInterfaceNamePrefix, nodeInfo and ipt are not needed for these cases
		{
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "eth"},
			},
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("eth")),
			errExpected: false,
		},
		{
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: ""},
			},
			errExpected: true,
		},
		{
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "1234567890123456789"},
			},
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("1234567890123456789")),
			errExpected: false,
		},
	}
	for i, c := range cases {
		r, err := getLocalDetector(c.mode, c.config, c.ipt, c.nodeInfo)
		if c.errExpected {
			if err == nil {
				t.Errorf("Case[%d] Expected error, but succeeded with %v", i, r)
			}
			continue
		}
		if err != nil {
			t.Errorf("Case[%d] Error resolving detect-local: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(r, c.expected) {
			t.Errorf("Case[%d] Unexpected detect-local implementation, expected: %q, got: %q", i, c.expected, r)
		}
	}
}

func Test_getDualStackLocalDetectorTuple(t *testing.T) {
	cases := []struct {
		mode        proxyconfigapi.LocalMode
		config      *proxyconfigapi.KubeProxyConfiguration
		ipt         [2]utiliptables.Interface
		expected    [2]proxyutiliptables.LocalTrafficDetector
		nodeInfo    *v1.Node
		errExpected bool
	}{
		// LocalModeClusterCIDR, nodeInfo would be nil for these cases
		{
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14,2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14", utiliptablestest.NewFake()))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/64", utiliptablestest.NewIPv6Fake())),
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64,10.0.0.0/14"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14", utiliptablestest.NewFake()))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/64", utiliptablestest.NewIPv6Fake())),
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14", utiliptablestest.NewFake())),
				proxyutiliptables.NewNoOpLocalDetector()},
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				proxyutiliptables.NewNoOpLocalDetector(),
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/64", utiliptablestest.NewIPv6Fake()))},
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected:    [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			errExpected: false,
		},
		// LocalModeNodeCIDR
		{
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14,2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24", utiliptablestest.NewFake()))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/96", utiliptablestest.NewIPv6Fake())),
			nodeInfo:    makeNodeWithPodCIDRs("10.0.0.0/24", "2002::1234:abcd:ffff:c0a8:101/96"),
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64,10.0.0.0/14"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24", utiliptablestest.NewFake()))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/96", utiliptablestest.NewIPv6Fake())),
			nodeInfo:    makeNodeWithPodCIDRs("2002::1234:abcd:ffff:c0a8:101/96", "10.0.0.0/24"),
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24", utiliptablestest.NewFake())),
				proxyutiliptables.NewNoOpLocalDetector()},
			nodeInfo:    makeNodeWithPodCIDRs("10.0.0.0/24"),
			errExpected: false,
		},
		{
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002::1234:abcd:ffff:c0a8:101/64"},
			ipt:    [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				proxyutiliptables.NewNoOpLocalDetector(),
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:c0a8:101/96", utiliptablestest.NewIPv6Fake()))},
			nodeInfo:    makeNodeWithPodCIDRs("2002::1234:abcd:ffff:c0a8:101/96"),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected:    [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			nodeInfo:    makeNodeWithPodCIDRs(),
			errExpected: false,
		},
		{
			mode:        proxyconfigapi.LocalModeNodeCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected:    [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			nodeInfo:    nil,
			errExpected: false,
		},
		// unknown mode, nodeInfo would be nil for these cases
		{
			mode:        proxyconfigapi.LocalMode("abcd"),
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			ipt:         [2]utiliptables.Interface{utiliptablestest.NewFake(), utiliptablestest.NewIPv6Fake()},
			expected:    [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			errExpected: false,
		},
		// LocalModeBridgeInterface, nodeInfo and ipt are not needed for these cases
		{
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "eth"},
			},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByBridgeInterface("eth"))(
				proxyutiliptables.NewDetectLocalByBridgeInterface("eth")),
			errExpected: false,
		},
		{
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: ""},
			},
			errExpected: true,
		},
		// LocalModeInterfaceNamePrefix, nodeInfo and ipt are not needed for these cases
		{
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "veth"},
			},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("veth"))(
				proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("veth")),
			errExpected: false,
		},
		{
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: ""},
			},
			errExpected: true,
		},
	}
	for i, c := range cases {
		r, err := getDualStackLocalDetectorTuple(c.mode, c.config, c.ipt, c.nodeInfo)
		if c.errExpected {
			if err == nil {
				t.Errorf("Case[%d] expected error, but succeeded with %q", i, r)
			}
			continue
		}
		if err != nil {
			t.Errorf("Case[%d] Error resolving detect-local: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(r, c.expected) {
			t.Errorf("Case[%d] Unexpected detect-local implementation, expected: %q, got: %q", i, c.expected, r)
		}
	}
}

func makeNodeWithAddresses(name, internal, external string) *v1.Node {
	if name == "" {
		return &v1.Node{}
	}

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{},
		},
	}

	if internal != "" {
		node.Status.Addresses = append(node.Status.Addresses,
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: internal},
		)
	}

	if external != "" {
		node.Status.Addresses = append(node.Status.Addresses,
			v1.NodeAddress{Type: v1.NodeExternalIP, Address: external},
		)
	}

	return node
}

func makeNodeWithPodCIDRs(cidrs ...string) *v1.Node {
	if len(cidrs) == 0 {
		return &v1.Node{}
	}
	return &v1.Node{
		Spec: v1.NodeSpec{
			PodCIDR:  cidrs[0],
			PodCIDRs: cidrs,
		},
	}
}

func resolveLocalDetector(t *testing.T) func(proxyutiliptables.LocalTrafficDetector, error) proxyutiliptables.LocalTrafficDetector {
	return func(localDetector proxyutiliptables.LocalTrafficDetector, err error) proxyutiliptables.LocalTrafficDetector {
		t.Helper()
		if err != nil {
			t.Fatalf("Error resolving detect-local: %v", err)
		}
		return localDetector
	}
}

func resolveDualStackLocalDetectors(t *testing.T) func(localDetector proxyutiliptables.LocalTrafficDetector, err1 error) func(proxyutiliptables.LocalTrafficDetector, error) [2]proxyutiliptables.LocalTrafficDetector {
	return func(localDetector proxyutiliptables.LocalTrafficDetector, err error) func(proxyutiliptables.LocalTrafficDetector, error) [2]proxyutiliptables.LocalTrafficDetector {
		t.Helper()
		if err != nil {
			t.Fatalf("Error resolving dual stack detect-local: %v", err)
		}
		return func(otherLocalDetector proxyutiliptables.LocalTrafficDetector, err1 error) [2]proxyutiliptables.LocalTrafficDetector {
			t.Helper()
			if err1 != nil {
				t.Fatalf("Error resolving dual stack detect-local: %v", err)
			}
			return [2]proxyutiliptables.LocalTrafficDetector{localDetector, otherLocalDetector}
		}
	}
}

func TestConfigChange(t *testing.T) {
	setUp := func() (*os.File, string, error) {
		tempDir, err := os.MkdirTemp("", "kubeproxy-config-change")
		if err != nil {
			return nil, "", fmt.Errorf("unable to create temporary directory: %v", err)
		}
		fullPath := filepath.Join(tempDir, "kube-proxy-config")
		file, err := os.Create(fullPath)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when creating temp file: %v", err)
		}

		_, err = file.WriteString(`apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
bindAddressHardFail: false
clientConnection:
  acceptContentTypes: ""
  burst: 10
  contentType: application/vnd.kubernetes.protobuf
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
  qps: 5
clusterCIDR: 10.244.0.0/16
configSyncPeriod: 15m0s
conntrack:
  maxPerCore: 32768
  min: 131072
  tcpCloseWaitTimeout: 1h0m0s
  tcpEstablishedTimeout: 24h0m0s
enableProfiling: false
healthzBindAddress: 0.0.0.0:10256
hostnameOverride: ""
iptables:
  masqueradeAll: false
  masqueradeBit: 14
  minSyncPeriod: 0s
  syncPeriod: 30s
ipvs:
  excludeCIDRs: null
  minSyncPeriod: 0s
  scheduler: ""
  syncPeriod: 30s
kind: KubeProxyConfiguration
metricsBindAddress: 127.0.0.1:10249
mode: ""
nodePortAddresses: null
oomScoreAdj: -999
portRange: ""
detectLocalMode: "BridgeInterface"`)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when writing content to temp kube-proxy config file: %v", err)
		}

		return file, tempDir, nil
	}

	tearDown := func(file *os.File, tempDir string) {
		file.Close()
		os.RemoveAll(tempDir)
	}

	testCases := []struct {
		name        string
		proxyServer proxyRun
		append      bool
		expectedErr string
	}{
		{
			name:        "update config file",
			proxyServer: new(fakeProxyServerLongRun),
			append:      true,
			expectedErr: "content of the proxy server's configuration file was updated",
		},
		{
			name:        "fake error",
			proxyServer: new(fakeProxyServerError),
			expectedErr: "mocking error from ProxyServer.Run()",
		},
	}

	for _, tc := range testCases {
		file, tempDir, err := setUp()
		if err != nil {
			t.Fatalf("unexpected error when setting up environment: %v", err)
		}

		opt := NewOptions()
		opt.ConfigFile = file.Name()
		err = opt.Complete()
		if err != nil {
			t.Fatal(err)
		}
		opt.proxyServer = tc.proxyServer

		errCh := make(chan error, 1)
		go func() {
			errCh <- opt.runLoop()
		}()

		if tc.append {
			file.WriteString("append fake content")
		}

		select {
		case err := <-errCh:
			if err != nil {
				if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("[%s] Expected error containing %v, got %v", tc.name, tc.expectedErr, err)
				}
			}
		case <-time.After(10 * time.Second):
			t.Errorf("[%s] Timeout: unable to get any events or internal timeout.", tc.name)
		}
		tearDown(file, tempDir)
	}
}

func Test_waitForPodCIDR(t *testing.T) {
	expected := []string{"192.168.0.0/24", "fd00:1:2::/64"}
	nodeName := "test-node"
	oldNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            nodeName,
			ResourceVersion: "1000",
		},
		Spec: v1.NodeSpec{
			PodCIDR:  "10.0.0.0/24",
			PodCIDRs: []string{"10.0.0.0/24", "2001:db2:1/64"},
		},
	}
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            nodeName,
			ResourceVersion: "1",
		},
	}
	updatedNode := node.DeepCopy()
	updatedNode.Spec.PodCIDRs = expected
	updatedNode.Spec.PodCIDR = expected[0]

	// start with the new node
	client := clientsetfake.NewSimpleClientset()
	client.AddReactor("list", "nodes", func(action clientgotesting.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.NodeList{}
		return true, obj, nil
	})
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("nodes", clientgotesting.DefaultWatchReactor(fakeWatch, nil))

	go func() {
		fakeWatch.Add(node)
		// receive a delete event for the old node
		fakeWatch.Delete(oldNode)
		// set the PodCIDRs on the new node
		fakeWatch.Modify(updatedNode)
	}()
	got, err := waitForPodCIDR(client, node.Name)
	if err != nil {
		t.Errorf("waitForPodCIDR() unexpected error %v", err)
		return
	}
	if !reflect.DeepEqual(got.Spec.PodCIDRs, expected) {
		t.Errorf("waitForPodCIDR() got %v expected to be %v ", got.Spec.PodCIDRs, expected)
	}
}
