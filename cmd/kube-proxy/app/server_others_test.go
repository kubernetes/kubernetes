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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	clientsetfake "k8s.io/client-go/kubernetes/fake"

	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utiliptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
)

type fakeIPSetVersioner struct {
	version string // what to return
	err     error  // what to return
}

func (fake *fakeIPSetVersioner) GetVersion() (string, error) {
	return fake.version, fake.err
}

type fakeKernelCompatTester struct {
	ok bool
}

func (fake *fakeKernelCompatTester) IsCompatible() error {
	if !fake.ok {
		return fmt.Errorf("error")
	}
	return nil
}

// fakeKernelHandler implements KernelHandler.
type fakeKernelHandler struct {
	modules       []string
	kernelVersion string
}

func (fake *fakeKernelHandler) GetModules() ([]string, error) {
	return fake.modules, nil
}

func (fake *fakeKernelHandler) GetKernelVersion() (string, error) {
	return fake.kernelVersion, nil
}

func Test_getProxyMode(t *testing.T) {
	var cases = []struct {
		flag          string
		ipsetVersion  string
		kmods         []string
		kernelVersion string
		kernelCompat  bool
		ipsetError    error
		expected      string
		scheduler     string
	}{
		{ // flag says userspace
			flag:     "userspace",
			expected: proxyModeUserspace,
		},
		{ // flag says iptables, kernel not compatible
			flag:         "iptables",
			kernelCompat: false,
			expected:     proxyModeUserspace,
		},
		{ // flag says iptables, kernel is compatible
			flag:         "iptables",
			kernelCompat: true,
			expected:     proxyModeIPTables,
		},
		{ // detect, kernel not compatible
			flag:         "",
			kernelCompat: false,
			expected:     proxyModeUserspace,
		},
		{ // detect, kernel is compatible
			flag:         "",
			kernelCompat: true,
			expected:     proxyModeIPTables,
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed for linux kernel before 4.19
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			kernelVersion: "4.18",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPVS,
			scheduler:     "rr",
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed for linux kernel 4.19
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPVS,
			scheduler:     "rr",
		},
		{ // flag says ipvs, ipset version too low, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  "0.0",
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
		{ // flag says ipvs, bad ipset version, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  "a.b.c",
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
		{ // flag says ipvs, required kernel modules are not installed, fallback on iptables mode
			flag:          "ipvs",
			kmods:         []string{"foo", "bar", "baz"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			kernelCompat:  true,
			expected:      proxyModeIPTables,
		},
		{ // flag says ipvs, ipset version ok, kernel modules installed for sed scheduler
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack", "ip_vs_sed"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPVS,
			scheduler:     "sed",
		},
		{ // flag says ipvs, kernel modules not installed for sed scheduler, fallback to iptables
			flag:          "ipvs",
			kmods:         []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  ipvs.MinIPSetCheckVersion,
			expected:      proxyModeIPTables,
			kernelCompat:  true,
			scheduler:     "sed",
		},
	}
	for i, c := range cases {
		kcompater := &fakeKernelCompatTester{c.kernelCompat}
		ipsetver := &fakeIPSetVersioner{c.ipsetVersion, c.ipsetError}
		khandler := &fakeKernelHandler{
			modules:       c.kmods,
			kernelVersion: c.kernelVersion,
		}
		canUseIPVS, _ := ipvs.CanUseIPVSProxier(khandler, ipsetver, cases[i].scheduler)
		r := getProxyMode(c.flag, canUseIPVS, kcompater)
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q, got %q", i, c.expected, r)
		}
	}
}

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
			expectedIP:  net.ParseIP("10.0.0.1"),
		},
		{
			name:        "Bind address IPv6 unicast address and no Node object",
			nodeInfo:    makeNodeWithAddresses("", "", ""),
			hostname:    "fakeHost",
			bindAddress: "fd00:4321::2",
			expectedIP:  net.ParseIP("fd00:4321::2"),
		},
		{
			name:        "No Valid IP found",
			nodeInfo:    makeNodeWithAddresses("", "", ""),
			hostname:    "fakeHost",
			bindAddress: "",
			expectedIP:  net.ParseIP("127.0.0.1"),
		},
		// Disabled because the GetNodeIP method has a backoff retry mechanism
		// and the test takes more than 30 seconds
		// ok  	k8s.io/kubernetes/cmd/kube-proxy/app	34.136s
		// {
		//	name:        "No Valid IP found and unspecified bind address",
		//	nodeInfo:    makeNodeWithAddresses("", "", ""),
		//	hostname:    "fakeHost",
		//	bindAddress: "0.0.0.0",
		//	expectedIP:  net.ParseIP("127.0.0.1"),
		// },
		{
			name:        "Bind address 0.0.0.0 and node with IPv4 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "192.168.1.1", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  net.ParseIP("192.168.1.1"),
		},
		{
			name:        "Bind address :: and node with IPv4 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "192.168.1.1", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  net.ParseIP("192.168.1.1"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with IPv6 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "fd00:1234::1", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  net.ParseIP("fd00:1234::1"),
		},
		{
			name:        "Bind address :: and node with IPv6 InternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "fd00:1234::1", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  net.ParseIP("fd00:1234::1"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with only IPv4 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  net.ParseIP("90.90.90.90"),
		},
		{
			name:        "Bind address :: and node with only IPv4 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "90.90.90.90"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  net.ParseIP("90.90.90.90"),
		},
		{
			name:        "Bind address 0.0.0.0 and node with only IPv6 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "0.0.0.0",
			expectedIP:  net.ParseIP("2001:db8::2"),
		},
		{
			name:        "Bind address :: and node with only IPv6 ExternalIP set",
			nodeInfo:    makeNodeWithAddresses("fakeHost", "", "2001:db8::2"),
			hostname:    "fakeHost",
			bindAddress: "::",
			expectedIP:  net.ParseIP("2001:db8::2"),
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
