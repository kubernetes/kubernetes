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
	"context"
	"fmt"
	"net"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fake "k8s.io/kubernetes/pkg/proxy/util/testing"
)

func TestValidateWorks(t *testing.T) {
	if isValidEndpoint("", 0) {
		t.Errorf("Didn't fail for empty set")
	}
	if isValidEndpoint("foobar", 0) {
		t.Errorf("Didn't fail with invalid port")
	}
	if isValidEndpoint("foobar", -1) {
		t.Errorf("Didn't fail with a negative port")
	}
	if !isValidEndpoint("foobar", 8080) {
		t.Errorf("Failed a valid config.")
	}
}

func TestBuildPortsToEndpointsMap(t *testing.T) {
	endpoints := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "testnamespace"},
		Subsets: []v1.EndpointSubset{
			{
				Addresses: []v1.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.2"},
				},
				Ports: []v1.EndpointPort{
					{Name: "http", Port: 80},
					{Name: "https", Port: 443},
				},
			},
			{
				Addresses: []v1.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.3"},
				},
				Ports: []v1.EndpointPort{
					{Name: "http", Port: 8080},
					{Name: "dns", Port: 53},
				},
			},
			{
				Addresses: []v1.EndpointAddress{},
				Ports: []v1.EndpointPort{
					{Name: "http", Port: 8888},
					{Name: "ssh", Port: 22},
				},
			},
			{
				Addresses: []v1.EndpointAddress{
					{IP: "10.0.0.1"},
				},
				Ports: []v1.EndpointPort{},
			},
		},
	}
	expectedPortsToEndpoints := map[string][]string{
		"http":  {"10.0.0.1:80", "10.0.0.2:80", "10.0.0.1:8080", "10.0.0.3:8080"},
		"https": {"10.0.0.1:443", "10.0.0.2:443"},
		"dns":   {"10.0.0.1:53", "10.0.0.3:53"},
	}

	portsToEndpoints := BuildPortsToEndpointsMap(endpoints)
	if !reflect.DeepEqual(expectedPortsToEndpoints, portsToEndpoints) {
		t.Errorf("expected ports to endpoints not seen")
	}
}

func TestIsProxyableIP(t *testing.T) {
	testCases := []struct {
		ip   string
		want error
	}{
		{"127.0.0.1", ErrAddressNotAllowed},
		{"127.0.0.2", ErrAddressNotAllowed},
		{"169.254.169.254", ErrAddressNotAllowed},
		{"169.254.1.1", ErrAddressNotAllowed},
		{"224.0.0.0", ErrAddressNotAllowed},
		{"10.0.0.1", nil},
		{"192.168.0.1", nil},
		{"172.16.0.1", nil},
		{"8.8.8.8", nil},
		{"::1", ErrAddressNotAllowed},
		{"fe80::", ErrAddressNotAllowed},
		{"ff02::", ErrAddressNotAllowed},
		{"ff01::", ErrAddressNotAllowed},
		{"2600::", nil},
		{"1", ErrAddressNotAllowed},
		{"", ErrAddressNotAllowed},
	}

	for i := range testCases {
		got := IsProxyableIP(testCases[i].ip)
		if testCases[i].want != got {
			t.Errorf("case %d: expected %v, got %v", i, testCases[i].want, got)
		}
	}
}

type dummyResolver struct {
	ips []string
	err error
}

func (r *dummyResolver) LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error) {
	if r.err != nil {
		return nil, r.err
	}
	resp := []net.IPAddr{}
	for _, ipString := range r.ips {
		resp = append(resp, net.IPAddr{IP: net.ParseIP(ipString)})
	}
	return resp, nil
}

func TestIsProxyableHostname(t *testing.T) {
	testCases := []struct {
		hostname string
		ips      []string
		want     error
	}{
		{"k8s.io", []string{}, ErrNoAddresses},
		{"k8s.io", []string{"8.8.8.8"}, nil},
		{"k8s.io", []string{"169.254.169.254"}, ErrAddressNotAllowed},
		{"k8s.io", []string{"127.0.0.1", "8.8.8.8"}, ErrAddressNotAllowed},
	}

	for i := range testCases {
		resolv := dummyResolver{ips: testCases[i].ips}
		got := IsProxyableHostname(context.Background(), &resolv, testCases[i].hostname)
		if testCases[i].want != got {
			t.Errorf("case %d: expected %v, got %v", i, testCases[i].want, got)
		}
	}
}

func TestIsAllowedHost(t *testing.T) {
	testCases := []struct {
		ip     string
		denied []string
		want   error
	}{
		{"8.8.8.8", []string{}, nil},
		{"169.254.169.254", []string{"169.0.0.0/8"}, ErrAddressNotAllowed},
		{"169.254.169.254", []string{"fce8::/15", "169.254.169.0/24"}, ErrAddressNotAllowed},
		{"fce9:beef::", []string{"fce8::/15", "169.254.169.0/24"}, ErrAddressNotAllowed},
		{"127.0.0.1", []string{"127.0.0.1/32"}, ErrAddressNotAllowed},
		{"34.107.204.206", []string{"fce8::/15"}, nil},
		{"fce9:beef::", []string{"127.0.0.1/32"}, nil},
		{"34.107.204.206", []string{"127.0.0.1/32"}, nil},
		{"127.0.0.1", []string{}, nil},
	}

	for i := range testCases {
		var denyList []*net.IPNet
		for _, cidrStr := range testCases[i].denied {
			_, ipNet, err := net.ParseCIDR(cidrStr)
			if err != nil {
				t.Fatalf("bad IP for test case: %v: %v", cidrStr, err)
			}
			denyList = append(denyList, ipNet)
		}
		got := IsAllowedHost(net.ParseIP(testCases[i].ip), denyList)
		if testCases[i].want != got {
			t.Errorf("case %d: expected %v, got %v", i, testCases[i].want, got)
		}
	}
}

func TestShouldSkipService(t *testing.T) {
	testCases := []struct {
		service    *v1.Service
		shouldSkip bool
	}{
		{
			// Cluster IP is None
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},
			shouldSkip: true,
		},
		{
			// Cluster IP is empty
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: "",
				},
			},
			shouldSkip: true,
		},
		{
			// ExternalName type service
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      v1.ServiceTypeExternalName,
				},
			},
			shouldSkip: true,
		},
		{
			// ClusterIP type service with ClusterIP set
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      v1.ServiceTypeClusterIP,
				},
			},
			shouldSkip: false,
		},
		{
			// NodePort type service with ClusterIP set
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      v1.ServiceTypeNodePort,
				},
			},
			shouldSkip: false,
		},
		{
			// LoadBalancer type service with ClusterIP set
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      v1.ServiceTypeLoadBalancer,
				},
			},
			shouldSkip: false,
		},
	}

	for i := range testCases {
		skip := ShouldSkipService(testCases[i].service)
		if skip != testCases[i].shouldSkip {
			t.Errorf("case %d: expect %v, got %v", i, testCases[i].shouldSkip, skip)
		}
	}
}

type InterfaceAddrsPair struct {
	itf   net.Interface
	addrs []net.Addr
}

func TestGetNodeAddressses(t *testing.T) {
	testCases := []struct {
		cidrs         []string
		nw            *fake.FakeNetwork
		itfAddrsPairs []InterfaceAddrsPair
		expected      sets.String
		expectedErr   error
	}{
		{ // case 0
			cidrs: []string{"10.20.30.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "100.200.201.1/24"}},
				},
			},
			expected: sets.NewString("10.20.30.51"),
		},
		{ // case 1
			cidrs: []string{"0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 2
			cidrs: []string{"2001:db8::/32", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("2001:db8::1", "::1"),
		},
		{ // case 3
			cidrs: []string{"::/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("::/0"),
		},
		{ // case 4
			cidrs: []string{"127.0.0.1/32"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("127.0.0.1"),
		},
		{ // case 5
			cidrs: []string{"127.0.0.0/8"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.1.1/8"}},
				},
			},
			expected: sets.NewString("127.0.1.1"),
		},
		{ // case 6
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "100.200.201.1/24"}},
				},
			},
			expected: sets.NewString("10.20.30.51", "100.200.201.1"),
		},
		{ // case 7
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "192.168.1.2/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected:    nil,
			expectedErr: fmt.Errorf("no addresses found for cidrs %v", []string{"10.20.30.0/24", "100.200.201.0/24"}),
		},
		{ // case 8
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "192.168.1.2/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{"1.2.3.0/24", "0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 10
			cidrs: []string{"0.0.0.0/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::1"),
		},
		{ // case 11
			cidrs: []string{"::/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("::/0", "1.2.3.4"),
		},
	}

	for i := range testCases {
		for _, pair := range testCases[i].itfAddrsPairs {
			testCases[i].nw.AddInterfaceAddr(&pair.itf, pair.addrs)
		}
		addrList, err := GetNodeAddresses(testCases[i].cidrs, testCases[i].nw)
		if !reflect.DeepEqual(err, testCases[i].expectedErr) {
			t.Errorf("case [%d], unexpected error: %v", i, err)
		}

		if !addrList.Equal(testCases[i].expected) {
			t.Errorf("case [%d], unexpected mismatch, expected: %v, got: %v", i, testCases[i].expected, addrList)
		}
	}
}

func TestAppendPortIfNeeded(t *testing.T) {
	testCases := []struct {
		name   string
		addr   string
		port   int32
		expect string
	}{
		{
			name:   "IPv4 all-zeros bind address has port",
			addr:   "0.0.0.0:12345",
			port:   23456,
			expect: "0.0.0.0:12345",
		},
		{
			name:   "non-zeros IPv4 config",
			addr:   "9.8.7.6",
			port:   12345,
			expect: "9.8.7.6:12345",
		},
		{
			name:   "IPv6 \"[::]\" bind address has port",
			addr:   "[::]:12345",
			port:   23456,
			expect: "[::]:12345",
		},
		{
			name:   "IPv6 config",
			addr:   "fd00:1::5",
			port:   23456,
			expect: "[fd00:1::5]:23456",
		},
		{
			name:   "Invalid IPv6 Config",
			addr:   "[fd00:1::5]",
			port:   12345,
			expect: "[fd00:1::5]",
		},
	}

	for i := range testCases {
		got := AppendPortIfNeeded(testCases[i].addr, testCases[i].port)
		if testCases[i].expect != got {
			t.Errorf("case %s: expected %v, got %v", testCases[i].name, testCases[i].expect, got)
		}
	}
}

func TestShuffleStrings(t *testing.T) {
	var src []string
	dest := ShuffleStrings(src)

	if dest != nil {
		t.Errorf("ShuffleStrings for a nil slice got a non-nil slice")
	}

	src = []string{"a", "b", "c", "d", "e", "f"}
	dest = ShuffleStrings(src)

	if len(src) != len(dest) {
		t.Errorf("Shuffled slice is wrong length, expected %v got %v", len(src), len(dest))
	}

	m := make(map[string]bool, len(dest))
	for _, s := range dest {
		m[s] = true
	}

	for _, k := range src {
		if _, exists := m[k]; !exists {
			t.Errorf("Element %v missing from shuffled slice", k)
		}
	}
}

func TestFilterIncorrectIPVersion(t *testing.T) {
	testCases := []struct {
		desc            string
		ipString        []string
		wantIPv6        bool
		expectCorrect   []string
		expectIncorrect []string
	}{
		{
			desc:            "empty input IPv4",
			ipString:        []string{},
			wantIPv6:        false,
			expectCorrect:   nil,
			expectIncorrect: nil,
		},
		{
			desc:            "empty input IPv6",
			ipString:        []string{},
			wantIPv6:        true,
			expectCorrect:   nil,
			expectIncorrect: nil,
		},
		{
			desc:            "want IPv4 and receive IPv6",
			ipString:        []string{"fd00:20::1"},
			wantIPv6:        false,
			expectCorrect:   nil,
			expectIncorrect: []string{"fd00:20::1"},
		},
		{
			desc:            "want IPv6 and receive IPv4",
			ipString:        []string{"192.168.200.2"},
			wantIPv6:        true,
			expectCorrect:   nil,
			expectIncorrect: []string{"192.168.200.2"},
		},
		{
			desc:            "want IPv6 and receive IPv4 and IPv6",
			ipString:        []string{"192.168.200.2", "192.1.34.23", "fd00:20::1", "2001:db9::3"},
			wantIPv6:        true,
			expectCorrect:   []string{"fd00:20::1", "2001:db9::3"},
			expectIncorrect: []string{"192.168.200.2", "192.1.34.23"},
		},
		{
			desc:            "want IPv4 and receive IPv4 and IPv6",
			ipString:        []string{"192.168.200.2", "192.1.34.23", "fd00:20::1", "2001:db9::3"},
			wantIPv6:        false,
			expectCorrect:   []string{"192.168.200.2", "192.1.34.23"},
			expectIncorrect: []string{"fd00:20::1", "2001:db9::3"},
		},
		{
			desc:            "want IPv4 and receive IPv4 only",
			ipString:        []string{"192.168.200.2", "192.1.34.23"},
			wantIPv6:        false,
			expectCorrect:   []string{"192.168.200.2", "192.1.34.23"},
			expectIncorrect: nil,
		},
		{
			desc:            "want IPv6 and receive IPv4 only",
			ipString:        []string{"192.168.200.2", "192.1.34.23"},
			wantIPv6:        true,
			expectCorrect:   nil,
			expectIncorrect: []string{"192.168.200.2", "192.1.34.23"},
		},
		{
			desc:            "want IPv4 and receive IPv6 only",
			ipString:        []string{"fd00:20::1", "2001:db9::3"},
			wantIPv6:        false,
			expectCorrect:   nil,
			expectIncorrect: []string{"fd00:20::1", "2001:db9::3"},
		},
		{
			desc:            "want IPv6 and receive IPv6 only",
			ipString:        []string{"fd00:20::1", "2001:db9::3"},
			wantIPv6:        true,
			expectCorrect:   []string{"fd00:20::1", "2001:db9::3"},
			expectIncorrect: nil,
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.desc, func(t *testing.T) {
			ipFamily := v1.IPv4Protocol
			if testcase.wantIPv6 {
				ipFamily = v1.IPv6Protocol
			}
			correct, incorrect := FilterIncorrectIPVersion(testcase.ipString, ipFamily)
			if !reflect.DeepEqual(testcase.expectCorrect, correct) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectCorrect, correct)
			}
			if !reflect.DeepEqual(testcase.expectIncorrect, incorrect) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectIncorrect, incorrect)
			}
		})
	}
}

func TestGetClusterIPByFamily(t *testing.T) {
	testCases := []struct {
		name           string
		service        v1.Service
		requestFamily  v1.IPFamily
		expectedResult string
	}{
		{
			name:           "old style service ipv4. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "10.0.0.10",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "10.0.0.10",
				},
			},
		},

		{
			name:           "old style service ipv4. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "10.0.0.10",
				},
			},
		},

		{
			name:           "old style service ipv6. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "2000::1",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "2000::1",
				},
			},
		},

		{
			name:           "old style service ipv6. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "2000::1",
				},
			},
		},

		{
			name:           "service single stack ipv4. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "10.0.0.10",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"10.0.0.10"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
		},

		{
			name:           "service single stack ipv4. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"10.0.0.10"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
		},

		{
			name:           "service single stack ipv6. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "2000::1",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"2000::1"},
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
		},

		{
			name:           "service single stack ipv6. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"2000::1"},
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
		},
		// dual stack
		{
			name:           "service dual stack ipv4,6. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "10.0.0.10",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol},
				},
			},
		},

		{
			name:           "service dual stack ipv4,6. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "2000::1",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol},
				},
			},
		},

		{
			name:           "service dual stack ipv6,4. want ipv6",
			requestFamily:  v1.IPv6Protocol,
			expectedResult: "2000::1",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"2000::1", "10.0.0.10"},
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol, v1.IPv4Protocol},
				},
			},
		},

		{
			name:           "service dual stack ipv6,4. want ipv4",
			requestFamily:  v1.IPv4Protocol,
			expectedResult: "10.0.0.10",
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIPs: []string{"2000::1", "10.0.0.10"},
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol, v1.IPv4Protocol},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ip := GetClusterIPByFamily(testCase.requestFamily, &testCase.service)
			if ip != testCase.expectedResult {
				t.Fatalf("expected ip:%v got %v", testCase.expectedResult, ip)
			}
		})
	}

}

func TestFilterIncorrectLoadBalancerIngress(t *testing.T) {
	ipModeVIP := v1.LoadBalancerIPModeVIP
	testCases := []struct {
		name              string
		ingresses         []v1.LoadBalancerIngress
		ipFamily          v1.IPFamily
		expectedCorrect   []v1.LoadBalancerIngress
		expectedIncorrect []v1.LoadBalancerIngress
	}{
		{
			name:     "IPv4 only valid ingresses",
			ipFamily: v1.IPv4Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP:     "1.2.3.4",
					IPMode: &ipModeVIP,
				},
				{
					IP: "1.2.3.5",
				},
			},
			expectedCorrect: []v1.LoadBalancerIngress{
				{
					IP:     "1.2.3.4",
					IPMode: &ipModeVIP,
				},
				{
					IP: "1.2.3.5",
				},
			},
			expectedIncorrect: nil,
		},
		{
			name:     "IPv4 some invalid ingresses",
			ipFamily: v1.IPv4Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP:     "1.2.3.4",
					IPMode: &ipModeVIP,
				},
				{
					IP: "2000::1",
				},
				{
					Hostname: "dummy",
				},
			},
			expectedCorrect: []v1.LoadBalancerIngress{
				{
					IP:     "1.2.3.4",
					IPMode: &ipModeVIP,
				},
				{
					Hostname: "dummy", // weirdly no IP is a valid IPv4 but invalid IPv6
				},
			},
			expectedIncorrect: []v1.LoadBalancerIngress{
				{
					IP: "2000::1",
				},
			},
		},
		{
			name:     "IPv4 only invalid ingresses",
			ipFamily: v1.IPv4Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP: "2000::1",
				},
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
			},
			expectedCorrect: nil,
			expectedIncorrect: []v1.LoadBalancerIngress{
				{
					IP: "2000::1",
				},
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
			},
		},
		{
			name:     "IPv6 only valid ingresses",
			ipFamily: v1.IPv6Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
				{
					IP: "2000::2",
				},
			},
			expectedCorrect: []v1.LoadBalancerIngress{
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
				{
					IP: "2000::2",
				},
			},
			expectedIncorrect: nil,
		},
		{
			name:     "IPv6 some invalid ingresses",
			ipFamily: v1.IPv6Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
				{
					IP: "1.2.3.4",
				},
				{
					Hostname: "dummy",
				},
			},
			expectedCorrect: []v1.LoadBalancerIngress{
				{
					IP:     "2000::1",
					IPMode: &ipModeVIP,
				},
			},
			expectedIncorrect: []v1.LoadBalancerIngress{
				{
					IP: "1.2.3.4",
				},
				{
					Hostname: "dummy", // weirdly no IP is a valid IPv4 but invalid IPv6
				},
			},
		},
		{
			name:     "IPv6 only invalid ingresses",
			ipFamily: v1.IPv6Protocol,
			ingresses: []v1.LoadBalancerIngress{
				{
					IP: "1.2.3.4",
				},
				{
					IP:     "1.2.3.5",
					IPMode: &ipModeVIP,
				},
			},
			expectedCorrect: nil,
			expectedIncorrect: []v1.LoadBalancerIngress{
				{
					IP: "1.2.3.4",
				},
				{
					IP:     "1.2.3.5",
					IPMode: &ipModeVIP,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			correctIngresses, incorrectIngresses := FilterIncorrectLoadBalancerIngress(testCase.ingresses, testCase.ipFamily)
			if !reflect.DeepEqual(correctIngresses, testCase.expectedCorrect) {
				t.Errorf("Test %v failed: expected %v, got %v", testCase.name, testCase.expectedCorrect, correctIngresses)
			}
			if !reflect.DeepEqual(incorrectIngresses, testCase.expectedIncorrect) {
				t.Errorf("Test %v failed: expected %v, got %v", testCase.name, testCase.expectedIncorrect, incorrectIngresses)
			}
		})
	}
}
