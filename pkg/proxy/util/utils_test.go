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
	"net"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	netutils "k8s.io/utils/net"
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

func TestMapIPsByIPFamily(t *testing.T) {
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
			otherIPFamily := v1.IPv6Protocol

			if testcase.wantIPv6 {
				ipFamily = v1.IPv6Protocol
				otherIPFamily = v1.IPv4Protocol
			}

			ipMap := MapIPsByIPFamily(testcase.ipString)

			var ipStr []string
			for _, ip := range ipMap[ipFamily] {
				ipStr = append(ipStr, ip.String())
			}
			if !reflect.DeepEqual(testcase.expectCorrect, ipStr) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectCorrect, ipMap[ipFamily])
			}
			ipStr = nil
			for _, ip := range ipMap[otherIPFamily] {
				ipStr = append(ipStr, ip.String())
			}
			if !reflect.DeepEqual(testcase.expectIncorrect, ipStr) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectIncorrect, ipMap[otherIPFamily])
			}
		})
	}
}

func TestMapCIDRsByIPFamily(t *testing.T) {
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
			ipString:        []string{"fd00:20::/64"},
			wantIPv6:        false,
			expectCorrect:   nil,
			expectIncorrect: []string{"fd00:20::/64"},
		},
		{
			desc:            "want IPv6 and receive IPv4",
			ipString:        []string{"192.168.200.0/24"},
			wantIPv6:        true,
			expectCorrect:   nil,
			expectIncorrect: []string{"192.168.200.0/24"},
		},
		{
			desc:            "want IPv6 and receive IPv4 and IPv6",
			ipString:        []string{"192.168.200.0/24", "192.1.34.0/24", "fd00:20::/64", "2001:db9::/64"},
			wantIPv6:        true,
			expectCorrect:   []string{"fd00:20::/64", "2001:db9::/64"},
			expectIncorrect: []string{"192.168.200.0/24", "192.1.34.0/24"},
		},
		{
			desc:            "want IPv4 and receive IPv4 and IPv6",
			ipString:        []string{"192.168.200.0/24", "192.1.34.0/24", "fd00:20::/64", "2001:db9::/64"},
			wantIPv6:        false,
			expectCorrect:   []string{"192.168.200.0/24", "192.1.34.0/24"},
			expectIncorrect: []string{"fd00:20::/64", "2001:db9::/64"},
		},
		{
			desc:            "want IPv4 and receive IPv4 only",
			ipString:        []string{"192.168.200.0/24", "192.1.34.0/24"},
			wantIPv6:        false,
			expectCorrect:   []string{"192.168.200.0/24", "192.1.34.0/24"},
			expectIncorrect: nil,
		},
		{
			desc:            "want IPv6 and receive IPv4 only",
			ipString:        []string{"192.168.200.0/24", "192.1.34.0/24"},
			wantIPv6:        true,
			expectCorrect:   nil,
			expectIncorrect: []string{"192.168.200.0/24", "192.1.34.0/24"},
		},
		{
			desc:            "want IPv4 and receive IPv6 only",
			ipString:        []string{"fd00:20::/64", "2001:db9::/64"},
			wantIPv6:        false,
			expectCorrect:   nil,
			expectIncorrect: []string{"fd00:20::/64", "2001:db9::/64"},
		},
		{
			desc:            "want IPv6 and receive IPv6 only",
			ipString:        []string{"fd00:20::/64", "2001:db9::/64"},
			wantIPv6:        true,
			expectCorrect:   []string{"fd00:20::/64", "2001:db9::/64"},
			expectIncorrect: nil,
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.desc, func(t *testing.T) {
			ipFamily := v1.IPv4Protocol
			otherIPFamily := v1.IPv6Protocol

			if testcase.wantIPv6 {
				ipFamily = v1.IPv6Protocol
				otherIPFamily = v1.IPv4Protocol
			}

			cidrMap := MapCIDRsByIPFamily(testcase.ipString)

			var cidrStr []string
			for _, cidr := range cidrMap[ipFamily] {
				cidrStr = append(cidrStr, cidr.String())
			}
			var cidrStrOther []string
			for _, cidr := range cidrMap[otherIPFamily] {
				cidrStrOther = append(cidrStrOther, cidr.String())
			}

			if !reflect.DeepEqual(testcase.expectCorrect, cidrStr) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectCorrect, cidrStr)
			}
			if !reflect.DeepEqual(testcase.expectIncorrect, cidrStrOther) {
				t.Errorf("Test %v failed: expected %v, got %v", testcase.desc, testcase.expectIncorrect, cidrStrOther)
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

type fakeClosable struct {
	closed bool
}

func (c *fakeClosable) Close() error {
	c.closed = true
	return nil
}

func TestRevertPorts(t *testing.T) {
	testCases := []struct {
		replacementPorts []netutils.LocalPort
		existingPorts    []netutils.LocalPort
		expectToBeClose  []bool
	}{
		{
			replacementPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts:   []netutils.LocalPort{},
			expectToBeClose: []bool{true, true, true},
		},
		{
			replacementPorts: []netutils.LocalPort{},
			existingPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{},
		},
		{
			replacementPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, false, false},
		},
		{
			replacementPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, true, false},
		},
		{
			replacementPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []netutils.LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
				{Port: 5004},
			},
			expectToBeClose: []bool{false, false, false},
		},
	}

	for i, tc := range testCases {
		replacementPortsMap := make(map[netutils.LocalPort]netutils.Closeable)
		for _, lp := range tc.replacementPorts {
			replacementPortsMap[lp] = &fakeClosable{}
		}
		existingPortsMap := make(map[netutils.LocalPort]netutils.Closeable)
		for _, lp := range tc.existingPorts {
			existingPortsMap[lp] = &fakeClosable{}
		}
		RevertPorts(replacementPortsMap, existingPortsMap)
		for j, expectation := range tc.expectToBeClose {
			if replacementPortsMap[tc.replacementPorts[j]].(*fakeClosable).closed != expectation {
				t.Errorf("Expect replacement localport %v to be %v in test case %v", tc.replacementPorts[j], expectation, i)
			}
		}
		for _, lp := range tc.existingPorts {
			if existingPortsMap[lp].(*fakeClosable).closed {
				t.Errorf("Expect existing localport %v to be false in test case %v", lp, i)
			}
		}
	}
}

func mustParseIPAddr(str string) net.Addr {
	a, err := net.ResolveIPAddr("ip", str)
	if err != nil {
		panic("mustParseIPAddr")
	}
	return a
}
func mustParseIPNet(str string) net.Addr {
	_, n, err := netutils.ParseCIDRSloppy(str)
	if err != nil {
		panic("mustParseIPNet")
	}
	return n
}
func mustParseUnix(str string) net.Addr {
	n, err := net.ResolveUnixAddr("unix", str)
	if err != nil {
		panic("mustParseUnix")
	}
	return n
}

type cidrValidator struct {
	cidr *net.IPNet
}

func (v *cidrValidator) isValid(ip net.IP) bool {
	return v.cidr.Contains(ip)
}
func newCidrValidator(cidr string) func(ip net.IP) bool {
	_, n, err := netutils.ParseCIDRSloppy(cidr)
	if err != nil {
		panic("mustParseIPNet")
	}
	obj := cidrValidator{n}
	return obj.isValid
}

func TestAddressSet(t *testing.T) {
	testCases := []struct {
		name      string
		validator func(ip net.IP) bool
		input     []net.Addr
		expected  sets.Set[string]
	}{
		{
			"Empty",
			func(ip net.IP) bool { return false },
			nil,
			nil,
		},
		{
			"Reject IPAddr x 2",
			func(ip net.IP) bool { return false },
			[]net.Addr{
				mustParseIPAddr("8.8.8.8"),
				mustParseIPAddr("1000::"),
			},
			nil,
		},
		{
			"Accept IPAddr x 2",
			func(ip net.IP) bool { return true },
			[]net.Addr{
				mustParseIPAddr("8.8.8.8"),
				mustParseIPAddr("1000::"),
			},
			sets.New("8.8.8.8", "1000::"),
		},
		{
			"Accept IPNet x 2",
			func(ip net.IP) bool { return true },
			[]net.Addr{
				mustParseIPNet("8.8.8.8/32"),
				mustParseIPNet("1000::/128"),
			},
			sets.New("8.8.8.8", "1000::"),
		},
		{
			"Accept Unix x 2",
			func(ip net.IP) bool { return true },
			[]net.Addr{
				mustParseUnix("/tmp/sock1"),
				mustParseUnix("/tmp/sock2"),
			},
			nil,
		},
		{
			"Cidr IPv4",
			newCidrValidator("192.168.1.0/24"),
			[]net.Addr{
				mustParseIPAddr("8.8.8.8"),
				mustParseIPAddr("1000::"),
				mustParseIPAddr("192.168.1.1"),
			},
			sets.New("192.168.1.1"),
		},
		{
			"Cidr IPv6",
			newCidrValidator("1000::/64"),
			[]net.Addr{
				mustParseIPAddr("8.8.8.8"),
				mustParseIPAddr("1000::"),
				mustParseIPAddr("192.168.1.1"),
			},
			sets.New("1000::"),
		},
	}

	for _, tc := range testCases {
		if !tc.expected.Equal(AddressSet(tc.validator, tc.input)) {
			t.Errorf("%s", tc.name)
		}
	}
}

func TestIsZeroCIDR(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "invalide cidr",
			input:    "",
			expected: false,
		},
		{
			name:     "ipv4 cidr",
			input:    "172.10.0.0/16",
			expected: false,
		},
		{
			name:     "ipv4 zero cidr",
			input:    IPv4ZeroCIDR,
			expected: true,
		},
		{
			name:     "ipv6 cidr",
			input:    "::/128",
			expected: false,
		},
		{
			name:     "ipv6 zero cidr",
			input:    IPv6ZeroCIDR,
			expected: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsZeroCIDR(tc.input); tc.expected != got {
				t.Errorf("IsZeroCIDR() = %t, want %t", got, tc.expected)
			}
		})
	}
}
