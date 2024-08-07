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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/utils/ktesting"
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

func makeNodeWithAddress(name, primaryIP string) *v1.Node {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{},
		},
	}

	if primaryIP != "" {
		node.Status.Addresses = append(node.Status.Addresses,
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: primaryIP},
		)
	}

	return node
}

// Test that getNodeIPs retries on failure
func Test_GetNodeIPs(t *testing.T) {
	var chans [3]chan error

	client := clientsetfake.NewSimpleClientset(
		// node1 initially has no IP address.
		makeNodeWithAddress("node1", ""),

		// node2 initially has an invalid IP address.
		makeNodeWithAddress("node2", "invalid-ip"),

		// node3 initially does not exist.
	)

	for i := range chans {
		chans[i] = make(chan error)
		ch := chans[i]
		nodeName := fmt.Sprintf("node%d", i+1)
		expectIP := fmt.Sprintf("192.168.0.%d", i+1)
		go func() {
			_, ctx := ktesting.NewTestContext(t)
			ips := GetNodeIPs(ctx, client, nodeName)
			if len(ips) == 0 {
				ch <- fmt.Errorf("expected IP %s for %s but got nil", expectIP, nodeName)
			} else if ips[0].String() != expectIP {
				ch <- fmt.Errorf("expected IP %s for %s but got %s", expectIP, nodeName, ips[0].String())
			} else if len(ips) != 1 {
				ch <- fmt.Errorf("expected IP %s for %s but got multiple IPs", expectIP, nodeName)
			}
			close(ch)
		}()
	}

	// Give the goroutines time to fetch the bad/non-existent nodes, then fix them.
	time.Sleep(1200 * time.Millisecond)

	_, _ = client.CoreV1().Nodes().UpdateStatus(context.TODO(),
		makeNodeWithAddress("node1", "192.168.0.1"),
		metav1.UpdateOptions{},
	)
	_, _ = client.CoreV1().Nodes().UpdateStatus(context.TODO(),
		makeNodeWithAddress("node2", "192.168.0.2"),
		metav1.UpdateOptions{},
	)
	_, _ = client.CoreV1().Nodes().Create(context.TODO(),
		makeNodeWithAddress("node3", "192.168.0.3"),
		metav1.CreateOptions{},
	)

	// Ensure each GetNodeIP completed as expected
	for i := range chans {
		err := <-chans[i]
		if err != nil {
			t.Error(err.Error())
		}
	}
}
