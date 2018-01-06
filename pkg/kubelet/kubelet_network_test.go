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

package kubelet

import (
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNetworkHostGetsPodNotFound(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	actualPod, _ := nh.GetPodByName("", "")
	if actualPod != nil {
		t.Fatalf("Was expected nil, received %v instead", actualPod)
	}
}

func TestNetworkHostGetsKubeClient(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetKubeClient() != testKubelet.fakeKubeClient {
		t.Fatalf("NetworkHost client does not match testKubelet's client")
	}
}

func TestNetworkHostGetsRuntime(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetRuntime() != testKubelet.fakeRuntime {
		t.Fatalf("NetworkHost runtime does not match testKubelet's runtime")
	}
}

func TestNetworkHostSupportsLegacyFeatures(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.SupportsLegacyFeatures() == false {
		t.Fatalf("SupportsLegacyFeatures should not be false")
	}
}

func TestNoOpHostGetsName(t *testing.T) {
	nh := NoOpLegacyHost{}
	pod, err := nh.GetPodByName("", "")
	if pod != nil && err != true {
		t.Fatalf("noOpLegacyHost getpodbyname expected to be nil and true")
	}
}

func TestNoOpHostGetsKubeClient(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetKubeClient() != nil {
		t.Fatalf("noOpLegacyHost client expected to be nil")
	}
}

func TestNoOpHostGetsRuntime(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetRuntime() != nil {
		t.Fatalf("noOpLegacyHost runtime expected to be nil")
	}
}

func TestNoOpHostSupportsLegacyFeatures(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.SupportsLegacyFeatures() != false {
		t.Fatalf("noOpLegacyHost legacy features expected to be false")
	}
}

func TestNodeIPParam(t *testing.T) {
	type test struct {
		nodeIP   string
		success  bool
		testName string
	}
	tests := []test{
		{
			nodeIP:   "",
			success:  false,
			testName: "IP not set",
		},
		{
			nodeIP:   "127.0.0.1",
			success:  false,
			testName: "IPv4 loopback address",
		},
		{
			nodeIP:   "::1",
			success:  false,
			testName: "IPv6 loopback address",
		},
		{
			nodeIP:   "224.0.0.1",
			success:  false,
			testName: "multicast IPv4 address",
		},
		{
			nodeIP:   "ff00::1",
			success:  false,
			testName: "multicast IPv6 address",
		},
		{
			nodeIP:   "169.254.0.1",
			success:  false,
			testName: "IPv4 link-local unicast address",
		},
		{
			nodeIP:   "fe80::0202:b3ff:fe1e:8329",
			success:  false,
			testName: "IPv6 link-local unicast address",
		},
		{
			nodeIP:   "0.0.0.0",
			success:  false,
			testName: "Unspecified IPv4 address",
		},
		{
			nodeIP:   "::",
			success:  false,
			testName: "Unspecified IPv6 address",
		},
		{
			nodeIP:   "1.2.3.4",
			success:  false,
			testName: "IPv4 address that doesn't belong to host",
		},
	}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		assert.Error(t, err, fmt.Sprintf(
			"Unable to obtain a list of the node's unicast interface addresses."))
	}
	for _, addr := range addrs {
		var ip net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip.IsLoopback() || ip.IsLinkLocalUnicast() {
			break
		}
		successTest := test{
			nodeIP:   ip.String(),
			success:  true,
			testName: fmt.Sprintf("Success test case for address %s", ip.String()),
		}
		tests = append(tests, successTest)
	}
	for _, test := range tests {
		err := validateNodeIP(net.ParseIP(test.nodeIP))
		if test.success {
			assert.NoError(t, err, "test %s", test.testName)
		} else {
			assert.Error(t, err, fmt.Sprintf("test %s", test.testName))
		}
	}
}

func TestGetIPTablesMark(t *testing.T) {
	tests := []struct {
		bit    int
		expect string
	}{
		{
			14,
			"0x00004000/0x00004000",
		},
		{
			15,
			"0x00008000/0x00008000",
		},
	}
	for _, tc := range tests {
		res := getIPTablesMark(tc.bit)
		assert.Equal(t, tc.expect, res, "input %d", tc.bit)
	}
}
