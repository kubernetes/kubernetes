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

package ipvs

import (
	"net"
	"testing"
)

func TestVirtualServerEqual(t *testing.T) {
	Tests := []struct {
		svcA   *VirtualServer
		svcB   *VirtualServer
		equal  bool
		reason string
	}{
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("10.20.30.40"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("10.20.30.41"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "IPv4 address not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("2017::beef"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "IPv6 address not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "TCP",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("2012::beeef"),
				Protocol:  "UDP",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "Protocol not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "TCP",
				Port:      8080,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "Port not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "rr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "wlc",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "Scheduler not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "rr",
				Flags:     2,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "rr",
				Flags:     3,
				Timeout:   0,
			},
			equal:  false,
			reason: "Flags not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   10800,
			},
			equal:  false,
			reason: "Timeout not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "rr",
				Flags:     0x1,
				Timeout:   10800,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "rr",
				Flags:     0x1,
				Timeout:   10800,
			},
			equal:  true,
			reason: "All fields equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "TCP",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("2012::beeef"),
				Protocol:  "SCTP",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
			equal:  false,
			reason: "Protocol not equal",
		},
		{
			svcA: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "SCTP",
				Port:      80,
				Scheduler: "rr",
				Flags:     0x1,
				Timeout:   10800,
			},
			svcB: &VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "SCTP",
				Port:      80,
				Scheduler: "rr",
				Flags:     0x1,
				Timeout:   10800,
			},
			equal:  true,
			reason: "All fields equal",
		},
	}

	for i := range Tests {
		equal := Tests[i].svcA.Equal(Tests[i].svcB)
		if equal != Tests[i].equal {
			t.Errorf("case: %d got %v, expected %v, reason: %s", i, equal, Tests[i].equal, Tests[i].reason)
		}
	}
}

func TestRealServerEqual(t *testing.T) {
	Tests := []struct {
		rsA    *RealServer
		rsB    *RealServer
		equal  bool
		reason string
	}{
		{
			rsA: &RealServer{
				Address: net.ParseIP("10.20.30.40"),
				Port:    80,
			},
			rsB: &RealServer{
				Address: net.ParseIP("10.20.30.41"),
				Port:    80,
			},
			equal:  false,
			reason: "IPv4 address not equal",
		},
		{
			rsA: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    80,
			},
			rsB: &RealServer{
				Address: net.ParseIP("2017::beef"),
				Port:    80,
			},
			equal:  false,
			reason: "IPv6 address not equal",
		},
		{
			rsA: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    80,
			},
			rsB: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    8080,
			},
			equal:  false,
			reason: "Port not equal",
		},
		{
			rsA: &RealServer{
				Address: net.ParseIP("1.2.3.4"),
				Port:    3080,
			},
			rsB: &RealServer{
				Address: net.ParseIP("1.2.3.4"),
				Port:    3080,
			},
			equal:  true,
			reason: "All fields equal",
		},
		{
			rsA: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    3080,
			},
			rsB: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    3080,
			},
			equal:  true,
			reason: "All fields equal",
		},
	}

	for i := range Tests {
		equal := Tests[i].rsA.Equal(Tests[i].rsB)
		if equal != Tests[i].equal {
			t.Errorf("case: %d got %v, expected %v, reason: %s", i, equal, Tests[i].equal, Tests[i].reason)
		}
	}
}

func TestFrontendServiceString(t *testing.T) {
	Tests := []struct {
		svc      *VirtualServer
		expected string
	}{
		{
			svc: &VirtualServer{
				Address:  net.ParseIP("10.20.30.40"),
				Protocol: "TCP",
				Port:     80,
			},
			expected: "10.20.30.40:80/TCP",
		},
		{
			svc: &VirtualServer{
				Address:  net.ParseIP("2012::beef"),
				Protocol: "UDP",
				Port:     8080,
			},
			expected: "[2012::beef]:8080/UDP",
		},
		{
			svc: &VirtualServer{
				Address:  net.ParseIP("10.20.30.41"),
				Protocol: "ESP",
				Port:     1234,
			},
			expected: "10.20.30.41:1234/ESP",
		},
	}

	for i := range Tests {
		if Tests[i].expected != Tests[i].svc.String() {
			t.Errorf("case: %d got %v, expected %v", i, Tests[i].svc.String(), Tests[i].expected)
		}
	}
}

func TestFrontendDestinationString(t *testing.T) {
	Tests := []struct {
		svc      *RealServer
		expected string
	}{
		{
			svc: &RealServer{
				Address: net.ParseIP("10.20.30.40"),
				Port:    80,
			},
			expected: "10.20.30.40:80",
		},
		{
			svc: &RealServer{
				Address: net.ParseIP("2012::beef"),
				Port:    8080,
			},
			expected: "[2012::beef]:8080",
		},
	}

	for i := range Tests {
		if Tests[i].expected != Tests[i].svc.String() {
			t.Errorf("case: %d got %v, expected %v", i, Tests[i].svc.String(), Tests[i].expected)
		}
	}
}
