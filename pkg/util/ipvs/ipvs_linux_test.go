// +build linux

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
	"fmt"
	"net"
	"reflect"
	"syscall"
	"testing"

	libipvs "github.com/docker/libnetwork/ipvs"
)

func Test_toVirtualServer(t *testing.T) {
	Tests := []struct {
		ipvsService   libipvs.Service
		virtualServer VirtualServer
		expectError   bool
		reason        string
	}{
		{
			libipvs.Service{
				Flags: 0x0,
			},
			VirtualServer{},
			true,
			fmt.Sprintf("IPVS Service Flags should be >= %d, got 0x0", FlagHashed),
		},
		{
			libipvs.Service{
				Flags: 0x1,
			},
			VirtualServer{},
			true,
			fmt.Sprintf("IPVS Service Flags should be >= %d, got 0x1", FlagHashed),
		},
		{
			libipvs.Service{
				Protocol:      syscall.IPPROTO_TCP,
				Port:          80,
				FWMark:        0,
				SchedName:     "",
				Flags:         uint32(FlagPersistent + FlagHashed),
				Timeout:       0,
				Netmask:       0xffffffff,
				AddressFamily: syscall.AF_INET,
				Address:       nil,
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("0.0.0.0"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "",
				Flags:     ServiceFlags(FlagPersistent),
				Timeout:   0,
			},
			false,
			"",
		},
		{
			libipvs.Service{
				Protocol:      syscall.IPPROTO_UDP,
				Port:          33434,
				FWMark:        0,
				SchedName:     "wlc",
				Flags:         uint32(0 + FlagHashed),
				Timeout:       100,
				Netmask:       128,
				AddressFamily: syscall.AF_INET6,
				Address:       net.ParseIP("2012::beef"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "UDP",
				Port:      33434,
				Scheduler: "wlc",
				Flags:     ServiceFlags(0),
				Timeout:   100,
			},
			false,
			"",
		},
		{
			libipvs.Service{
				Protocol:      0,
				Port:          0,
				FWMark:        0,
				SchedName:     "lc",
				Flags:         uint32(0 + FlagHashed),
				Timeout:       0,
				Netmask:       0xffffffff,
				AddressFamily: syscall.AF_INET,
				Address:       net.ParseIP("1.2.3.4"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "",
				Port:      0,
				Scheduler: "lc",
				Flags:     ServiceFlags(0),
				Timeout:   0,
			},
			false,
			"",
		},
		{
			libipvs.Service{
				Protocol:      0,
				Port:          0,
				FWMark:        0,
				SchedName:     "wrr",
				Flags:         uint32(FlagPersistent + FlagHashed),
				Timeout:       0,
				Netmask:       128,
				AddressFamily: syscall.AF_INET6,
				Address:       nil,
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("::0"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     ServiceFlags(FlagPersistent),
				Timeout:   0,
			},
			false,
			"",
		},
		{
			libipvs.Service{
				Protocol:      syscall.IPPROTO_SCTP,
				Port:          80,
				FWMark:        0,
				SchedName:     "",
				Flags:         uint32(FlagPersistent + FlagHashed),
				Timeout:       0,
				Netmask:       0xffffffff,
				AddressFamily: syscall.AF_INET,
				Address:       nil,
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("0.0.0.0"),
				Protocol:  "SCTP",
				Port:      80,
				Scheduler: "",
				Flags:     ServiceFlags(FlagPersistent),
				Timeout:   0,
			},
			false,
			"",
		},
	}

	for i := range Tests {
		got, err := toVirtualServer(&Tests[i].ipvsService)
		if Tests[i].expectError && err == nil {
			t.Errorf("case: %d, expected error: %s, got nil", i, Tests[i].reason)
		}
		if !Tests[i].expectError && err != nil {
			t.Errorf("case: %d, unexpected error: %v", i, err)
		}
		if got != nil && &Tests[i].virtualServer != nil {
			if !reflect.DeepEqual(*got, Tests[i].virtualServer) {
				t.Errorf("case: %d, got %#v, want %#v", i, *got, Tests[i].virtualServer)
			}
		}
	}
}

func Test_toIPVSService(t *testing.T) {
	Tests := []struct {
		ipvsService   libipvs.Service
		virtualServer VirtualServer
	}{
		{
			libipvs.Service{
				Protocol:      syscall.IPPROTO_TCP,
				Port:          80,
				FWMark:        0,
				SchedName:     "",
				Flags:         0,
				Timeout:       0,
				Netmask:       0xffffffff,
				AddressFamily: syscall.AF_INET,
				Address:       net.ParseIP("0.0.0.0"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("0.0.0.0"),
				Protocol:  "TCP",
				Port:      80,
				Scheduler: "",
				Flags:     0,
				Timeout:   0,
			},
		},
		{
			libipvs.Service{
				Protocol:      syscall.IPPROTO_UDP,
				Port:          33434,
				FWMark:        0,
				SchedName:     "wlc",
				Flags:         1234,
				Timeout:       100,
				Netmask:       128,
				AddressFamily: syscall.AF_INET6,
				Address:       net.ParseIP("2012::beef"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("2012::beef"),
				Protocol:  "UDP",
				Port:      33434,
				Scheduler: "wlc",
				Flags:     1234,
				Timeout:   100,
			},
		},
		{
			libipvs.Service{
				Protocol:      0,
				Port:          0,
				FWMark:        0,
				SchedName:     "lc",
				Flags:         0,
				Timeout:       0,
				Netmask:       0xffffffff,
				AddressFamily: syscall.AF_INET,
				Address:       net.ParseIP("1.2.3.4"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  "",
				Port:      0,
				Scheduler: "lc",
				Flags:     0,
				Timeout:   0,
			},
		},
		{
			libipvs.Service{
				Protocol:      0,
				Port:          0,
				FWMark:        0,
				SchedName:     "wrr",
				Flags:         0,
				Timeout:       0,
				Netmask:       128,
				AddressFamily: syscall.AF_INET6,
				Address:       net.ParseIP("::0"),
				PEName:        "",
			},
			VirtualServer{
				Address:   net.ParseIP("::0"),
				Protocol:  "",
				Port:      0,
				Scheduler: "wrr",
				Flags:     0,
				Timeout:   0,
			},
		},
	}

	for i := range Tests {
		got, err := toIPVSService(&Tests[i].virtualServer)
		if err != nil {
			t.Errorf("case: %d, unexpected error: %v", i, err)
		}
		if !reflect.DeepEqual(*got, Tests[i].ipvsService) {
			t.Errorf("case: %d - got %#v, want %#v", i, *got, Tests[i].ipvsService)
		}
	}
}

func Test_toRealServer(t *testing.T) {
	Tests := []struct {
		ipvsDestination libipvs.Destination
		realServer      RealServer
	}{
		{
			libipvs.Destination{
				Port:            54321,
				ConnectionFlags: 0,
				Weight:          1,
				Address:         net.ParseIP("1.2.3.4"),
			},
			RealServer{
				Address: net.ParseIP("1.2.3.4"),
				Port:    54321,
				Weight:  1,
			},
		},
		{
			libipvs.Destination{
				Port:            53,
				ConnectionFlags: 0,
				Weight:          1,
				Address:         net.ParseIP("2002::cafe"),
			},
			RealServer{
				Address: net.ParseIP("2002::cafe"),
				Port:    53,
				Weight:  1,
			},
		},
	}
	for i := range Tests {
		got, err := toRealServer(&Tests[i].ipvsDestination)
		if err != nil {
			t.Errorf("case %d unexpected error: %v", i, err)
		}
		if !reflect.DeepEqual(*got, Tests[i].realServer) {
			t.Errorf("case %d Failed to translate Destination - got %#v, want %#v", i, *got, Tests[i].realServer)
		}
	}
}

func Test_toIPVSDestination(t *testing.T) {
	Tests := []struct {
		realServer      RealServer
		ipvsDestination libipvs.Destination
	}{
		{
			RealServer{
				Address: net.ParseIP("1.2.3.4"),
				Port:    54321,
				Weight:  1,
			},
			libipvs.Destination{
				Port:            54321,
				ConnectionFlags: 0,
				Weight:          1,
				Address:         net.ParseIP("1.2.3.4"),
			},
		},
		{
			RealServer{
				Address: net.ParseIP("2002::cafe"),
				Port:    53,
				Weight:  1,
			},
			libipvs.Destination{
				Port:            53,
				ConnectionFlags: 0,
				Weight:          1,
				Address:         net.ParseIP("2002::cafe"),
			},
		},
	}
	for i := range Tests {
		got, err := toIPVSDestination(&Tests[i].realServer)
		if err != nil {
			t.Errorf("case %d unexpected error: %v", i, err)
		}
		if !reflect.DeepEqual(*got, Tests[i].ipvsDestination) {
			t.Errorf("case %d failed to translate Destination - got %#v, want %#v", i, *got, Tests[i].ipvsDestination)
		}
	}
}

func Test_stringToProtocol(t *testing.T) {
	tests := []string{
		"TCP", "UDP", "ICMP", "SCTP",
	}
	expected := []uint16{
		uint16(syscall.IPPROTO_TCP), uint16(syscall.IPPROTO_UDP), uint16(0), uint16(syscall.IPPROTO_SCTP),
	}
	for i := range tests {
		got := stringToProtocol(tests[i])
		if got != expected[i] {
			t.Errorf("stringToProtocol() failed - got %#v, want %#v",
				got, expected[i])
		}
	}
}

func Test_protocolToString(t *testing.T) {
	tests := []Protocol{
		syscall.IPPROTO_TCP, syscall.IPPROTO_UDP, Protocol(0), syscall.IPPROTO_SCTP,
	}
	expected := []string{
		"TCP", "UDP", "", "SCTP",
	}
	for i := range tests {
		got := protocolToString(tests[i])
		if got != expected[i] {
			t.Errorf("protocolToString() failed - got %#v, want %#v",
				got, expected[i])
		}
	}
}
