// Copyright 2012 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: jsing@google.com (Joel Sing)

package ipvs

import (
	"bytes"
	"net"
	"reflect"
	"syscall"
	"testing"

	"github.com/google/seesaw/netlink"
)

var testStats = Stats{
	Connections: 1234,
	PacketsIn:   100000,
	PacketsOut:  200000,
	BytesIn:     300000,
	BytesOut:    400000,
	CPS:         10,
	PPSIn:       100,
	PPSOut:      200,
	BPSIn:       300,
	BPSOut:      400,
}

var ipvsServiceTests = []struct {
	desc        string
	ipvsService ipvsService
	want        Service
}{
	{
		"Zeroed structs",
		ipvsService{},
		Service{
			Address:    net.ParseIP("::0"),
			Statistics: &ServiceStats{},
		},
	},
	{
		"IPv4 1.2.3.4 with TCP/80 using wlc",
		ipvsService{
			Protocol:          syscall.IPPROTO_TCP,
			Port:              80,
			FirewallMark:      0,
			Scheduler:         "wlc",
			Flags:             0,
			Timeout:           0,
			Netmask:           0xffffffff,
			Stats:             &ServiceStats{Stats: testStats},
			AddrFamily:        syscall.AF_INET,
			Address:           net.ParseIP("1.2.3.4"),
			PersistenceEngine: "",
		},
		Service{
			Address:           net.ParseIP("1.2.3.4"),
			Protocol:          syscall.IPPROTO_TCP,
			Port:              80,
			FirewallMark:      0,
			Scheduler:         "wlc",
			Flags:             0,
			Timeout:           0,
			PersistenceEngine: "",
			Statistics:        &ServiceStats{Stats: testStats},
		},
	},
	{
		"IPv6 2012::beef with UDP/33434 using wlc",
		ipvsService{
			Protocol:          syscall.IPPROTO_UDP,
			Port:              33434,
			FirewallMark:      0,
			Scheduler:         "wlc",
			Flags:             1234,
			Timeout:           100,
			Netmask:           128,
			Stats:             &ServiceStats{Stats: testStats},
			AddrFamily:        syscall.AF_INET6,
			Address:           net.ParseIP("2012::beef"),
			PersistenceEngine: "",
		},
		Service{
			Address:           net.ParseIP("2012::beef"),
			Protocol:          syscall.IPPROTO_UDP,
			Port:              33434,
			FirewallMark:      0,
			Scheduler:         "wlc",
			Flags:             1234,
			Timeout:           100,
			PersistenceEngine: "",
			Statistics:        &ServiceStats{Stats: testStats},
		},
	},
	{
		"IPv4 FWM 4 using lc",
		ipvsService{
			Protocol:          0,
			Port:              0,
			FirewallMark:      4,
			Scheduler:         "lc",
			Flags:             0,
			Timeout:           0,
			Netmask:           0xffffffff,
			Stats:             &ServiceStats{Stats: testStats},
			AddrFamily:        syscall.AF_INET,
			Address:           nil,
			PersistenceEngine: "",
		},
		Service{
			Address:           net.ParseIP("0.0.0.0"),
			Protocol:          0,
			Port:              0,
			FirewallMark:      4,
			Scheduler:         "lc",
			Flags:             0,
			Timeout:           0,
			PersistenceEngine: "",
			Statistics:        &ServiceStats{Stats: testStats},
		},
	},
	{
		"IPv6 FWM 6 using wrr",
		ipvsService{
			Protocol:          0,
			Port:              0,
			FirewallMark:      6,
			Scheduler:         "wrr",
			Flags:             0,
			Timeout:           0,
			Netmask:           0xffffffff,
			Stats:             &ServiceStats{Stats: testStats},
			AddrFamily:        syscall.AF_INET6,
			Address:           nil,
			PersistenceEngine: "",
		},
		Service{
			Address:           net.ParseIP("::0"),
			Protocol:          0,
			Port:              0,
			FirewallMark:      6,
			Scheduler:         "wrr",
			Flags:             0,
			Timeout:           0,
			PersistenceEngine: "",
			Statistics:        &ServiceStats{Stats: testStats},
		},
	},
}

func TestIPVSServiceToService(t *testing.T) {
	for _, test := range ipvsServiceTests {
		got := test.ipvsService.toService()
		if !reflect.DeepEqual(*got, test.want) {
			t.Errorf("toService() failed for %s - got %#v, want %#v",
				test.desc, *got, test.want)
		}
	}
}

var ipvsDestinationTests = []struct {
	desc            string
	ipvsDestination ipvsDestination
	want            Destination
}{
	{
		"Zeroed structs",
		ipvsDestination{},
		Destination{
			Statistics: &DestinationStats{},
		},
	},
	{
		"IPv4 1.2.4.4 with port 54321",
		ipvsDestination{
			Port:           54321,
			Flags:          0,
			Weight:         1,
			UpperThreshold: 100000,
			LowerThreshold: 10000,
			ActiveConns:    12345678,
			InactiveConns:  87654321,
			PersistConns:   1234,
			Stats:          &DestinationStats{Stats: testStats},
			Address:        net.ParseIP("1.2.3.4"),
		},
		Destination{
			Address:        net.ParseIP("1.2.3.4"),
			Port:           54321,
			Weight:         1,
			Flags:          0,
			LowerThreshold: 10000,
			UpperThreshold: 100000,
			Statistics: &DestinationStats{
				Stats:         testStats,
				ActiveConns:   12345678,
				InactiveConns: 87654321,
				PersistConns:  1234,
			},
		},
	},
	{
		"IPv6 2002::cafe with port 53",
		ipvsDestination{
			Port:           53,
			Flags:          0xf0f0f0f0,
			Weight:         1,
			UpperThreshold: 0,
			LowerThreshold: 0,
			ActiveConns:    12345678,
			InactiveConns:  87654321,
			PersistConns:   1234,
			Stats:          &DestinationStats{Stats: testStats},
			Address:        net.ParseIP("2002::cafe"),
		},
		Destination{
			Address:        net.ParseIP("2002::cafe"),
			Port:           53,
			Weight:         1,
			Flags:          0xf0f0f0f0,
			LowerThreshold: 0,
			UpperThreshold: 0,
			Statistics: &DestinationStats{
				Stats:         testStats,
				ActiveConns:   12345678,
				InactiveConns: 87654321,
				PersistConns:  1234,
			},
		},
	},
}

func TestIPVSDestinationToDestination(t *testing.T) {
	for _, test := range ipvsDestinationTests {
		got := test.ipvsDestination.toDestination()
		if !reflect.DeepEqual(*got, test.want) {
			t.Errorf("toDestination() failed for %s - got %#v, want %#v",
				test.desc, *got, test.want)
		}
	}
}

var serviceTests = []struct {
	desc    string
	service Service
	want    ipvsService
}{
	{
		"Zeroed structs",
		Service{},
		ipvsService{
			AddrFamily: syscall.AF_INET6,
			Netmask:    128,
		},
	},
	{
		"IPv4 1.2.3.4 with TCP/54321 using wlc",
		Service{
			Address:      net.ParseIP("1.2.3.4"),
			Protocol:     syscall.IPPROTO_TCP,
			Port:         54321,
			FirewallMark: 1,
			Scheduler:    "wlc",
			Flags:        0,
			Timeout:      100000,
		},
		ipvsService{
			Protocol:          syscall.IPPROTO_TCP,
			Port:              54321,
			FirewallMark:      1,
			Scheduler:         "wlc",
			Flags:             0,
			Timeout:           100000,
			Netmask:           0xffffffff,
			AddrFamily:        syscall.AF_INET,
			Address:           net.ParseIP("1.2.3.4"),
			PersistenceEngine: "",
		},
	},
	{
		"IPv6 2002::cafe with UDP/53",
		Service{
			Address:      net.ParseIP("2002::cafe"),
			Protocol:     syscall.IPPROTO_UDP,
			Port:         53,
			FirewallMark: 0,
			Scheduler:    "xxxxxxxxxxxxxxxx",
			Flags:        0xf0f0f0f0,
			Timeout:      0,
		},
		ipvsService{
			Protocol:          syscall.IPPROTO_UDP,
			Port:              53,
			FirewallMark:      0,
			Scheduler:         "xxxxxxxxxxxxxxxx",
			Flags:             0xf0f0f0f0,
			Timeout:           0,
			Netmask:           128,
			AddrFamily:        syscall.AF_INET6,
			Address:           net.ParseIP("2002::cafe"),
			PersistenceEngine: "",
		},
	},
}

func TestServiceToIPVSService(t *testing.T) {
	for _, test := range serviceTests {
		got := newIPVSService(&test.service)
		if !reflect.DeepEqual(*got, test.want) {
			t.Errorf("toIPVSService() failed for %s - got %#v, want %#v)",
				test.desc, *got, test.want)
		}
	}
}

var destinationTests = []struct {
	desc        string
	destination Destination
	want        ipvsDestination
}{
	{
		"Zeroed structs",
		Destination{},
		ipvsDestination{},
	},
	{
		"IPv4 1.2.4.4 with port 54321",
		Destination{
			Address:        net.ParseIP("1.2.3.4"),
			Port:           54321,
			Weight:         2,
			Flags:          0,
			LowerThreshold: 10000,
			UpperThreshold: 100000,
		},
		ipvsDestination{
			Port:           54321,
			Flags:          0,
			Weight:         2,
			UpperThreshold: 100000,
			LowerThreshold: 10000,
			Address:        net.ParseIP("1.2.3.4"),
		},
	},
	{
		"IPv6 2002::cafe with port 53",
		Destination{
			Address:        net.ParseIP("2002::cafe"),
			Port:           53,
			Weight:         3,
			Flags:          0xf0f0f0f0,
			LowerThreshold: 0,
			UpperThreshold: 0,
		},
		ipvsDestination{
			Port:           53,
			Flags:          0xf0f0f0f0,
			Weight:         3,
			UpperThreshold: 0,
			LowerThreshold: 0,
			Address:        net.ParseIP("2002::cafe"),
		},
	},
}

func TestDestinationToIPVSDestination(t *testing.T) {
	for _, test := range destinationTests {
		got := newIPVSDestination(&test.destination)
		if !reflect.DeepEqual(*got, test.want) {
			t.Errorf("toIPVSDestination() failed for %s - got %#v, want %#v)",
				test.desc, *got, test.want)
		}
	}
}

const (
	nlTestCommand = 1
	nlTestFamily  = 25
)

var (
	nlmIPVSAddDestination = []byte{
		0x6c, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x58, 0x00, 0x02, 0x00,
		0x14, 0x00, 0x01, 0x00, 0x20, 0x02, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0xca, 0xfe, 0x06, 0x00, 0x02, 0x00,
		0x00, 0x35, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
		0xf4, 0xf3, 0xf2, 0xf1, 0x08, 0x00, 0x04, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x05, 0x00,
		0xd0, 0x07, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0xe8, 0x03, 0x00, 0x00, 0x08, 0x00, 0x07, 0x00,
		0x4e, 0x61, 0xbc, 0x00, 0x08, 0x00, 0x08, 0x00,
		0xb1, 0x7f, 0x39, 0x05, 0x08, 0x00, 0x09, 0x00,
		0xd2, 0x04, 0x00, 0x00,
	}

	nlmIPVSDestination = []byte{
		0xc8, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0xb4, 0x00, 0x02, 0x00,
		0x14, 0x00, 0x01, 0x00, 0x20, 0x02, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0xca, 0xfe, 0x06, 0x00, 0x02, 0x00,
		0x00, 0x35, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
		0xf4, 0xf3, 0xf2, 0xf1, 0x08, 0x00, 0x04, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x05, 0x00,
		0xd0, 0x07, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0xe8, 0x03, 0x00, 0x00, 0x08, 0x00, 0x07, 0x00,
		0x4e, 0x61, 0xbc, 0x00, 0x08, 0x00, 0x08, 0x00,
		0xb1, 0x7f, 0x39, 0x05, 0x08, 0x00, 0x09, 0x00,
		0xd2, 0x04, 0x00, 0x00, 0x5c, 0x00, 0x0a, 0x00,
		0x08, 0x00, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x02, 0x00, 0x0c, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0c, 0x00, 0x04, 0x00, 0x1a, 0x04, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x05, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x07, 0x00, 0x01, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x09, 0x00, 0x54, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
	}

	nlmIPVSAddService = []byte{
		0x68, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x54, 0x00, 0x01, 0x00,
		0x06, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
		0x06, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00,
		0x14, 0x00, 0x03, 0x00, 0x01, 0x01, 0x01, 0x01,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x04, 0x00,
		0x50, 0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0x77, 0x6c, 0x63, 0x00, 0x0c, 0x00, 0x07, 0x00,
		0xf4, 0xf3, 0xf2, 0xf1, 0xff, 0xff, 0xff, 0xff,
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x09, 0x00, 0xff, 0xff, 0xff, 0xff,
	}

	nlmIPVSService = []byte{
		0xc4, 0x00, 0x00, 0x00, 0x16, 0x00, 0x02, 0x00,
		0xb0, 0xb3, 0xc8, 0x55, 0x79, 0x02, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0xb0, 0x00, 0x01, 0x00,
		0x06, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
		0x06, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00,
		0x14, 0x00, 0x03, 0x00, 0x01, 0x01, 0x01, 0x01,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x04, 0x00,
		0x50, 0x00, 0x00, 0x00, 0x07, 0x00, 0x06, 0x00,
		0x77, 0x6c, 0x63, 0x00, 0x0c, 0x00, 0x07, 0x00,
		0xf4, 0xf3, 0xf2, 0xf1, 0xff, 0xff, 0xff, 0xff,
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x09, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x5c, 0x00, 0x0a, 0x00, 0x08, 0x00, 0x01, 0x00,
		0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x02, 0x00,
		0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x04, 0x00,
		0x1a, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0c, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x07, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x09, 0x00,
		0x54, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0a, 0x00,
		0x00, 0x00, 0x00, 0x00,
	}

	testIPVSDestination = &ipvsDestination{
		Address:        net.ParseIP("2002::cafe"),
		Port:           53,
		Flags:          0xf1f2f3f4,
		Weight:         1,
		UpperThreshold: 2000,
		LowerThreshold: 1000,
		ActiveConns:    12345678,
		InactiveConns:  87654321,
		PersistConns:   1234,
		Stats:          nil,
	}

	testIPVSService = &ipvsService{
		AddrFamily:        syscall.AF_INET,
		Protocol:          syscall.IPPROTO_TCP,
		Address:           net.IPv4(1, 1, 1, 1),
		Port:              0x5000,
		FirewallMark:      0x0,
		Scheduler:         "wlc",
		Flags:             0xf1f2f3f4,
		Timeout:           0x0,
		Netmask:           0xffffffff,
		Stats:             nil,
		PersistenceEngine: "",
	}

	testIPVSStats = Stats{
		Connections: 0x3,
		PacketsIn:   0xc,
		PacketsOut:  0x0,
		BytesIn:     0x41a,
		BytesOut:    0x0,
		CPS:         0x0,
		PPSIn:       0x1,
		PPSOut:      0x0,
		BPSIn:       0x54,
		BPSOut:      0x0,
	}
)

func TestDestinationNetlinkMarshal(t *testing.T) {
	m, err := netlink.NewMessage(nlTestCommand, nlTestFamily, 0)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	ic := &ipvsCommand{Destination: testIPVSDestination}
	if err := m.Marshal(ic); err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	got, err := m.Bytes()
	if err != nil {
		t.Fatalf("Failed to get message bytes: %v", err)
	}
	if want := nlmIPVSAddDestination; !bytes.Equal(got, want) {
		t.Errorf("Got netlink bytes %#v, want %#v", got, want)
	}
}

func TestDestinationNetlinkUnmarshal(t *testing.T) {
	m, err := netlink.NewMessageFromBytes(nlmIPVSDestination)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	got := &ipvsCommand{}
	if err := m.Unmarshal(got); err != nil {
		t.Errorf("Failed to unmarshal message: %v", err)
	}

	want := *testIPVSDestination
	want.Stats = &DestinationStats{Stats: testIPVSStats}
	if !reflect.DeepEqual(got.Destination, &want) {
		t.Errorf("Got IPVS destination %#v, want %#v", got.Destination, &want)
	}
}

func TestServiceNetlinkMarshal(t *testing.T) {
	m, err := netlink.NewMessage(nlTestCommand, nlTestFamily, 0)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	ic := &ipvsCommand{Service: testIPVSService}
	if err := m.Marshal(ic); err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	got, err := m.Bytes()
	if err != nil {
		t.Fatalf("Failed to get message bytes: %v", err)
	}
	if want := nlmIPVSAddService; !bytes.Equal(got, want) {
		t.Errorf("Got netlink bytes %#v, want %#v", got, want)
	}
}

func TestServiceNetlinkUnmarshal(t *testing.T) {
	m, err := netlink.NewMessageFromBytes(nlmIPVSService)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	got := &ipvsCommand{}
	if err := m.Unmarshal(got); err != nil {
		t.Fatalf("Failed to unmarshal message: %v", err)
	}

	want := *testIPVSService
	want.Stats = &ServiceStats{Stats: testIPVSStats}
	if !reflect.DeepEqual(got.Service, &want) {
		t.Errorf("Got IPVS service %#v, want %#v", got.Service, &want)
	}
}
