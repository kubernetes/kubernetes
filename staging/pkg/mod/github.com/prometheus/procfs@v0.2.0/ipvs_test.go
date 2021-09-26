// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"net"
	"testing"
)

var (
	expectedIPVSStats = IPVSStats{
		Connections:     23765872,
		IncomingPackets: 3811989221,
		OutgoingPackets: 0,
		IncomingBytes:   89991519156915,
		OutgoingBytes:   0,
	}
	expectedIPVSBackendStatuses = []IPVSBackendStatus{
		{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.82.22"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     2,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.83.24"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     2,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.83.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     1,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.84.22"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.82.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    1499,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.50.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    1498,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.55"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.50.26"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("192.168.0.55"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.49.32"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    0,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("2620::1"),
			LocalPort:     80,
			RemoteAddress: net.ParseIP("2620::2"),
			RemotePort:    80,
			Proto:         "TCP",
			Weight:        1,
			ActiveConn:    0,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("2620::1"),
			LocalPort:     80,
			RemoteAddress: net.ParseIP("2620::3"),
			RemotePort:    80,
			Proto:         "TCP",
			Weight:        1,
			ActiveConn:    0,
			InactConn:     0,
		},
		{
			LocalAddress:  net.ParseIP("2620::1"),
			LocalPort:     80,
			RemoteAddress: net.ParseIP("2620::4"),
			RemotePort:    80,
			Proto:         "TCP",
			Weight:        1,
			ActiveConn:    1,
			InactConn:     1,
		},
		{
			LocalMark:     "10001000",
			RemoteAddress: net.ParseIP("192.168.50.26"),
			RemotePort:    3306,
			Proto:         "FWM",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     1,
		},
		{
			LocalMark:     "10001000",
			RemoteAddress: net.ParseIP("192.168.50.21"),
			RemotePort:    3306,
			Proto:         "FWM",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     2,
		},
	}
)

func TestIPVSStats(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := fs.IPVSStats()
	if err != nil {
		t.Fatal(err)
	}

	if stats != expectedIPVSStats {
		t.Errorf("want %+v, have %+v", expectedIPVSStats, stats)
	}
}

func TestParseIPPort(t *testing.T) {
	ip := net.ParseIP("192.168.0.22")
	port := uint16(3306)

	gotIP, gotPort, err := parseIPPort("C0A80016:0CEA")
	if err != nil {
		t.Fatal(err)
	}
	if !(gotIP.Equal(ip) && port == gotPort) {
		t.Errorf("want %s:%d, have %s:%d", ip, port, gotIP, gotPort)
	}
}

func TestParseIPPortInvalid(t *testing.T) {
	testcases := []string{
		"",
		"C0A80016",
		"C0A800:1234",
		"FOOBARBA:1234",
		"C0A80016:0CEA:1234",
	}

	for _, s := range testcases {
		ip, port, err := parseIPPort(s)
		if ip != nil || port != uint16(0) || err == nil {
			t.Errorf("Expected error for input %s, have ip = %s, port = %v, err = %v", s, ip, port, err)
		}
	}
}

func TestParseIPPortIPv6(t *testing.T) {
	ip := net.ParseIP("dead:beef::1")
	port := uint16(8080)

	gotIP, gotPort, err := parseIPPort("[DEAD:BEEF:0000:0000:0000:0000:0000:0001]:1F90")
	if err != nil {
		t.Fatal(err)
	}
	if !(gotIP.Equal(ip) && port == gotPort) {
		t.Errorf("want %s:%d, have %s:%d", ip, port, gotIP, gotPort)
	}
}

func TestIPVSBackendStatus(t *testing.T) {
	backendStats, err := getProcFixtures(t).IPVSBackendStatus()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := len(expectedIPVSBackendStatuses), len(backendStats); want != have {
		t.Fatalf("want %d backend statuses, have %d", want, have)
	}

	for idx, expect := range expectedIPVSBackendStatuses {
		if !backendStats[idx].LocalAddress.Equal(expect.LocalAddress) {
			t.Errorf("want LocalAddress %s, have %s", expect.LocalAddress, backendStats[idx].LocalAddress)
		}
		if backendStats[idx].LocalPort != expect.LocalPort {
			t.Errorf("want LocalPort %d, have %d", expect.LocalPort, backendStats[idx].LocalPort)
		}
		if !backendStats[idx].RemoteAddress.Equal(expect.RemoteAddress) {
			t.Errorf("want RemoteAddress %s, have %s", expect.RemoteAddress, backendStats[idx].RemoteAddress)
		}
		if backendStats[idx].RemotePort != expect.RemotePort {
			t.Errorf("want RemotePort %d, have %d", expect.RemotePort, backendStats[idx].RemotePort)
		}
		if backendStats[idx].Proto != expect.Proto {
			t.Errorf("want Proto %s, have %s", expect.Proto, backendStats[idx].Proto)
		}
		if backendStats[idx].Weight != expect.Weight {
			t.Errorf("want Weight %d, have %d", expect.Weight, backendStats[idx].Weight)
		}
		if backendStats[idx].ActiveConn != expect.ActiveConn {
			t.Errorf("want ActiveConn %d, have %d", expect.ActiveConn, backendStats[idx].ActiveConn)
		}
		if backendStats[idx].InactConn != expect.InactConn {
			t.Errorf("want InactConn %d, have %d", expect.InactConn, backendStats[idx].InactConn)
		}
	}
}
