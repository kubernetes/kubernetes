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
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.82.22"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     2,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.83.24"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     2,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.22"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.83.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    248,
			InactConn:     1,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.84.22"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     0,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.82.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    1499,
			InactConn:     0,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.57"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.50.21"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    1498,
			InactConn:     0,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.55"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.50.26"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        0,
			ActiveConn:    0,
			InactConn:     0,
		},
		IPVSBackendStatus{
			LocalAddress:  net.ParseIP("192.168.0.55"),
			LocalPort:     3306,
			RemoteAddress: net.ParseIP("192.168.49.32"),
			RemotePort:    3306,
			Proto:         "TCP",
			Weight:        100,
			ActiveConn:    0,
			InactConn:     0,
		},
	}
)

func TestIPVSStats(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}
	stats, err := fs.NewIPVSStats()
	if err != nil {
		t.Fatal(err)
	}

	if stats != expectedIPVSStats {
		t.Errorf("want %+v, got %+v", expectedIPVSStats, stats)
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
		t.Errorf("want %s:%d, got %s:%d", ip, port, gotIP, gotPort)
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
			t.Errorf("Expected error for input %s, got ip = %s, port = %v, err = %v", s, ip, port, err)
		}
	}
}

func TestParseIPPortIPv6(t *testing.T) {
	ip := net.ParseIP("dead:beef::1")
	port := uint16(8080)

	gotIP, gotPort, err := parseIPPort("DEADBEEF000000000000000000000001:1F90")
	if err != nil {
		t.Fatal(err)
	}
	if !(gotIP.Equal(ip) && port == gotPort) {
		t.Errorf("want %s:%d, got %s:%d", ip, port, gotIP, gotPort)
	}

}

func TestIPVSBackendStatus(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}

	backendStats, err := fs.NewIPVSBackendStatus()
	if err != nil {
		t.Fatal(err)
	}

	for idx, expect := range expectedIPVSBackendStatuses {
		if !backendStats[idx].LocalAddress.Equal(expect.LocalAddress) {
			t.Errorf("expected LocalAddress %s, got %s", expect.LocalAddress, backendStats[idx].LocalAddress)
		}
		if backendStats[idx].LocalPort != expect.LocalPort {
			t.Errorf("expected LocalPort %d, got %d", expect.LocalPort, backendStats[idx].LocalPort)
		}
		if !backendStats[idx].RemoteAddress.Equal(expect.RemoteAddress) {
			t.Errorf("expected RemoteAddress %s, got %s", expect.RemoteAddress, backendStats[idx].RemoteAddress)
		}
		if backendStats[idx].RemotePort != expect.RemotePort {
			t.Errorf("expected RemotePort %d, got %d", expect.RemotePort, backendStats[idx].RemotePort)
		}
		if backendStats[idx].Proto != expect.Proto {
			t.Errorf("expected Proto %s, got %s", expect.Proto, backendStats[idx].Proto)
		}
		if backendStats[idx].Weight != expect.Weight {
			t.Errorf("expected Weight %d, got %d", expect.Weight, backendStats[idx].Weight)
		}
		if backendStats[idx].ActiveConn != expect.ActiveConn {
			t.Errorf("expected ActiveConn %d, got %d", expect.ActiveConn, backendStats[idx].ActiveConn)
		}
		if backendStats[idx].InactConn != expect.InactConn {
			t.Errorf("expected InactConn %d, got %d", expect.InactConn, backendStats[idx].InactConn)
		}
	}
}
