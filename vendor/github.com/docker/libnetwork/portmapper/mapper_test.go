package portmapper

import (
	"net"
	"strings"
	"testing"

	"github.com/docker/libnetwork/iptables"
	_ "github.com/docker/libnetwork/testutils"
)

func init() {
	// override this func to mock out the proxy server
	newProxy = newMockProxyCommand
}

func TestSetIptablesChain(t *testing.T) {
	pm := New("")

	c := &iptables.ChainInfo{
		Name: "TEST",
	}

	if pm.chain != nil {
		t.Fatal("chain should be nil at init")
	}

	pm.SetIptablesChain(c, "lo")
	if pm.chain == nil {
		t.Fatal("chain should not be nil after set")
	}
}

func TestMapTCPPorts(t *testing.T) {
	pm := New("")
	dstIP1 := net.ParseIP("192.168.0.1")
	dstIP2 := net.ParseIP("192.168.0.2")
	dstAddr1 := &net.TCPAddr{IP: dstIP1, Port: 80}
	dstAddr2 := &net.TCPAddr{IP: dstIP2, Port: 80}

	srcAddr1 := &net.TCPAddr{Port: 1080, IP: net.ParseIP("172.16.0.1")}
	srcAddr2 := &net.TCPAddr{Port: 1080, IP: net.ParseIP("172.16.0.2")}

	addrEqual := func(addr1, addr2 net.Addr) bool {
		return (addr1.Network() == addr2.Network()) && (addr1.String() == addr2.String())
	}

	if host, err := pm.Map(srcAddr1, dstIP1, 80, true); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	} else if !addrEqual(dstAddr1, host) {
		t.Fatalf("Incorrect mapping result: expected %s:%s, got %s:%s",
			dstAddr1.String(), dstAddr1.Network(), host.String(), host.Network())
	}

	if _, err := pm.Map(srcAddr1, dstIP1, 80, true); err == nil {
		t.Fatalf("Port is in use - mapping should have failed")
	}

	if _, err := pm.Map(srcAddr2, dstIP1, 80, true); err == nil {
		t.Fatalf("Port is in use - mapping should have failed")
	}

	if _, err := pm.Map(srcAddr2, dstIP2, 80, true); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	}

	if pm.Unmap(dstAddr1) != nil {
		t.Fatalf("Failed to release port")
	}

	if pm.Unmap(dstAddr2) != nil {
		t.Fatalf("Failed to release port")
	}

	if pm.Unmap(dstAddr2) == nil {
		t.Fatalf("Port already released, but no error reported")
	}
}

func TestGetUDPKey(t *testing.T) {
	addr := &net.UDPAddr{IP: net.ParseIP("192.168.1.5"), Port: 53}

	key := getKey(addr)

	if expected := "192.168.1.5:53/udp"; key != expected {
		t.Fatalf("expected key %s got %s", expected, key)
	}
}

func TestGetTCPKey(t *testing.T) {
	addr := &net.TCPAddr{IP: net.ParseIP("192.168.1.5"), Port: 80}

	key := getKey(addr)

	if expected := "192.168.1.5:80/tcp"; key != expected {
		t.Fatalf("expected key %s got %s", expected, key)
	}
}

func TestGetUDPIPAndPort(t *testing.T) {
	addr := &net.UDPAddr{IP: net.ParseIP("192.168.1.5"), Port: 53}

	ip, port := getIPAndPort(addr)
	if expected := "192.168.1.5"; ip.String() != expected {
		t.Fatalf("expected ip %s got %s", expected, ip)
	}

	if ep := 53; port != ep {
		t.Fatalf("expected port %d got %d", ep, port)
	}
}

func TestMapUDPPorts(t *testing.T) {
	pm := New("")
	dstIP1 := net.ParseIP("192.168.0.1")
	dstIP2 := net.ParseIP("192.168.0.2")
	dstAddr1 := &net.UDPAddr{IP: dstIP1, Port: 80}
	dstAddr2 := &net.UDPAddr{IP: dstIP2, Port: 80}

	srcAddr1 := &net.UDPAddr{Port: 1080, IP: net.ParseIP("172.16.0.1")}
	srcAddr2 := &net.UDPAddr{Port: 1080, IP: net.ParseIP("172.16.0.2")}

	addrEqual := func(addr1, addr2 net.Addr) bool {
		return (addr1.Network() == addr2.Network()) && (addr1.String() == addr2.String())
	}

	if host, err := pm.Map(srcAddr1, dstIP1, 80, true); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	} else if !addrEqual(dstAddr1, host) {
		t.Fatalf("Incorrect mapping result: expected %s:%s, got %s:%s",
			dstAddr1.String(), dstAddr1.Network(), host.String(), host.Network())
	}

	if _, err := pm.Map(srcAddr1, dstIP1, 80, true); err == nil {
		t.Fatalf("Port is in use - mapping should have failed")
	}

	if _, err := pm.Map(srcAddr2, dstIP1, 80, true); err == nil {
		t.Fatalf("Port is in use - mapping should have failed")
	}

	if _, err := pm.Map(srcAddr2, dstIP2, 80, true); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	}

	if pm.Unmap(dstAddr1) != nil {
		t.Fatalf("Failed to release port")
	}

	if pm.Unmap(dstAddr2) != nil {
		t.Fatalf("Failed to release port")
	}

	if pm.Unmap(dstAddr2) == nil {
		t.Fatalf("Port already released, but no error reported")
	}
}

func TestMapAllPortsSingleInterface(t *testing.T) {
	pm := New("")
	dstIP1 := net.ParseIP("0.0.0.0")
	srcAddr1 := &net.TCPAddr{Port: 1080, IP: net.ParseIP("172.16.0.1")}

	hosts := []net.Addr{}
	var host net.Addr
	var err error

	defer func() {
		for _, val := range hosts {
			pm.Unmap(val)
		}
	}()

	for i := 0; i < 10; i++ {
		start, end := pm.Allocator.Begin, pm.Allocator.End
		for i := start; i < end; i++ {
			if host, err = pm.Map(srcAddr1, dstIP1, 0, true); err != nil {
				t.Fatal(err)
			}

			hosts = append(hosts, host)
		}

		if _, err := pm.Map(srcAddr1, dstIP1, start, true); err == nil {
			t.Fatalf("Port %d should be bound but is not", start)
		}

		for _, val := range hosts {
			if err := pm.Unmap(val); err != nil {
				t.Fatal(err)
			}
		}

		hosts = []net.Addr{}
	}
}

func TestMapTCPDummyListen(t *testing.T) {
	pm := New("")
	dstIP := net.ParseIP("0.0.0.0")
	dstAddr := &net.TCPAddr{IP: dstIP, Port: 80}

	// no-op for dummy
	srcAddr := &net.TCPAddr{Port: 1080, IP: net.ParseIP("172.16.0.1")}

	addrEqual := func(addr1, addr2 net.Addr) bool {
		return (addr1.Network() == addr2.Network()) && (addr1.String() == addr2.String())
	}

	if host, err := pm.Map(srcAddr, dstIP, 80, false); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	} else if !addrEqual(dstAddr, host) {
		t.Fatalf("Incorrect mapping result: expected %s:%s, got %s:%s",
			dstAddr.String(), dstAddr.Network(), host.String(), host.Network())
	}
	if _, err := net.Listen("tcp", "0.0.0.0:80"); err == nil {
		t.Fatal("Listen on mapped port without proxy should fail")
	} else {
		if !strings.Contains(err.Error(), "address already in use") {
			t.Fatalf("Error should be about address already in use, got %v", err)
		}
	}
	if _, err := net.Listen("tcp", "0.0.0.0:81"); err != nil {
		t.Fatal(err)
	}
	if host, err := pm.Map(srcAddr, dstIP, 81, false); err == nil {
		t.Fatalf("Bound port shouldn't be allocated, but it was on: %v", host)
	} else {
		if !strings.Contains(err.Error(), "address already in use") {
			t.Fatalf("Error should be about address already in use, got %v", err)
		}
	}
}

func TestMapUDPDummyListen(t *testing.T) {
	pm := New("")
	dstIP := net.ParseIP("0.0.0.0")
	dstAddr := &net.UDPAddr{IP: dstIP, Port: 80}

	// no-op for dummy
	srcAddr := &net.UDPAddr{Port: 1080, IP: net.ParseIP("172.16.0.1")}

	addrEqual := func(addr1, addr2 net.Addr) bool {
		return (addr1.Network() == addr2.Network()) && (addr1.String() == addr2.String())
	}

	if host, err := pm.Map(srcAddr, dstIP, 80, false); err != nil {
		t.Fatalf("Failed to allocate port: %s", err)
	} else if !addrEqual(dstAddr, host) {
		t.Fatalf("Incorrect mapping result: expected %s:%s, got %s:%s",
			dstAddr.String(), dstAddr.Network(), host.String(), host.Network())
	}
	if _, err := net.ListenUDP("udp", &net.UDPAddr{IP: dstIP, Port: 80}); err == nil {
		t.Fatal("Listen on mapped port without proxy should fail")
	} else {
		if !strings.Contains(err.Error(), "address already in use") {
			t.Fatalf("Error should be about address already in use, got %v", err)
		}
	}
	if _, err := net.ListenUDP("udp", &net.UDPAddr{IP: dstIP, Port: 81}); err != nil {
		t.Fatal(err)
	}
	if host, err := pm.Map(srcAddr, dstIP, 81, false); err == nil {
		t.Fatalf("Bound port shouldn't be allocated, but it was on: %v", host)
	} else {
		if !strings.Contains(err.Error(), "address already in use") {
			t.Fatalf("Error should be about address already in use, got %v", err)
		}
	}
}
