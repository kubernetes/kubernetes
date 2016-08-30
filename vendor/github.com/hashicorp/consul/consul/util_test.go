package consul

import (
	"errors"
	"fmt"
	"net"
	"regexp"
	"testing"

	"github.com/hashicorp/serf/serf"
)

func TestGetPrivateIP(t *testing.T) {
	ip, _, err := net.ParseCIDR("10.1.2.3/32")
	if err != nil {
		t.Fatalf("failed to parse private cidr: %v", err)
	}

	pubIP, _, err := net.ParseCIDR("8.8.8.8/32")
	if err != nil {
		t.Fatalf("failed to parse public cidr: %v", err)
	}

	tests := []struct {
		addrs    []net.Addr
		expected net.IP
		err      error
	}{
		{
			addrs: []net.Addr{
				&net.IPAddr{
					IP: ip,
				},
				&net.IPAddr{
					IP: pubIP,
				},
			},
			expected: ip,
		},
		{
			addrs: []net.Addr{
				&net.IPAddr{
					IP: pubIP,
				},
			},
			err: errors.New("No private IP address found"),
		},
		{
			addrs: []net.Addr{
				&net.IPAddr{
					IP: ip,
				},
				&net.IPAddr{
					IP: ip,
				},
				&net.IPAddr{
					IP: pubIP,
				},
			},
			err: errors.New("Multiple private IPs found. Please configure one."),
		},
	}

	for _, test := range tests {
		ip, err := getPrivateIP(test.addrs)
		switch {
		case test.err != nil && err != nil:
			if err.Error() != test.err.Error() {
				t.Fatalf("unexpected error: %v != %v", test.err, err)
			}
		case (test.err == nil && err != nil) || (test.err != nil && err == nil):
			t.Fatalf("unexpected error: %v != %v", test.err, err)
		default:
			if !test.expected.Equal(ip) {
				t.Fatalf("unexpected ip: %v != %v", ip, test.expected)
			}
		}
	}
}

func TestIsPrivateIP(t *testing.T) {
	if !isPrivateIP("192.168.1.1") {
		t.Fatalf("bad")
	}
	if !isPrivateIP("172.16.45.100") {
		t.Fatalf("bad")
	}
	if !isPrivateIP("10.1.2.3") {
		t.Fatalf("bad")
	}
	if !isPrivateIP("100.115.110.19") {
		t.Fatalf("bad")
	}
	if isPrivateIP("8.8.8.8") {
		t.Fatalf("bad")
	}
	if !isPrivateIP("127.0.0.1") {
		t.Fatalf("bad")
	}
}

func TestUtil_CanServersUnderstandProtocol(t *testing.T) {
	var members []serf.Member

	// All empty list cases should return false.
	for v := ProtocolVersionMin; v <= ProtocolVersionMax; v++ {
		grok, err := CanServersUnderstandProtocol(members, v)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if grok {
			t.Fatalf("empty list should always return false")
		}
	}

	// Add a non-server member.
	members = append(members, serf.Member{
		Tags: map[string]string{
			"vsn_min": fmt.Sprintf("%d", ProtocolVersionMin),
			"vsn_max": fmt.Sprintf("%d", ProtocolVersionMax),
		},
	})

	// Make sure it doesn't get counted.
	for v := ProtocolVersionMin; v <= ProtocolVersionMax; v++ {
		grok, err := CanServersUnderstandProtocol(members, v)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if grok {
			t.Fatalf("non-server members should not be counted")
		}
	}

	// Add a server member.
	members = append(members, serf.Member{
		Tags: map[string]string{
			"role":    "consul",
			"vsn_min": fmt.Sprintf("%d", ProtocolVersionMin),
			"vsn_max": fmt.Sprintf("%d", ProtocolVersionMax),
		},
	})

	// Now it should report that it understands.
	for v := ProtocolVersionMin; v <= ProtocolVersionMax; v++ {
		grok, err := CanServersUnderstandProtocol(members, v)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if !grok {
			t.Fatalf("server should grok")
		}
	}

	// Nobody should understand anything from the future.
	for v := uint8(ProtocolVersionMax + 1); v <= uint8(ProtocolVersionMax+10); v++ {
		grok, err := CanServersUnderstandProtocol(members, v)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if grok {
			t.Fatalf("server should not grok")
		}
	}

	// Add an older server.
	members = append(members, serf.Member{
		Tags: map[string]string{
			"role":    "consul",
			"vsn_min": fmt.Sprintf("%d", ProtocolVersionMin),
			"vsn_max": fmt.Sprintf("%d", ProtocolVersionMax-1),
		},
	})

	// The servers should no longer understand the max version.
	for v := ProtocolVersionMin; v <= ProtocolVersionMax; v++ {
		grok, err := CanServersUnderstandProtocol(members, v)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		expected := v < ProtocolVersionMax
		if grok != expected {
			t.Fatalf("bad: %v != %v", grok, expected)
		}
	}

	// Try a version that's too low for the minimum.
	{
		grok, err := CanServersUnderstandProtocol(members, 0)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if grok {
			t.Fatalf("server should not grok")
		}
	}
}

func TestIsConsulServer(t *testing.T) {
	m := serf.Member{
		Name: "foo",
		Addr: net.IP([]byte{127, 0, 0, 1}),
		Tags: map[string]string{
			"role": "consul",
			"dc":   "east-aws",
			"port": "10000",
			"vsn":  "1",
		},
	}
	valid, parts := isConsulServer(m)
	if !valid || parts.Datacenter != "east-aws" || parts.Port != 10000 {
		t.Fatalf("bad: %v %v", valid, parts)
	}
	if parts.Name != "foo" {
		t.Fatalf("bad: %v", parts)
	}
	if parts.Bootstrap {
		t.Fatalf("unexpected bootstrap")
	}
	if parts.Expect != 0 {
		t.Fatalf("bad: %v", parts.Expect)
	}
	m.Tags["bootstrap"] = "1"
	valid, parts = isConsulServer(m)
	if !valid || !parts.Bootstrap {
		t.Fatalf("expected bootstrap")
	}
	if parts.Addr.String() != "127.0.0.1:10000" {
		t.Fatalf("bad addr: %v", parts.Addr)
	}
	if parts.Version != 1 {
		t.Fatalf("bad: %v", parts)
	}
	m.Tags["expect"] = "3"
	delete(m.Tags, "bootstrap")
	valid, parts = isConsulServer(m)
	if !valid || parts.Expect != 3 {
		t.Fatalf("bad: %v", parts.Expect)
	}
}

func TestIsConsulNode(t *testing.T) {
	m := serf.Member{
		Tags: map[string]string{
			"role": "node",
			"dc":   "east-aws",
		},
	}
	valid, dc := isConsulNode(m)
	if !valid || dc != "east-aws" {
		t.Fatalf("bad: %v %v", valid, dc)
	}
}

func TestByteConversion(t *testing.T) {
	var val uint64 = 2 << 50
	raw := uint64ToBytes(val)
	if bytesToUint64(raw) != val {
		t.Fatalf("no match")
	}
}

func TestGenerateUUID(t *testing.T) {
	prev := generateUUID()
	for i := 0; i < 100; i++ {
		id := generateUUID()
		if prev == id {
			t.Fatalf("Should get a new ID!")
		}

		matched, err := regexp.MatchString(
			"[\\da-f]{8}-[\\da-f]{4}-[\\da-f]{4}-[\\da-f]{4}-[\\da-f]{12}", id)
		if !matched || err != nil {
			t.Fatalf("expected match %s %v %s", id, matched, err)
		}
	}
}
