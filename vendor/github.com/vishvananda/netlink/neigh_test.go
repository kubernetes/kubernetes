// +build linux

package netlink

import (
	"net"
	"testing"
)

type arpEntry struct {
	ip  net.IP
	mac net.HardwareAddr
}

type proxyEntry struct {
	ip  net.IP
	dev int
}

func parseMAC(s string) net.HardwareAddr {
	m, err := net.ParseMAC(s)
	if err != nil {
		panic(err)
	}
	return m
}

func dumpContains(dump []Neigh, e arpEntry) bool {
	for _, n := range dump {
		if n.IP.Equal(e.ip) && (n.State&NUD_INCOMPLETE) == 0 {
			return true
		}
	}
	return false
}

func dumpContainsNeigh(dump []Neigh, ne Neigh) bool {
	for _, n := range dump {
		if n.IP.Equal(ne.IP) && n.LLIPAddr.Equal(ne.LLIPAddr) {
			return true
		}
	}
	return false
}

func dumpContainsProxy(dump []Neigh, p proxyEntry) bool {
	for _, n := range dump {
		if n.IP.Equal(p.ip) && (n.LinkIndex == p.dev) && (n.Flags&NTF_PROXY) == NTF_PROXY {
			return true
		}
	}
	return false
}

func TestNeighAddDelLLIPAddr(t *testing.T) {
	setUpNetlinkTestWithKModule(t, "ipip")

	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	dummy := Iptun{
		LinkAttrs: LinkAttrs{Name: "neigh0"},
		PMtuDisc:  1,
		Local:     net.IPv4(127, 0, 0, 1),
		Remote:    net.IPv4(127, 0, 0, 1)}
	if err := LinkAdd(&dummy); err != nil {
		t.Errorf("Failed to create link: %v", err)
	}
	ensureIndex(dummy.Attrs())

	entry := Neigh{
		LinkIndex: dummy.Index,
		State:     NUD_PERMANENT,
		IP:        net.IPv4(198, 51, 100, 2),
		LLIPAddr:  net.IPv4(198, 51, 100, 1),
	}

	err := NeighAdd(&entry)
	if err != nil {
		t.Errorf("Failed to NeighAdd: %v", err)
	}

	// Dump and see that all added entries are there
	dump, err := NeighList(dummy.Index, 0)
	if err != nil {
		t.Errorf("Failed to NeighList: %v", err)
	}

	if !dumpContainsNeigh(dump, entry) {
		t.Errorf("Dump does not contain: %v: %v", entry, dump)

	}

	// Delete the entry
	err = NeighDel(&entry)
	if err != nil {
		t.Errorf("Failed to NeighDel: %v", err)
	}

	if err := LinkDel(&dummy); err != nil {
		t.Fatal(err)
	}
}

func TestNeighAddDel(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	dummy := Dummy{LinkAttrs{Name: "neigh0"}}
	if err := LinkAdd(&dummy); err != nil {
		t.Fatal(err)
	}

	ensureIndex(dummy.Attrs())

	arpTable := []arpEntry{
		{net.ParseIP("10.99.0.1"), parseMAC("aa:bb:cc:dd:00:01")},
		{net.ParseIP("10.99.0.2"), parseMAC("aa:bb:cc:dd:00:02")},
		{net.ParseIP("10.99.0.3"), parseMAC("aa:bb:cc:dd:00:03")},
		{net.ParseIP("10.99.0.4"), parseMAC("aa:bb:cc:dd:00:04")},
		{net.ParseIP("10.99.0.5"), parseMAC("aa:bb:cc:dd:00:05")},
	}

	// Add the arpTable
	for _, entry := range arpTable {
		err := NeighAdd(&Neigh{
			LinkIndex:    dummy.Index,
			State:        NUD_REACHABLE,
			IP:           entry.ip,
			HardwareAddr: entry.mac,
		})

		if err != nil {
			t.Errorf("Failed to NeighAdd: %v", err)
		}
	}

	// Dump and see that all added entries are there
	dump, err := NeighList(dummy.Index, 0)
	if err != nil {
		t.Errorf("Failed to NeighList: %v", err)
	}

	for _, entry := range arpTable {
		if !dumpContains(dump, entry) {
			t.Errorf("Dump does not contain: %v", entry)
		}
	}

	// Delete the arpTable
	for _, entry := range arpTable {
		err := NeighDel(&Neigh{
			LinkIndex:    dummy.Index,
			IP:           entry.ip,
			HardwareAddr: entry.mac,
		})

		if err != nil {
			t.Errorf("Failed to NeighDel: %v", err)
		}
	}

	// TODO: seems not working because of cache
	//// Dump and see that none of deleted entries are there
	//dump, err = NeighList(dummy.Index, 0)
	//if err != nil {
	//t.Errorf("Failed to NeighList: %v", err)
	//}

	//for _, entry := range arpTable {
	//if dumpContains(dump, entry) {
	//t.Errorf("Dump contains: %v", entry)
	//}
	//}

	if err := LinkDel(&dummy); err != nil {
		t.Fatal(err)
	}
}

func TestNeighAddDelProxy(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	dummy := Dummy{LinkAttrs{Name: "neigh0"}}
	if err := LinkAdd(&dummy); err != nil {
		t.Fatal(err)
	}

	ensureIndex(dummy.Attrs())

	proxyTable := []proxyEntry{
		{net.ParseIP("10.99.0.1"), dummy.Index},
		{net.ParseIP("10.99.0.2"), dummy.Index},
		{net.ParseIP("10.99.0.3"), dummy.Index},
		{net.ParseIP("10.99.0.4"), dummy.Index},
		{net.ParseIP("10.99.0.5"), dummy.Index},
	}

	// Add the proxyTable
	for _, entry := range proxyTable {
		err := NeighAdd(&Neigh{
			LinkIndex: dummy.Index,
			Flags:     NTF_PROXY,
			IP:        entry.ip,
		})

		if err != nil {
			t.Errorf("Failed to NeighAdd: %v", err)
		}
	}

	// Dump and see that all added entries are there
	dump, err := NeighProxyList(dummy.Index, 0)
	if err != nil {
		t.Errorf("Failed to NeighList: %v", err)
	}

	for _, entry := range proxyTable {
		if !dumpContainsProxy(dump, entry) {
			t.Errorf("Dump does not contain: %v", entry)
		}
	}

	// Delete the proxyTable
	for _, entry := range proxyTable {
		err := NeighDel(&Neigh{
			LinkIndex: dummy.Index,
			Flags:     NTF_PROXY,
			IP:        entry.ip,
		})

		if err != nil {
			t.Errorf("Failed to NeighDel: %v", err)
		}
	}

	// Dump and see that none of deleted entries are there
	dump, err = NeighProxyList(dummy.Index, 0)
	if err != nil {
		t.Errorf("Failed to NeighList: %v", err)
	}

	for _, entry := range proxyTable {
		if dumpContainsProxy(dump, entry) {
			t.Errorf("Dump contains: %v", entry)
		}
	}

	if err := LinkDel(&dummy); err != nil {
		t.Fatal(err)
	}
}
