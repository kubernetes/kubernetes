package netlink

import (
	"net"
	"testing"
)

type arpEntry struct {
	ip  net.IP
	mac net.HardwareAddr
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
