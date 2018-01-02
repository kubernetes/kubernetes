package hostdiscovery

import (
	"net"
	"testing"

	mapset "github.com/deckarep/golang-set"
	_ "github.com/docker/libnetwork/testutils"

	"github.com/docker/docker/pkg/discovery"
)

func TestDiff(t *testing.T) {
	existing := mapset.NewSetFromSlice([]interface{}{"1.1.1.1", "2.2.2.2"})
	addedIP := "3.3.3.3"
	updated := existing.Clone()
	updated.Add(addedIP)

	added, removed := diff(existing, updated)
	if len(added) != 1 {
		t.Fatalf("Diff failed for an Add update. Expecting 1 element, but got %d elements", len(added))
	}
	if added[0].String() != addedIP {
		t.Fatalf("Expecting : %v, Got : %v", addedIP, added[0])
	}
	if len(removed) > 0 {
		t.Fatalf("Diff failed for remove use-case. Expecting 0 element, but got %d elements", len(removed))
	}

	updated = mapset.NewSetFromSlice([]interface{}{addedIP})
	added, removed = diff(existing, updated)
	if len(removed) != 2 {
		t.Fatalf("Diff failed for a remove update. Expecting 2 element, but got %d elements", len(removed))
	}
	if len(added) != 1 {
		t.Fatalf("Diff failed for add use-case. Expecting 1 element, but got %d elements", len(added))
	}
}

func TestAddedCallback(t *testing.T) {
	hd := hostDiscovery{}
	hd.nodes = mapset.NewSetFromSlice([]interface{}{"1.1.1.1"})
	update := []*discovery.Entry{{Host: "1.1.1.1", Port: "0"}, {Host: "2.2.2.2", Port: "0"}}

	added := false
	removed := false
	hd.processCallback(update, func() {}, func(hosts []net.IP) { added = true }, func(hosts []net.IP) { removed = true })
	if !added {
		t.Fatal("Expecting an Added callback notification. But none received")
	}
}

func TestRemovedCallback(t *testing.T) {
	hd := hostDiscovery{}
	hd.nodes = mapset.NewSetFromSlice([]interface{}{"1.1.1.1", "2.2.2.2"})
	update := []*discovery.Entry{{Host: "1.1.1.1", Port: "0"}}

	added := false
	removed := false
	hd.processCallback(update, func() {}, func(hosts []net.IP) { added = true }, func(hosts []net.IP) { removed = true })
	if !removed {
		t.Fatal("Expecting a Removed callback notification. But none received")
	}
}

func TestNoCallback(t *testing.T) {
	hd := hostDiscovery{}
	hd.nodes = mapset.NewSetFromSlice([]interface{}{"1.1.1.1", "2.2.2.2"})
	update := []*discovery.Entry{{Host: "1.1.1.1", Port: "0"}, {Host: "2.2.2.2", Port: "0"}}

	added := false
	removed := false
	hd.processCallback(update, func() {}, func(hosts []net.IP) { added = true }, func(hosts []net.IP) { removed = true })
	if added || removed {
		t.Fatal("Not expecting any callback notification. But received a callback")
	}
}
