// +build linux

package netlink

import (
	"net"
	"testing"
)

func TestPDPv0AddDel(t *testing.T) {
	tearDown := setUpNetlinkTestWithKModule(t, "gtp")
	defer tearDown()

	if err := LinkAdd(testGTPLink(t)); err != nil {
		t.Fatal(err)
	}
	link, err := LinkByName("gtp0")
	if err != nil {
		t.Fatal(err)
	}
	err = GTPPDPAdd(link, &PDP{
		PeerAddress: net.ParseIP("1.1.1.1"),
		MSAddress:   net.ParseIP("2.2.2.2"),
		TID:         10,
	})
	if err != nil {
		t.Fatal(err)
	}
	list, err := GTPPDPList()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 {
		t.Fatal("Failed to add v0 PDP")
	}
	pdp, err := GTPPDPByMSAddress(link, net.ParseIP("2.2.2.2"))
	if err != nil {
		t.Fatal(err)
	}
	if pdp == nil {
		t.Fatal("failed to get v0 PDP by MS address")
	}
	pdp, err = GTPPDPByTID(link, 10)
	if err != nil {
		t.Fatal(err)
	}
	if pdp == nil {
		t.Fatal("failed to get v0 PDP by TID")
	}
	err = GTPPDPDel(link, &PDP{TID: 10})
	if err != nil {
		t.Fatal(err)
	}
	list, err = GTPPDPList()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 0 {
		t.Fatal("Failed to delete v0 PDP")
	}
}

func TestPDPv1AddDel(t *testing.T) {
	tearDown := setUpNetlinkTestWithKModule(t, "gtp")
	defer tearDown()

	if err := LinkAdd(testGTPLink(t)); err != nil {
		t.Fatal(err)
	}
	link, err := LinkByName("gtp0")
	if err != nil {
		t.Fatal(err)
	}
	err = GTPPDPAdd(link, &PDP{
		PeerAddress: net.ParseIP("1.1.1.1"),
		MSAddress:   net.ParseIP("2.2.2.2"),
		Version:     1,
		ITEI:        10,
		OTEI:        10,
	})
	if err != nil {
		t.Fatal(err)
	}
	list, err := GTPPDPList()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 {
		t.Fatal("Failed to add v1 PDP")
	}
	pdp, err := GTPPDPByMSAddress(link, net.ParseIP("2.2.2.2"))
	if err != nil {
		t.Fatal(err)
	}
	if pdp == nil {
		t.Fatal("failed to get v1 PDP by MS address")
	}
	pdp, err = GTPPDPByITEI(link, 10)
	if err != nil {
		t.Fatal(err)
	}
	if pdp == nil {
		t.Fatal("failed to get v1 PDP by ITEI")
	}
	err = GTPPDPDel(link, &PDP{Version: 1, ITEI: 10})
	if err != nil {
		t.Fatal(err)
	}
	list, err = GTPPDPList()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 0 {
		t.Fatal("Failed to delete v1 PDP")
	}
}
