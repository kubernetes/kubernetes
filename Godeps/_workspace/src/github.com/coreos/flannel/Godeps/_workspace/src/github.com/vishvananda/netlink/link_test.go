package netlink

import (
	"bytes"
	"net"
	"testing"

	"github.com/vishvananda/netns"
)

const testTxQLen uint32 = 100

func testLinkAddDel(t *testing.T, link Link) {
	links, err := LinkList()
	if err != nil {
		t.Fatal(err)
	}
	num := len(links)

	if err := LinkAdd(link); err != nil {
		t.Fatal(err)
	}

	base := link.Attrs()

	result, err := LinkByName(base.Name)
	if err != nil {
		t.Fatal(err)
	}

	rBase := result.Attrs()

	if vlan, ok := link.(*Vlan); ok {
		other, ok := result.(*Vlan)
		if !ok {
			t.Fatal("Result of create is not a vlan")
		}
		if vlan.VlanId != other.VlanId {
			t.Fatal("Link.VlanId id doesn't match")
		}
	}

	if rBase.ParentIndex == 0 && base.ParentIndex != 0 {
		t.Fatal("Created link doesn't have a Parent but it should")
	} else if rBase.ParentIndex != 0 && base.ParentIndex == 0 {
		t.Fatal("Created link has a Parent but it shouldn't")
	} else if rBase.ParentIndex != 0 && base.ParentIndex != 0 {
		if rBase.ParentIndex != base.ParentIndex {
			t.Fatal("Link.ParentIndex doesn't match")
		}
	}

	if veth, ok := link.(*Veth); ok {
		if veth.TxQLen != testTxQLen {
			t.Fatalf("TxQLen is %d, should be %d", veth.TxQLen, testTxQLen)
		}
		if rBase.MTU != base.MTU {
			t.Fatalf("MTU is %d, should be %d", rBase.MTU, base.MTU)
		}

		if veth.PeerName != "" {
			var peer *Veth
			other, err := LinkByName(veth.PeerName)
			if err != nil {
				t.Fatalf("Peer %s not created", veth.PeerName)
			}
			if peer, ok = other.(*Veth); !ok {
				t.Fatalf("Peer %s is incorrect type", veth.PeerName)
			}
			if peer.TxQLen != testTxQLen {
				t.Fatalf("TxQLen of peer is %d, should be %d", peer.TxQLen, testTxQLen)
			}
		}
	}

	if vxlan, ok := link.(*Vxlan); ok {
		other, ok := result.(*Vxlan)
		if !ok {
			t.Fatal("Result of create is not a vxlan")
		}
		compareVxlan(t, vxlan, other)
	}

	if ipv, ok := link.(*IPVlan); ok {
		other, ok := result.(*IPVlan)
		if !ok {
			t.Fatal("Result of create is not a ipvlan")
		}
		if ipv.Mode != other.Mode {
			t.Fatalf("Got unexpected mode: %d, expected: %d", other.Mode, ipv.Mode)
		}
	}

	if macv, ok := link.(*Macvlan); ok {
		other, ok := result.(*Macvlan)
		if !ok {
			t.Fatal("Result of create is not a macvlan")
		}
		if macv.Mode != other.Mode {
			t.Fatalf("Got unexpected mode: %d, expected: %d", other.Mode, macv.Mode)
		}
	}

	if err = LinkDel(link); err != nil {
		t.Fatal(err)
	}

	links, err = LinkList()
	if err != nil {
		t.Fatal(err)
	}

	if len(links) != num {
		t.Fatal("Link not removed properly")
	}
}

func compareVxlan(t *testing.T, expected, actual *Vxlan) {

	if actual.VxlanId != expected.VxlanId {
		t.Fatal("Vxlan.VxlanId doesn't match")
	}
	if expected.SrcAddr != nil && !actual.SrcAddr.Equal(expected.SrcAddr) {
		t.Fatal("Vxlan.SrcAddr doesn't match")
	}
	if expected.Group != nil && !actual.Group.Equal(expected.Group) {
		t.Fatal("Vxlan.Group doesn't match")
	}
	if expected.TTL != -1 && actual.TTL != expected.TTL {
		t.Fatal("Vxlan.TTL doesn't match")
	}
	if expected.TOS != -1 && actual.TOS != expected.TOS {
		t.Fatal("Vxlan.TOS doesn't match")
	}
	if actual.Learning != expected.Learning {
		t.Fatal("Vxlan.Learning doesn't match")
	}
	if actual.Proxy != expected.Proxy {
		t.Fatal("Vxlan.Proxy doesn't match")
	}
	if actual.RSC != expected.RSC {
		t.Fatal("Vxlan.RSC doesn't match")
	}
	if actual.L2miss != expected.L2miss {
		t.Fatal("Vxlan.L2miss doesn't match")
	}
	if actual.L3miss != expected.L3miss {
		t.Fatal("Vxlan.L3miss doesn't match")
	}
	if expected.NoAge {
		if !actual.NoAge {
			t.Fatal("Vxlan.NoAge doesn't match")
		}
	} else if expected.Age > 0 && actual.Age != expected.Age {
		t.Fatal("Vxlan.Age doesn't match")
	}
	if expected.Limit > 0 && actual.Limit != expected.Limit {
		t.Fatal("Vxlan.Limit doesn't match")
	}
	if expected.Port > 0 && actual.Port != expected.Port {
		t.Fatal("Vxlan.Port doesn't match")
	}
	if expected.PortLow > 0 || expected.PortHigh > 0 {
		if actual.PortLow != expected.PortLow {
			t.Fatal("Vxlan.PortLow doesn't match")
		}
		if actual.PortHigh != expected.PortHigh {
			t.Fatal("Vxlan.PortHigh doesn't match")
		}
	}
}

func TestLinkAddDelDummy(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	testLinkAddDel(t, &Dummy{LinkAttrs{Name: "foo"}})
}

func TestLinkAddDelBridge(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	testLinkAddDel(t, &Bridge{LinkAttrs{Name: "foo", MTU: 1400}})
}

func TestLinkAddDelVlan(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	parent := &Dummy{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(parent); err != nil {
		t.Fatal(err)
	}

	testLinkAddDel(t, &Vlan{LinkAttrs{Name: "bar", ParentIndex: parent.Attrs().Index}, 900})

	if err := LinkDel(parent); err != nil {
		t.Fatal(err)
	}
}

func TestLinkAddDelMacvlan(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	parent := &Dummy{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(parent); err != nil {
		t.Fatal(err)
	}

	testLinkAddDel(t, &Macvlan{
		LinkAttrs: LinkAttrs{Name: "bar", ParentIndex: parent.Attrs().Index},
		Mode:      MACVLAN_MODE_PRIVATE,
	})

	if err := LinkDel(parent); err != nil {
		t.Fatal(err)
	}
}

func TestLinkAddDelVeth(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	testLinkAddDel(t, &Veth{LinkAttrs{Name: "foo", TxQLen: testTxQLen, MTU: 1400}, "bar"})
}

func TestLinkAddDelBridgeMaster(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	master := &Bridge{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(master); err != nil {
		t.Fatal(err)
	}
	testLinkAddDel(t, &Dummy{LinkAttrs{Name: "bar", MasterIndex: master.Attrs().Index}})

	if err := LinkDel(master); err != nil {
		t.Fatal(err)
	}
}

func TestLinkSetUnsetResetMaster(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	master := &Bridge{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(master); err != nil {
		t.Fatal(err)
	}

	newmaster := &Bridge{LinkAttrs{Name: "bar"}}
	if err := LinkAdd(newmaster); err != nil {
		t.Fatal(err)
	}

	slave := &Dummy{LinkAttrs{Name: "baz"}}
	if err := LinkAdd(slave); err != nil {
		t.Fatal(err)
	}

	if err := LinkSetMaster(slave, master); err != nil {
		t.Fatal(err)
	}

	link, err := LinkByName("baz")
	if err != nil {
		t.Fatal(err)
	}

	if link.Attrs().MasterIndex != master.Attrs().Index {
		t.Fatal("Master not set properly")
	}

	if err := LinkSetMaster(slave, newmaster); err != nil {
		t.Fatal(err)
	}

	link, err = LinkByName("baz")
	if err != nil {
		t.Fatal(err)
	}

	if link.Attrs().MasterIndex != newmaster.Attrs().Index {
		t.Fatal("Master not reset properly")
	}

	if err := LinkSetMaster(slave, nil); err != nil {
		t.Fatal(err)
	}

	link, err = LinkByName("baz")
	if err != nil {
		t.Fatal(err)
	}

	if link.Attrs().MasterIndex != 0 {
		t.Fatal("Master not unset properly")
	}
	if err := LinkDel(slave); err != nil {
		t.Fatal(err)
	}

	if err := LinkDel(newmaster); err != nil {
		t.Fatal(err)
	}

	if err := LinkDel(master); err != nil {
		t.Fatal(err)
	}
}

func TestLinkSetNs(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	basens, err := netns.Get()
	if err != nil {
		t.Fatal("Failed to get basens")
	}
	defer basens.Close()

	newns, err := netns.New()
	if err != nil {
		t.Fatal("Failed to create newns")
	}
	defer newns.Close()

	link := &Veth{LinkAttrs{Name: "foo"}, "bar"}
	if err := LinkAdd(link); err != nil {
		t.Fatal(err)
	}

	peer, err := LinkByName("bar")
	if err != nil {
		t.Fatal(err)
	}

	LinkSetNsFd(peer, int(basens))
	if err != nil {
		t.Fatal("Failed to set newns for link")
	}

	_, err = LinkByName("bar")
	if err == nil {
		t.Fatal("Link bar is still in newns")
	}

	err = netns.Set(basens)
	if err != nil {
		t.Fatal("Failed to set basens")
	}

	peer, err = LinkByName("bar")
	if err != nil {
		t.Fatal("Link is not in basens")
	}

	if err := LinkDel(peer); err != nil {
		t.Fatal(err)
	}

	err = netns.Set(newns)
	if err != nil {
		t.Fatal("Failed to set newns")
	}

	_, err = LinkByName("foo")
	if err == nil {
		t.Fatal("Other half of veth pair not deleted")
	}

}

func TestLinkAddDelVxlan(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	parent := &Dummy{
		LinkAttrs{Name: "foo"},
	}
	if err := LinkAdd(parent); err != nil {
		t.Fatal(err)
	}

	vxlan := Vxlan{
		LinkAttrs: LinkAttrs{
			Name: "bar",
		},
		VxlanId:      10,
		VtepDevIndex: parent.Index,
		Learning:     true,
		L2miss:       true,
		L3miss:       true,
	}

	testLinkAddDel(t, &vxlan)
	if err := LinkDel(parent); err != nil {
		t.Fatal(err)
	}
}

func TestLinkAddDelIPVlanL2(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	parent := &Dummy{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(parent); err != nil {
		t.Fatal(err)
	}

	ipv := IPVlan{
		LinkAttrs: LinkAttrs{
			Name:        "bar",
			ParentIndex: parent.Index,
		},
		Mode: IPVLAN_MODE_L2,
	}

	testLinkAddDel(t, &ipv)
}

func TestLinkAddDelIPVlanL3(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	parent := &Dummy{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(parent); err != nil {
		t.Fatal(err)
	}

	ipv := IPVlan{
		LinkAttrs: LinkAttrs{
			Name:        "bar",
			ParentIndex: parent.Index,
		},
		Mode: IPVLAN_MODE_L3,
	}

	testLinkAddDel(t, &ipv)
}

func TestLinkAddDelIPVlanNoParent(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	ipv := IPVlan{
		LinkAttrs: LinkAttrs{
			Name: "bar",
		},
		Mode: IPVLAN_MODE_L3,
	}
	err := LinkAdd(&ipv)
	if err == nil {
		t.Fatal("Add should fail if ipvlan creating without ParentIndex")
	}
	if err.Error() != "Can't create ipvlan link without ParentIndex" {
		t.Fatalf("Error should be about missing ParentIndex, got %q", err)
	}
}

func TestLinkByIndex(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	dummy := &Dummy{LinkAttrs{Name: "dummy"}}
	if err := LinkAdd(dummy); err != nil {
		t.Fatal(err)
	}

	found, err := LinkByIndex(dummy.Index)
	if err != nil {
		t.Fatal(err)
	}

	if found.Attrs().Index != dummy.Attrs().Index {
		t.Fatalf("Indices don't match: %v != %v", found.Attrs().Index, dummy.Attrs().Index)
	}

	LinkDel(dummy)

	// test not found
	_, err = LinkByIndex(dummy.Attrs().Index)
	if err == nil {
		t.Fatalf("LinkByIndex(%v) found deleted link", err)
	}
}

func TestLinkSet(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	iface := &Dummy{LinkAttrs{Name: "foo"}}
	if err := LinkAdd(iface); err != nil {
		t.Fatal(err)
	}

	link, err := LinkByName("foo")
	if err != nil {
		t.Fatal(err)
	}

	err = LinkSetName(link, "bar")
	if err != nil {
		t.Fatalf("Could not change interface name: %v", err)
	}

	link, err = LinkByName("bar")
	if err != nil {
		t.Fatalf("Interface name not changed: %v", err)
	}

	err = LinkSetMTU(link, 1400)
	if err != nil {
		t.Fatalf("Could not set MTU: %v", err)
	}

	link, err = LinkByName("bar")
	if err != nil {
		t.Fatal(err)
	}

	if link.Attrs().MTU != 1400 {
		t.Fatal("MTU not changed!")
	}

	addr, err := net.ParseMAC("00:12:34:56:78:AB")
	if err != nil {
		t.Fatal(err)
	}

	err = LinkSetHardwareAddr(link, addr)
	if err != nil {
		t.Fatal(err)
	}

	link, err = LinkByName("bar")
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(link.Attrs().HardwareAddr, addr) {
		t.Fatalf("hardware address not changed!")
	}
}
