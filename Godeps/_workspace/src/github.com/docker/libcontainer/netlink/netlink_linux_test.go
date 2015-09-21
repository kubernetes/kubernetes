package netlink

import (
	"net"
	"strings"
	"syscall"
	"testing"
)

type testLink struct {
	name     string
	linkType string
}

func addLink(t *testing.T, name string, linkType string) {
	if err := NetworkLinkAdd(name, linkType); err != nil {
		t.Fatalf("Unable to create %s link: %s", name, err)
	}
}

func readLink(t *testing.T, name string) *net.Interface {
	iface, err := net.InterfaceByName(name)
	if err != nil {
		t.Fatalf("Could not find %s interface: %s", name, err)
	}

	return iface
}

func deleteLink(t *testing.T, name string) {
	if err := NetworkLinkDel(name); err != nil {
		t.Fatalf("Unable to delete %s link: %s", name, err)
	}
}

func upLink(t *testing.T, name string) {
	iface := readLink(t, name)
	if err := NetworkLinkUp(iface); err != nil {
		t.Fatalf("Could not bring UP %#v interface: %s", iface, err)
	}
}

func downLink(t *testing.T, name string) {
	iface := readLink(t, name)
	if err := NetworkLinkDown(iface); err != nil {
		t.Fatalf("Could not bring DOWN %#v interface: %s", iface, err)
	}
}

func ipAssigned(iface *net.Interface, ip net.IP) bool {
	addrs, _ := iface.Addrs()

	for _, addr := range addrs {
		args := strings.SplitN(addr.String(), "/", 2)
		if args[0] == ip.String() {
			return true
		}
	}

	return false
}

func TestNetworkLinkAddDel(t *testing.T) {
	if testing.Short() {
		return
	}

	testLinks := []testLink{
		{"tstEth", "dummy"},
		{"tstBr", "bridge"},
	}

	for _, tl := range testLinks {
		addLink(t, tl.name, tl.linkType)
		defer deleteLink(t, tl.name)
		readLink(t, tl.name)
	}
}

func TestNetworkLinkUpDown(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := testLink{name: "tstEth", linkType: "dummy"}

	addLink(t, tl.name, tl.linkType)
	defer deleteLink(t, tl.name)

	upLink(t, tl.name)
	ifcAfterUp := readLink(t, tl.name)

	if (ifcAfterUp.Flags & syscall.IFF_UP) != syscall.IFF_UP {
		t.Fatalf("Could not bring UP %#v initerface", tl)
	}

	downLink(t, tl.name)
	ifcAfterDown := readLink(t, tl.name)

	if (ifcAfterDown.Flags & syscall.IFF_UP) == syscall.IFF_UP {
		t.Fatalf("Could not bring DOWN %#v initerface", tl)
	}
}

func TestNetworkSetMacAddress(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := testLink{name: "tstEth", linkType: "dummy"}
	macaddr := "22:ce:e0:99:63:6f"

	addLink(t, tl.name, tl.linkType)
	defer deleteLink(t, tl.name)

	ifcBeforeSet := readLink(t, tl.name)

	if err := NetworkSetMacAddress(ifcBeforeSet, macaddr); err != nil {
		t.Fatalf("Could not set %s MAC address on %#v interface: %s", macaddr, tl, err)
	}

	ifcAfterSet := readLink(t, tl.name)

	if ifcAfterSet.HardwareAddr.String() != macaddr {
		t.Fatalf("Could not set %s MAC address on %#v interface", macaddr, tl)
	}
}

func TestNetworkSetMTU(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := testLink{name: "tstEth", linkType: "dummy"}
	mtu := 1400

	addLink(t, tl.name, tl.linkType)
	defer deleteLink(t, tl.name)

	ifcBeforeSet := readLink(t, tl.name)

	if err := NetworkSetMTU(ifcBeforeSet, mtu); err != nil {
		t.Fatalf("Could not set %d MTU on %#v interface: %s", mtu, tl, err)
	}

	ifcAfterSet := readLink(t, tl.name)

	if ifcAfterSet.MTU != mtu {
		t.Fatalf("Could not set %d MTU on %#v interface", mtu, tl)
	}
}

func TestNetworkSetMasterNoMaster(t *testing.T) {
	if testing.Short() {
		return
	}

	master := testLink{"tstBr", "bridge"}
	slave := testLink{"tstEth", "dummy"}
	testLinks := []testLink{master, slave}

	for _, tl := range testLinks {
		addLink(t, tl.name, tl.linkType)
		defer deleteLink(t, tl.name)
		upLink(t, tl.name)
	}

	masterIfc := readLink(t, master.name)
	slaveIfc := readLink(t, slave.name)
	if err := NetworkSetMaster(slaveIfc, masterIfc); err != nil {
		t.Fatalf("Could not set %#v to be the master of %#v: %s", master, slave, err)
	}

	// Trying to figure out a way to test which will not break on RHEL6.
	// We could check for existence of /sys/class/net/tstEth/upper_tstBr
	// which should point to the ../tstBr which is the UPPER device i.e. network bridge

	if err := NetworkSetNoMaster(slaveIfc); err != nil {
		t.Fatalf("Could not UNset %#v master of %#v: %s", master, slave, err)
	}
}

func TestNetworkChangeName(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := testLink{"tstEth", "dummy"}
	newName := "newTst"

	addLink(t, tl.name, tl.linkType)

	linkIfc := readLink(t, tl.name)
	if err := NetworkChangeName(linkIfc, newName); err != nil {
		deleteLink(t, tl.name)
		t.Fatalf("Could not change %#v interface name to %s: %s", tl, newName, err)
	}

	readLink(t, newName)
	deleteLink(t, newName)
}

func TestNetworkLinkAddVlan(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := struct {
		name string
		id   uint16
	}{
		name: "tstVlan",
		id:   32,
	}
	masterLink := testLink{"tstEth", "dummy"}

	addLink(t, masterLink.name, masterLink.linkType)
	defer deleteLink(t, masterLink.name)

	if err := NetworkLinkAddVlan(masterLink.name, tl.name, tl.id); err != nil {
		t.Fatalf("Unable to create %#v VLAN interface: %s", tl, err)
	}

	readLink(t, tl.name)
}

func TestNetworkLinkAddMacVlan(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := struct {
		name string
		mode string
	}{
		name: "tstVlan",
		mode: "private",
	}
	masterLink := testLink{"tstEth", "dummy"}

	addLink(t, masterLink.name, masterLink.linkType)
	defer deleteLink(t, masterLink.name)

	if err := NetworkLinkAddMacVlan(masterLink.name, tl.name, tl.mode); err != nil {
		t.Fatalf("Unable to create %#v MAC VLAN interface: %s", tl, err)
	}

	readLink(t, tl.name)
}

func TestNetworkLinkAddMacVtap(t *testing.T) {
	if testing.Short() {
		return
	}

	tl := struct {
		name string
		mode string
	}{
		name: "tstVtap",
		mode: "private",
	}
	masterLink := testLink{"tstEth", "dummy"}

	addLink(t, masterLink.name, masterLink.linkType)
	defer deleteLink(t, masterLink.name)

	if err := NetworkLinkAddMacVtap(masterLink.name, tl.name, tl.mode); err != nil {
		t.Fatalf("Unable to create %#v MAC VTAP interface: %s", tl, err)
	}

	readLink(t, tl.name)
}

func TestAddDelNetworkIp(t *testing.T) {
	if testing.Short() {
		return
	}

	ifaceName := "lo"
	ip := net.ParseIP("127.0.1.1")
	mask := net.IPv4Mask(255, 255, 255, 255)
	ipNet := &net.IPNet{IP: ip, Mask: mask}

	iface, err := net.InterfaceByName(ifaceName)
	if err != nil {
		t.Skip("No 'lo' interface; skipping tests")
	}

	if err := NetworkLinkAddIp(iface, ip, ipNet); err != nil {
		t.Fatalf("Could not add IP address %s to interface %#v: %s", ip.String(), iface, err)
	}

	if !ipAssigned(iface, ip) {
		t.Fatalf("Could not locate address '%s' in lo address list.", ip.String())
	}

	if err := NetworkLinkDelIp(iface, ip, ipNet); err != nil {
		t.Fatalf("Could not delete IP address %s from interface %#v: %s", ip.String(), iface, err)
	}

	if ipAssigned(iface, ip) {
		t.Fatalf("Located address '%s' in lo address list after removal.", ip.String())
	}
}

func TestAddRouteSourceSelection(t *testing.T) {
	tstIp := "127.1.1.1"
	tl := testLink{name: "tstEth", linkType: "dummy"}

	addLink(t, tl.name, tl.linkType)
	defer deleteLink(t, tl.name)

	ip := net.ParseIP(tstIp)
	mask := net.IPv4Mask(255, 255, 255, 255)
	ipNet := &net.IPNet{IP: ip, Mask: mask}

	iface, err := net.InterfaceByName(tl.name)
	if err != nil {
		t.Fatalf("Lost created link %#v", tl)
	}

	if err := NetworkLinkAddIp(iface, ip, ipNet); err != nil {
		t.Fatalf("Could not add IP address %s to interface %#v: %s", ip.String(), iface, err)
	}

	upLink(t, tl.name)
	defer downLink(t, tl.name)

	if err := AddRoute("127.0.0.0/8", tstIp, "", tl.name); err != nil {
		t.Fatalf("Failed to add route with source address")
	}
}

func TestCreateVethPair(t *testing.T) {
	if testing.Short() {
		return
	}

	var (
		name1 = "veth1"
		name2 = "veth2"
	)

	if err := NetworkCreateVethPair(name1, name2, 0); err != nil {
		t.Fatalf("Could not create veth pair %s %s: %s", name1, name2, err)
	}
	defer NetworkLinkDel(name1)

	readLink(t, name1)
	readLink(t, name2)
}

//
// netlink package tests which do not use RTNETLINK
//
func TestCreateBridgeWithMac(t *testing.T) {
	if testing.Short() {
		return
	}

	name := "testbridge"

	if err := CreateBridge(name, true); err != nil {
		t.Fatal(err)
	}

	if _, err := net.InterfaceByName(name); err != nil {
		t.Fatal(err)
	}

	// cleanup and tests

	if err := DeleteBridge(name); err != nil {
		t.Fatal(err)
	}

	if _, err := net.InterfaceByName(name); err == nil {
		t.Fatalf("expected error getting interface because %s bridge was deleted", name)
	}
}

func TestSetMacAddress(t *testing.T) {
	if testing.Short() {
		return
	}

	name := "testmac"
	mac := randMacAddr()

	if err := NetworkLinkAdd(name, "bridge"); err != nil {
		t.Fatal(err)
	}
	defer NetworkLinkDel(name)

	if err := SetMacAddress(name, mac); err != nil {
		t.Fatal(err)
	}

	iface, err := net.InterfaceByName(name)
	if err != nil {
		t.Fatal(err)
	}

	if iface.HardwareAddr.String() != mac {
		t.Fatalf("mac address %q does not match %q", iface.HardwareAddr, mac)
	}
}
