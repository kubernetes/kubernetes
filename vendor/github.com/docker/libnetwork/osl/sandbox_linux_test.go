package osl

import (
	"crypto/rand"
	"encoding/hex"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
)

const (
	vethName1     = "wierdlongname1"
	vethName2     = "wierdlongname2"
	vethName3     = "wierdlongname3"
	vethName4     = "wierdlongname4"
	sboxIfaceName = "containername"
)

func generateRandomName(prefix string, size int) (string, error) {
	id := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, id); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(id)[:size], nil
}

func newKey(t *testing.T) (string, error) {
	name, err := generateRandomName("netns", 12)
	if err != nil {
		return "", err
	}

	name = filepath.Join("/tmp", name)
	if _, err := os.Create(name); err != nil {
		return "", err
	}

	// Set the rpmCleanupPeriod to be low to make the test run quicker
	gpmLock.Lock()
	gpmCleanupPeriod = 2 * time.Second
	gpmLock.Unlock()

	return name, nil
}

func newInfo(hnd *netlink.Handle, t *testing.T) (Sandbox, error) {
	veth := &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: vethName1, TxQLen: 0},
		PeerName:  vethName2}
	if err := hnd.LinkAdd(veth); err != nil {
		return nil, err
	}

	// Store the sandbox side pipe interface
	// This is needed for cleanup on DeleteEndpoint()
	intf1 := &nwIface{}
	intf1.srcName = vethName2
	intf1.dstName = sboxIfaceName

	ip4, addr, err := net.ParseCIDR("192.168.1.100/24")
	if err != nil {
		return nil, err
	}
	intf1.address = addr
	intf1.address.IP = ip4

	ip6, addrv6, err := net.ParseCIDR("2001:DB8::ABCD/48")
	if err != nil {
		return nil, err
	}
	intf1.addressIPv6 = addrv6
	intf1.addressIPv6.IP = ip6

	_, route, err := net.ParseCIDR("192.168.2.1/32")
	if err != nil {
		return nil, err
	}

	intf1.routes = []*net.IPNet{route}

	intf2 := &nwIface{}
	intf2.srcName = "testbridge"
	intf2.dstName = sboxIfaceName
	intf2.bridge = true

	veth = &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: vethName3, TxQLen: 0},
		PeerName:  vethName4}

	if err := hnd.LinkAdd(veth); err != nil {
		return nil, err
	}

	intf3 := &nwIface{}
	intf3.srcName = vethName4
	intf3.dstName = sboxIfaceName
	intf3.master = "testbridge"

	info := &networkNamespace{iFaces: []*nwIface{intf1, intf2, intf3}}

	info.gw = net.ParseIP("192.168.1.1")
	// sinfo.GatewayIPv6 = net.ParseIP("2001:DB8::1")
	info.gwv6 = net.ParseIP("fe80::1")

	return info, nil
}

func verifySandbox(t *testing.T, s Sandbox, ifaceSuffixes []string) {
	_, ok := s.(*networkNamespace)
	if !ok {
		t.Fatalf("The sandox interface returned is not of type networkNamespace")
	}

	sbNs, err := netns.GetFromPath(s.Key())
	if err != nil {
		t.Fatalf("Failed top open network namespace path %q: %v", s.Key(), err)
	}
	defer sbNs.Close()

	nh, err := netlink.NewHandleAt(sbNs)
	if err != nil {
		t.Fatal(err)
	}
	defer nh.Delete()

	for _, suffix := range ifaceSuffixes {
		_, err = nh.LinkByName(sboxIfaceName + suffix)
		if err != nil {
			t.Fatalf("Could not find the interface %s inside the sandbox: %v",
				sboxIfaceName+suffix, err)
		}
	}
}

func verifyCleanup(t *testing.T, s Sandbox, wait bool) {
	if wait {
		time.Sleep(time.Duration(gpmCleanupPeriod * 2))
	}

	if _, err := os.Stat(s.Key()); err == nil {
		if wait {
			t.Fatalf("The sandbox path %s is not getting cleaned up even after twice the cleanup period", s.Key())
		} else {
			t.Fatalf("The sandbox path %s is not cleaned up after running gc", s.Key())
		}
	}
}

func TestScanStatistics(t *testing.T) {
	data :=
		"Inter-|   Receive                                                |  Transmit\n" +
			"	face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n" +
			"  eth0:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0\n" +
			" wlan0: 7787685   11141    0    0    0     0          0         0  1681390    7220    0    0    0     0       0          0\n" +
			"    lo:  783782    1853    0    0    0     0          0         0   783782    1853    0    0    0     0       0          0\n" +
			"lxcbr0:       0       0    0    0    0     0          0         0     9006      61    0    0    0     0       0          0\n"

	i := &types.InterfaceStatistics{}

	if err := scanInterfaceStats(data, "wlan0", i); err != nil {
		t.Fatal(err)
	}
	if i.TxBytes != 1681390 || i.TxPackets != 7220 || i.RxBytes != 7787685 || i.RxPackets != 11141 {
		t.Fatalf("Error scanning the statistics")
	}

	if err := scanInterfaceStats(data, "lxcbr0", i); err != nil {
		t.Fatal(err)
	}
	if i.TxBytes != 9006 || i.TxPackets != 61 || i.RxBytes != 0 || i.RxPackets != 0 {
		t.Fatalf("Error scanning the statistics")
	}
}

func TestDisableIPv6DAD(t *testing.T) {
	if testutils.RunningOnCircleCI() {
		t.Skipf("Skipping as not supported on CIRCLE CI kernel")
	}

	defer testutils.SetupTestOSContext(t)()

	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()
	defer s.Destroy()

	n, ok := s.(*networkNamespace)
	if !ok {
		t.Fatal(ok)
	}
	nlh := n.nlHandle

	ipv6, _ := types.ParseCIDR("2001:db8::44/64")
	iface := &nwIface{addressIPv6: ipv6, ns: n, dstName: "sideA"}

	veth := &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: "sideA"},
		PeerName:  "sideB",
	}

	err = nlh.LinkAdd(veth)
	if err != nil {
		t.Fatal(err)
	}

	link, err := nlh.LinkByName("sideA")
	if err != nil {
		t.Fatal(err)
	}

	err = setInterfaceIPv6(nlh, link, iface)
	if err != nil {
		t.Fatal(err)
	}

	addrList, err := nlh.AddrList(link, nl.FAMILY_V6)
	if err != nil {
		t.Fatal(err)
	}

	if addrList[0].Flags&syscall.IFA_F_NODAD == 0 {
		t.Fatalf("Unexpected interface flags: 0x%x. Expected to contain 0x%x", addrList[0].Flags, syscall.IFA_F_NODAD)
	}
}

func TestSetInterfaceIP(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	key, err := newKey(t)
	if err != nil {
		t.Fatalf("Failed to obtain a key: %v", err)
	}

	s, err := NewSandbox(key, true, false)
	if err != nil {
		t.Fatalf("Failed to create a new sandbox: %v", err)
	}
	runtime.LockOSThread()
	defer s.Destroy()

	n, ok := s.(*networkNamespace)
	if !ok {
		t.Fatal(ok)
	}
	nlh := n.nlHandle

	ipv4, _ := types.ParseCIDR("172.30.0.33/24")
	ipv6, _ := types.ParseCIDR("2001:db8::44/64")
	iface := &nwIface{address: ipv4, addressIPv6: ipv6, ns: n, dstName: "sideA"}

	if err := nlh.LinkAdd(&netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: "sideA"},
		PeerName:  "sideB",
	}); err != nil {
		t.Fatal(err)
	}

	linkA, err := nlh.LinkByName("sideA")
	if err != nil {
		t.Fatal(err)
	}

	linkB, err := nlh.LinkByName("sideB")
	if err != nil {
		t.Fatal(err)
	}

	if err := nlh.LinkSetUp(linkA); err != nil {
		t.Fatal(err)
	}

	if err := nlh.LinkSetUp(linkB); err != nil {
		t.Fatal(err)
	}

	if err := setInterfaceIP(nlh, linkA, iface); err != nil {
		t.Fatal(err)
	}

	if err := setInterfaceIPv6(nlh, linkA, iface); err != nil {
		t.Fatal(err)
	}

	err = setInterfaceIP(nlh, linkB, iface)
	if err == nil {
		t.Fatalf("Expected route conflict error, but succeeded")
	}
	if !strings.Contains(err.Error(), "conflicts with existing route") {
		t.Fatalf("Unexpected error: %v", err)
	}

	err = setInterfaceIPv6(nlh, linkB, iface)
	if err == nil {
		t.Fatalf("Expected route conflict error, but succeeded")
	}
	if !strings.Contains(err.Error(), "conflicts with existing route") {
		t.Fatalf("Unexpected error: %v", err)
	}
}
