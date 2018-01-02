// +build linux

package netlink

import (
	"net"
	"os"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestAddrAdd(t *testing.T) {
	DoTestAddr(t, AddrAdd)
}

func TestAddrReplace(t *testing.T) {
	DoTestAddr(t, AddrReplace)
}

func DoTestAddr(t *testing.T, FunctionUndertest func(Link, *Addr) error) {
	if os.Getenv("TRAVIS_BUILD_DIR") != "" {
		t.Skipf("Fails in travis with: addr_test.go:68: Address flags not set properly, got=0, expected=128")
	}
	// TODO: IFA_F_PERMANENT does not seem to be set by default on older kernels?
	var address = &net.IPNet{IP: net.IPv4(127, 0, 0, 2), Mask: net.CIDRMask(24, 32)}
	var peer = &net.IPNet{IP: net.IPv4(127, 0, 0, 3), Mask: net.CIDRMask(24, 32)}
	var addrTests = []struct {
		addr     *Addr
		expected *Addr
	}{
		{
			&Addr{IPNet: address},
			&Addr{IPNet: address, Label: "lo", Scope: unix.RT_SCOPE_UNIVERSE, Flags: unix.IFA_F_PERMANENT},
		},
		{
			&Addr{IPNet: address, Label: "local"},
			&Addr{IPNet: address, Label: "local", Scope: unix.RT_SCOPE_UNIVERSE, Flags: unix.IFA_F_PERMANENT},
		},
		{
			&Addr{IPNet: address, Flags: unix.IFA_F_OPTIMISTIC},
			&Addr{IPNet: address, Label: "lo", Flags: unix.IFA_F_OPTIMISTIC | unix.IFA_F_PERMANENT, Scope: unix.RT_SCOPE_UNIVERSE},
		},
		{
			&Addr{IPNet: address, Flags: unix.IFA_F_OPTIMISTIC | unix.IFA_F_DADFAILED},
			&Addr{IPNet: address, Label: "lo", Flags: unix.IFA_F_OPTIMISTIC | unix.IFA_F_DADFAILED | unix.IFA_F_PERMANENT, Scope: unix.RT_SCOPE_UNIVERSE},
		},
		{
			&Addr{IPNet: address, Scope: unix.RT_SCOPE_NOWHERE},
			&Addr{IPNet: address, Label: "lo", Flags: unix.IFA_F_PERMANENT, Scope: unix.RT_SCOPE_NOWHERE},
		},
		{
			&Addr{IPNet: address, Peer: peer},
			&Addr{IPNet: address, Peer: peer, Label: "lo", Scope: unix.RT_SCOPE_UNIVERSE, Flags: unix.IFA_F_PERMANENT},
		},
	}

	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range addrTests {
		if err = FunctionUndertest(link, tt.addr); err != nil {
			t.Fatal(err)
		}

		addrs, err := AddrList(link, FAMILY_ALL)
		if err != nil {
			t.Fatal(err)
		}

		if len(addrs) != 1 {
			t.Fatal("Address not added properly")
		}

		if !addrs[0].Equal(*tt.expected) {
			t.Fatalf("Address ip no set properly, got=%s, expected=%s", addrs[0], tt.expected)
		}

		if addrs[0].Label != tt.expected.Label {
			t.Fatalf("Address label not set properly, got=%s, expected=%s", addrs[0].Label, tt.expected.Label)
		}

		if addrs[0].Flags != tt.expected.Flags {
			t.Fatalf("Address flags not set properly, got=%d, expected=%d", addrs[0].Flags, tt.expected.Flags)
		}

		if addrs[0].Scope != tt.expected.Scope {
			t.Fatalf("Address scope not set properly, got=%d, expected=%d", addrs[0].Scope, tt.expected.Scope)
		}

		if tt.expected.Peer != nil {
			if !addrs[0].PeerEqual(*tt.expected) {
				t.Fatalf("Peer Address ip no set properly, got=%s, expected=%s", addrs[0].Peer, tt.expected.Peer)
			}
		}

		// Pass FAMILY_V4, we should get the same results as FAMILY_ALL
		addrs, err = AddrList(link, FAMILY_V4)
		if err != nil {
			t.Fatal(err)
		}
		if len(addrs) != 1 {
			t.Fatal("Address not added properly")
		}

		// Pass a wrong family number, we should get nil list
		addrs, err = AddrList(link, 0x8)
		if err != nil {
			t.Fatal(err)
		}

		if len(addrs) != 0 {
			t.Fatal("Address not expected")
		}

		if err = AddrDel(link, tt.addr); err != nil {
			t.Fatal(err)
		}

		addrs, err = AddrList(link, FAMILY_ALL)
		if err != nil {
			t.Fatal(err)
		}

		if len(addrs) != 0 {
			t.Fatal("Address not removed properly")
		}
	}

}

func TestAddrAddReplace(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	var address = &net.IPNet{IP: net.IPv4(127, 0, 0, 2), Mask: net.CIDRMask(24, 32)}
	var addr = &Addr{IPNet: address}

	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	err = AddrAdd(link, addr)
	if err != nil {
		t.Fatal(err)
	}

	addrs, err := AddrList(link, FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}

	if len(addrs) != 1 {
		t.Fatal("Address not added properly")
	}

	err = AddrAdd(link, addr)
	if err == nil {
		t.Fatal("Re-adding address should fail (but succeeded unexpectedly).")
	}

	err = AddrReplace(link, addr)
	if err != nil {
		t.Fatal("Replacing address failed.")
	}

	addrs, err = AddrList(link, FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}

	if len(addrs) != 1 {
		t.Fatal("Address not added properly")
	}

	if err = AddrDel(link, addr); err != nil {
		t.Fatal(err)
	}

	addrs, err = AddrList(link, FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}

	if len(addrs) != 0 {
		t.Fatal("Address not removed properly")
	}
}

func expectAddrUpdate(ch <-chan AddrUpdate, add bool, dst net.IP) bool {
	for {
		timeout := time.After(time.Minute)
		select {
		case update := <-ch:
			if update.NewAddr == add && update.LinkAddress.IP.Equal(dst) {
				return true
			}
		case <-timeout:
			return false
		}
	}
}

func TestAddrSubscribeWithOptions(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	ch := make(chan AddrUpdate)
	done := make(chan struct{})
	defer close(done)
	var lastError error
	defer func() {
		if lastError != nil {
			t.Fatalf("Fatal error received during subscription: %v", lastError)
		}
	}()
	if err := AddrSubscribeWithOptions(ch, done, AddrSubscribeOptions{
		ErrorCallback: func(err error) {
			lastError = err
		},
	}); err != nil {
		t.Fatal(err)
	}

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	ip := net.IPv4(127, 0, 0, 1)
	if !expectAddrUpdate(ch, true, ip) {
		t.Fatal("Add update not received as expected")
	}
}
