package macvlan

import (
	"testing"

	"github.com/vishvananda/netlink"
)

// TestValidateLink tests the parentExists function
func TestValidateLink(t *testing.T) {
	validIface := "lo"
	invalidIface := "foo12345"

	// test a valid parent interface validation
	if ok := parentExists(validIface); !ok {
		t.Fatalf("failed validating loopback %s", validIface)
	}
	// test an invalid parent interface validation
	if ok := parentExists(invalidIface); ok {
		t.Fatalf("failed to invalidate interface %s", invalidIface)
	}
}

// TestValidateSubLink tests valid 802.1q naming convention
func TestValidateSubLink(t *testing.T) {
	validSubIface := "lo.10"
	invalidSubIface1 := "lo"
	invalidSubIface2 := "lo:10"
	invalidSubIface3 := "foo123.456"

	// test a valid parent_iface.vlan_id
	_, _, err := parseVlan(validSubIface)
	if err != nil {
		t.Fatalf("failed subinterface validation: %v", err)
	}
	// test an invalid vid with a valid parent link
	_, _, err = parseVlan(invalidSubIface1)
	if err == nil {
		t.Fatalf("failed subinterface validation test: %s", invalidSubIface1)
	}
	// test a valid vid with a valid parent link with an invalid delimiter
	_, _, err = parseVlan(invalidSubIface2)
	if err == nil {
		t.Fatalf("failed subinterface validation test: %v", invalidSubIface2)
	}
	// test an invalid parent link with a valid vid
	_, _, err = parseVlan(invalidSubIface3)
	if err == nil {
		t.Fatalf("failed subinterface validation test: %v", invalidSubIface3)
	}
}

// TestSetMacVlanMode tests the macvlan mode setter
func TestSetMacVlanMode(t *testing.T) {
	// test macvlan bridge mode
	mode, err := setMacVlanMode(modeBridge)
	if err != nil {
		t.Fatalf("error parsing %v vlan mode: %v", mode, err)
	}
	if mode != netlink.MACVLAN_MODE_BRIDGE {
		t.Fatalf("expected %d got %d", netlink.MACVLAN_MODE_BRIDGE, mode)
	}
	// test macvlan passthrough mode
	mode, err = setMacVlanMode(modePassthru)
	if err != nil {
		t.Fatalf("error parsing %v vlan mode: %v", mode, err)
	}
	if mode != netlink.MACVLAN_MODE_PASSTHRU {
		t.Fatalf("expected %d got %d", netlink.MACVLAN_MODE_PASSTHRU, mode)
	}
	// test macvlan private mode
	mode, err = setMacVlanMode(modePrivate)
	if err != nil {
		t.Fatalf("error parsing %v vlan mode: %v", mode, err)
	}
	if mode != netlink.MACVLAN_MODE_PRIVATE {
		t.Fatalf("expected %d got %d", netlink.MACVLAN_MODE_PRIVATE, mode)
	}
	// test macvlan vepa mode
	mode, err = setMacVlanMode(modeVepa)
	if err != nil {
		t.Fatalf("error parsing %v vlan mode: %v", mode, err)
	}
	if mode != netlink.MACVLAN_MODE_VEPA {
		t.Fatalf("expected %d got %d", netlink.MACVLAN_MODE_VEPA, mode)
	}
	// test invalid mode
	mode, err = setMacVlanMode("foo")
	if err == nil {
		t.Fatal("invalid macvlan mode should have returned an error")
	}
	if mode != 0 {
		t.Fatalf("expected 0 got %d", mode)
	}
	// test null mode
	mode, err = setMacVlanMode("")
	if err == nil {
		t.Fatal("invalid macvlan mode should have returned an error")
	}
	if mode != 0 {
		t.Fatalf("expected 0 got %d", mode)
	}
}
