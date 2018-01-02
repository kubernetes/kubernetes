package bridge

import (
	"testing"

	"github.com/docker/libnetwork/testutils"
	"github.com/vishvananda/netlink"
)

func TestInterfaceDefaultName(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	config := &networkConfiguration{}
	_, err = newInterface(nh, config)
	if err != nil {
		t.Fatalf("newInterface() failed: %v", err)
	}

	if config.BridgeName != DefaultBridgeName {
		t.Fatalf("Expected default interface name %q, got %q", DefaultBridgeName, config.BridgeName)
	}
}

func TestAddressesEmptyInterface(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	inf, err := newInterface(nh, &networkConfiguration{})
	if err != nil {
		t.Fatalf("newInterface() failed: %v", err)
	}

	addrsv4, addrsv6, err := inf.addresses()
	if err != nil {
		t.Fatalf("Failed to get addresses of default interface: %v", err)
	}
	if len(addrsv4) != 0 {
		t.Fatalf("Default interface has unexpected IPv4: %s", addrsv4)
	}
	if len(addrsv6) != 0 {
		t.Fatalf("Default interface has unexpected IPv6: %v", addrsv6)
	}
}
