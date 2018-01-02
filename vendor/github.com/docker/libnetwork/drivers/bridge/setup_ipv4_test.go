package bridge

import (
	"net"
	"testing"

	"github.com/docker/libnetwork/testutils"
	"github.com/vishvananda/netlink"
)

func setupTestInterface(t *testing.T, nh *netlink.Handle) (*networkConfiguration, *bridgeInterface) {
	config := &networkConfiguration{
		BridgeName: DefaultBridgeName}
	br := &bridgeInterface{nlh: nh}

	if err := setupDevice(config, br); err != nil {
		t.Fatalf("Bridge creation failed: %v", err)
	}
	return config, br
}

func TestSetupBridgeIPv4Fixed(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	ip, netw, err := net.ParseCIDR("192.168.1.1/24")
	if err != nil {
		t.Fatalf("Failed to parse bridge IPv4: %v", err)
	}

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer nh.Delete()

	config, br := setupTestInterface(t, nh)
	config.AddressIPv4 = &net.IPNet{IP: ip, Mask: netw.Mask}
	if err := setupBridgeIPv4(config, br); err != nil {
		t.Fatalf("Failed to setup bridge IPv4: %v", err)
	}

	addrsv4, err := nh.AddrList(br.Link, netlink.FAMILY_V4)
	if err != nil {
		t.Fatalf("Failed to list device IPv4 addresses: %v", err)
	}

	var found bool
	for _, addr := range addrsv4 {
		if config.AddressIPv4.String() == addr.IPNet.String() {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("Bridge device does not have requested IPv4 address %v", config.AddressIPv4)
	}
}

func TestSetupGatewayIPv4(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer nh.Delete()

	ip, nw, _ := net.ParseCIDR("192.168.0.24/16")
	nw.IP = ip
	gw := net.ParseIP("192.168.2.254")

	config := &networkConfiguration{
		BridgeName:         DefaultBridgeName,
		DefaultGatewayIPv4: gw}

	br := &bridgeInterface{bridgeIPv4: nw, nlh: nh}

	if err := setupGatewayIPv4(config, br); err != nil {
		t.Fatalf("Set Default Gateway failed: %v", err)
	}

	if !gw.Equal(br.gatewayIPv4) {
		t.Fatalf("Set Default Gateway failed. Expected %v, Found %v", gw, br.gatewayIPv4)
	}
}
