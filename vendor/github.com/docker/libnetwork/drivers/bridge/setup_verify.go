package bridge

import (
	"fmt"
	"strings"

	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

func setupVerifyAndReconcile(config *networkConfiguration, i *bridgeInterface) error {
	// Fetch a slice of IPv4 addresses and a slice of IPv6 addresses from the bridge.
	addrsv4, addrsv6, err := i.addresses()
	if err != nil {
		return fmt.Errorf("Failed to verify ip addresses: %v", err)
	}

	addrv4, _ := selectIPv4Address(addrsv4, config.AddressIPv4)

	// Verify that the bridge does have an IPv4 address.
	if addrv4.IPNet == nil {
		return &ErrNoIPAddr{}
	}

	// Verify that the bridge IPv4 address matches the requested configuration.
	if config.AddressIPv4 != nil && !addrv4.IP.Equal(config.AddressIPv4.IP) {
		return &IPv4AddrNoMatchError{IP: addrv4.IP, CfgIP: config.AddressIPv4.IP}
	}

	// Verify that one of the bridge IPv6 addresses matches the requested
	// configuration.
	if config.EnableIPv6 && !findIPv6Address(netlink.Addr{IPNet: bridgeIPv6}, addrsv6) {
		return (*IPv6AddrNoMatchError)(bridgeIPv6)
	}

	// Release any residual IPv6 address that might be there because of older daemon instances
	for _, addrv6 := range addrsv6 {
		if addrv6.IP.IsGlobalUnicast() && !types.CompareIPNet(addrv6.IPNet, i.bridgeIPv6) {
			if err := i.nlh.AddrDel(i.Link, &addrv6); err != nil {
				logrus.Warnf("Failed to remove residual IPv6 address %s from bridge: %v", addrv6.IPNet, err)
			}
		}
	}

	return nil
}

func findIPv6Address(addr netlink.Addr, addresses []netlink.Addr) bool {
	for _, addrv6 := range addresses {
		if addrv6.String() == addr.String() {
			return true
		}
	}
	return false
}

func bridgeInterfaceExists(name string) (bool, error) {
	nlh := ns.NlHandle()
	link, err := nlh.LinkByName(name)
	if err != nil {
		if strings.Contains(err.Error(), "Link not found") {
			return false, nil
		}
		return false, fmt.Errorf("failed to check bridge interface existence: %v", err)
	}

	if link.Type() == "bridge" {
		return true, nil
	}
	return false, fmt.Errorf("existing interface %s is not a bridge", name)
}
