package bridge

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"path/filepath"

	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

func selectIPv4Address(addresses []netlink.Addr, selector *net.IPNet) (netlink.Addr, error) {
	if len(addresses) == 0 {
		return netlink.Addr{}, errors.New("unable to select an address as the address pool is empty")
	}
	if selector != nil {
		for _, addr := range addresses {
			if selector.Contains(addr.IP) {
				return addr, nil
			}
		}
	}
	return addresses[0], nil
}

func setupBridgeIPv4(config *networkConfiguration, i *bridgeInterface) error {
	addrv4List, _, err := i.addresses()
	if err != nil {
		return fmt.Errorf("failed to retrieve bridge interface addresses: %v", err)
	}

	addrv4, _ := selectIPv4Address(addrv4List, config.AddressIPv4)

	if !types.CompareIPNet(addrv4.IPNet, config.AddressIPv4) {
		if addrv4.IPNet != nil {
			if err := i.nlh.AddrDel(i.Link, &addrv4); err != nil {
				return fmt.Errorf("failed to remove current ip address from bridge: %v", err)
			}
		}
		logrus.Debugf("Assigning address to bridge interface %s: %s", config.BridgeName, config.AddressIPv4)
		if err := i.nlh.AddrAdd(i.Link, &netlink.Addr{IPNet: config.AddressIPv4}); err != nil {
			return &IPv4AddrAddError{IP: config.AddressIPv4, Err: err}
		}
	}

	// Store bridge network and default gateway
	i.bridgeIPv4 = config.AddressIPv4
	i.gatewayIPv4 = config.AddressIPv4.IP

	return nil
}

func setupGatewayIPv4(config *networkConfiguration, i *bridgeInterface) error {
	if !i.bridgeIPv4.Contains(config.DefaultGatewayIPv4) {
		return &ErrInvalidGateway{}
	}

	// Store requested default gateway
	i.gatewayIPv4 = config.DefaultGatewayIPv4

	return nil
}

func setupLoopbackAdressesRouting(config *networkConfiguration, i *bridgeInterface) error {
	sysPath := filepath.Join("/proc/sys/net/ipv4/conf", config.BridgeName, "route_localnet")
	ipv4LoRoutingData, err := ioutil.ReadFile(sysPath)
	if err != nil {
		return fmt.Errorf("Cannot read IPv4 local routing setup: %v", err)
	}
	// Enable loopback adresses routing only if it isn't already enabled
	if ipv4LoRoutingData[0] != '1' {
		if err := ioutil.WriteFile(sysPath, []byte{'1', '\n'}, 0644); err != nil {
			return fmt.Errorf("Unable to enable local routing for hairpin mode: %v", err)
		}
	}
	return nil
}
