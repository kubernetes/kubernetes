package bridge

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"

	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

var bridgeIPv6 *net.IPNet

const (
	bridgeIPv6Str          = "fe80::1/64"
	ipv6ForwardConfPerm    = 0644
	ipv6ForwardConfDefault = "/proc/sys/net/ipv6/conf/default/forwarding"
	ipv6ForwardConfAll     = "/proc/sys/net/ipv6/conf/all/forwarding"
)

func init() {
	// We allow ourselves to panic in this special case because we indicate a
	// failure to parse a compile-time define constant.
	var err error
	if bridgeIPv6, err = types.ParseCIDR(bridgeIPv6Str); err != nil {
		panic(fmt.Sprintf("Cannot parse default bridge IPv6 address %q: %v", bridgeIPv6Str, err))
	}
}

func setupBridgeIPv6(config *networkConfiguration, i *bridgeInterface) error {
	procFile := "/proc/sys/net/ipv6/conf/" + config.BridgeName + "/disable_ipv6"
	ipv6BridgeData, err := ioutil.ReadFile(procFile)
	if err != nil {
		return fmt.Errorf("Cannot read IPv6 setup for bridge %v: %v", config.BridgeName, err)
	}
	// Enable IPv6 on the bridge only if it isn't already enabled
	if ipv6BridgeData[0] != '0' {
		if err := ioutil.WriteFile(procFile, []byte{'0', '\n'}, ipv6ForwardConfPerm); err != nil {
			return fmt.Errorf("Unable to enable IPv6 addresses on bridge: %v", err)
		}
	}

	// Store bridge network and default gateway
	i.bridgeIPv6 = bridgeIPv6
	i.gatewayIPv6 = i.bridgeIPv6.IP

	if err := i.programIPv6Address(); err != nil {
		return err
	}

	if config.AddressIPv6 == nil {
		return nil
	}

	// Store the user specified bridge network and network gateway and program it
	i.bridgeIPv6 = config.AddressIPv6
	i.gatewayIPv6 = config.AddressIPv6.IP

	if err := i.programIPv6Address(); err != nil {
		return err
	}

	// Setting route to global IPv6 subnet
	logrus.Debugf("Adding route to IPv6 network %s via device %s", config.AddressIPv6.String(), config.BridgeName)
	err = i.nlh.RouteAdd(&netlink.Route{
		Scope:     netlink.SCOPE_UNIVERSE,
		LinkIndex: i.Link.Attrs().Index,
		Dst:       config.AddressIPv6,
	})
	if err != nil && !os.IsExist(err) {
		logrus.Errorf("Could not add route to IPv6 network %s via device %s", config.AddressIPv6.String(), config.BridgeName)
	}

	return nil
}

func setupGatewayIPv6(config *networkConfiguration, i *bridgeInterface) error {
	if config.AddressIPv6 == nil {
		return &ErrInvalidContainerSubnet{}
	}
	if !config.AddressIPv6.Contains(config.DefaultGatewayIPv6) {
		return &ErrInvalidGateway{}
	}

	// Store requested default gateway
	i.gatewayIPv6 = config.DefaultGatewayIPv6

	return nil
}

func setupIPv6Forwarding(config *networkConfiguration, i *bridgeInterface) error {
	// Get current IPv6 default forwarding setup
	ipv6ForwardDataDefault, err := ioutil.ReadFile(ipv6ForwardConfDefault)
	if err != nil {
		return fmt.Errorf("Cannot read IPv6 default forwarding setup: %v", err)
	}
	// Enable IPv6 default forwarding only if it is not already enabled
	if ipv6ForwardDataDefault[0] != '1' {
		if err := ioutil.WriteFile(ipv6ForwardConfDefault, []byte{'1', '\n'}, ipv6ForwardConfPerm); err != nil {
			logrus.Warnf("Unable to enable IPv6 default forwarding: %v", err)
		}
	}

	// Get current IPv6 all forwarding setup
	ipv6ForwardDataAll, err := ioutil.ReadFile(ipv6ForwardConfAll)
	if err != nil {
		return fmt.Errorf("Cannot read IPv6 all forwarding setup: %v", err)
	}
	// Enable IPv6 all forwarding only if it is not already enabled
	if ipv6ForwardDataAll[0] != '1' {
		if err := ioutil.WriteFile(ipv6ForwardConfAll, []byte{'1', '\n'}, ipv6ForwardConfPerm); err != nil {
			logrus.Warnf("Unable to enable IPv6 all forwarding: %v", err)
		}
	}

	return nil
}
