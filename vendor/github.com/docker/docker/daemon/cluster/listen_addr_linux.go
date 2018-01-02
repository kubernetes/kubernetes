// +build linux

package cluster

import (
	"net"

	"github.com/vishvananda/netlink"
)

func (c *Cluster) resolveSystemAddr() (net.IP, error) {
	// Use the system's only device IP address, or fail if there are
	// multiple addresses to choose from.
	interfaces, err := netlink.LinkList()
	if err != nil {
		return nil, err
	}

	var (
		systemAddr      net.IP
		systemInterface string
		deviceFound     bool
	)

	for _, intf := range interfaces {
		// Skip non device or inactive interfaces
		if intf.Type() != "device" || intf.Attrs().Flags&net.FlagUp == 0 {
			continue
		}

		addrs, err := netlink.AddrList(intf, netlink.FAMILY_ALL)
		if err != nil {
			continue
		}

		var interfaceAddr4, interfaceAddr6 net.IP

		for _, addr := range addrs {
			ipAddr := addr.IPNet.IP

			// Skip loopback and link-local addresses
			if !ipAddr.IsGlobalUnicast() {
				continue
			}

			// At least one non-loopback device is found and it is administratively up
			deviceFound = true

			if ipAddr.To4() != nil {
				if interfaceAddr4 != nil {
					return nil, errMultipleIPs(intf.Attrs().Name, intf.Attrs().Name, interfaceAddr4, ipAddr)
				}
				interfaceAddr4 = ipAddr
			} else {
				if interfaceAddr6 != nil {
					return nil, errMultipleIPs(intf.Attrs().Name, intf.Attrs().Name, interfaceAddr6, ipAddr)
				}
				interfaceAddr6 = ipAddr
			}
		}

		// In the case that this interface has exactly one IPv4 address
		// and exactly one IPv6 address, favor IPv4 over IPv6.
		if interfaceAddr4 != nil {
			if systemAddr != nil {
				return nil, errMultipleIPs(systemInterface, intf.Attrs().Name, systemAddr, interfaceAddr4)
			}
			systemAddr = interfaceAddr4
			systemInterface = intf.Attrs().Name
		} else if interfaceAddr6 != nil {
			if systemAddr != nil {
				return nil, errMultipleIPs(systemInterface, intf.Attrs().Name, systemAddr, interfaceAddr6)
			}
			systemAddr = interfaceAddr6
			systemInterface = intf.Attrs().Name
		}
	}

	if systemAddr == nil {
		if !deviceFound {
			// If no non-loopback device type interface is found,
			// fall back to the regular auto-detection mechanism.
			// This is to cover the case where docker is running
			// inside a container (eths are in fact veths).
			return c.resolveSystemAddrViaSubnetCheck()
		}
		return nil, errNoIP
	}

	return systemAddr, nil
}
