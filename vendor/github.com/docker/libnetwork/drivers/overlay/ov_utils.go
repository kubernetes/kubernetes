package overlay

import (
	"fmt"
	"strings"
	"syscall"

	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/osl"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netns"
)

var soTimeout = ns.NetlinkSocketsTimeout

func validateID(nid, eid string) error {
	if nid == "" {
		return fmt.Errorf("invalid network id")
	}

	if eid == "" {
		return fmt.Errorf("invalid endpoint id")
	}

	return nil
}

func createVethPair() (string, string, error) {
	defer osl.InitOSContext()()
	nlh := ns.NlHandle()

	// Generate a name for what will be the host side pipe interface
	name1, err := netutils.GenerateIfaceName(nlh, vethPrefix, vethLen)
	if err != nil {
		return "", "", fmt.Errorf("error generating veth name1: %v", err)
	}

	// Generate a name for what will be the sandbox side pipe interface
	name2, err := netutils.GenerateIfaceName(nlh, vethPrefix, vethLen)
	if err != nil {
		return "", "", fmt.Errorf("error generating veth name2: %v", err)
	}

	// Generate and add the interface pipe host <-> sandbox
	veth := &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: name1, TxQLen: 0},
		PeerName:  name2}
	if err := nlh.LinkAdd(veth); err != nil {
		return "", "", fmt.Errorf("error creating veth pair: %v", err)
	}

	return name1, name2, nil
}

func createVxlan(name string, vni uint32, mtu int) error {
	defer osl.InitOSContext()()

	vxlan := &netlink.Vxlan{
		LinkAttrs: netlink.LinkAttrs{Name: name, MTU: mtu},
		VxlanId:   int(vni),
		Learning:  true,
		Port:      vxlanPort,
		Proxy:     true,
		L3miss:    true,
		L2miss:    true,
	}

	if err := ns.NlHandle().LinkAdd(vxlan); err != nil {
		return fmt.Errorf("error creating vxlan interface: %v", err)
	}

	return nil
}

func deleteInterfaceBySubnet(brPrefix string, s *subnet) error {
	defer osl.InitOSContext()()

	nlh := ns.NlHandle()
	links, err := nlh.LinkList()
	if err != nil {
		return fmt.Errorf("failed to list interfaces while deleting bridge interface by subnet: %v", err)
	}

	for _, l := range links {
		name := l.Attrs().Name
		if _, ok := l.(*netlink.Bridge); ok && strings.HasPrefix(name, brPrefix) {
			addrList, err := nlh.AddrList(l, netlink.FAMILY_V4)
			if err != nil {
				logrus.Errorf("error getting AddressList for bridge %s", name)
				continue
			}
			for _, addr := range addrList {
				if netutils.NetworkOverlaps(addr.IPNet, s.subnetIP) {
					err = nlh.LinkDel(l)
					if err != nil {
						logrus.Errorf("error deleting bridge (%s) with subnet %v: %v", name, addr.IPNet, err)
					}
				}
			}
		}
	}
	return nil

}

func deleteInterface(name string) error {
	defer osl.InitOSContext()()

	link, err := ns.NlHandle().LinkByName(name)
	if err != nil {
		return fmt.Errorf("failed to find interface with name %s: %v", name, err)
	}

	if err := ns.NlHandle().LinkDel(link); err != nil {
		return fmt.Errorf("error deleting interface with name %s: %v", name, err)
	}

	return nil
}

func deleteVxlanByVNI(path string, vni uint32) error {
	defer osl.InitOSContext()()

	nlh := ns.NlHandle()
	if path != "" {
		ns, err := netns.GetFromPath(path)
		if err != nil {
			return fmt.Errorf("failed to get ns handle for %s: %v", path, err)
		}
		defer ns.Close()

		nlh, err = netlink.NewHandleAt(ns, syscall.NETLINK_ROUTE)
		if err != nil {
			return fmt.Errorf("failed to get netlink handle for ns %s: %v", path, err)
		}
		defer nlh.Delete()
		err = nlh.SetSocketTimeout(soTimeout)
		if err != nil {
			logrus.Warnf("Failed to set the timeout on the netlink handle sockets for vxlan deletion: %v", err)
		}
	}

	links, err := nlh.LinkList()
	if err != nil {
		return fmt.Errorf("failed to list interfaces while deleting vxlan interface by vni: %v", err)
	}

	for _, l := range links {
		if l.Type() == "vxlan" && (vni == 0 || l.(*netlink.Vxlan).VxlanId == int(vni)) {
			err = nlh.LinkDel(l)
			if err != nil {
				return fmt.Errorf("error deleting vxlan interface with id %d: %v", vni, err)
			}
			return nil
		}
	}

	return fmt.Errorf("could not find a vxlan interface to delete with id %d", vni)
}
