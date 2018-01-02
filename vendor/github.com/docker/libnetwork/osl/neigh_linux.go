package osl

import (
	"bytes"
	"fmt"
	"net"

	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

// NeighOption is a function option type to set interface options
type NeighOption func(nh *neigh)

type neigh struct {
	dstIP    net.IP
	dstMac   net.HardwareAddr
	linkName string
	linkDst  string
	family   int
}

func (n *networkNamespace) findNeighbor(dstIP net.IP, dstMac net.HardwareAddr) *neigh {
	n.Lock()
	defer n.Unlock()

	for _, nh := range n.neighbors {
		if nh.dstIP.Equal(dstIP) && bytes.Equal(nh.dstMac, dstMac) {
			return nh
		}
	}

	return nil
}

func (n *networkNamespace) DeleteNeighbor(dstIP net.IP, dstMac net.HardwareAddr, osDelete bool) error {
	var (
		iface netlink.Link
		err   error
	)

	nh := n.findNeighbor(dstIP, dstMac)
	if nh == nil {
		return fmt.Errorf("could not find the neighbor entry to delete")
	}

	if osDelete {
		n.Lock()
		nlh := n.nlHandle
		n.Unlock()

		if nh.linkDst != "" {
			iface, err = nlh.LinkByName(nh.linkDst)
			if err != nil {
				return fmt.Errorf("could not find interface with destination name %s: %v",
					nh.linkDst, err)
			}
		}

		nlnh := &netlink.Neigh{
			IP:     dstIP,
			State:  netlink.NUD_PERMANENT,
			Family: nh.family,
		}

		if nlnh.Family > 0 {
			nlnh.HardwareAddr = dstMac
			nlnh.Flags = netlink.NTF_SELF
		}

		if nh.linkDst != "" {
			nlnh.LinkIndex = iface.Attrs().Index
		}

		// If the kernel deletion fails for the neighbor entry still remote it
		// from the namespace cache. Otherwise if the neighbor moves back to the
		// same host again, kernel update can fail.
		if err := nlh.NeighDel(nlnh); err != nil {
			logrus.Warnf("Deleting neighbor IP %s, mac %s failed, %v", dstIP, dstMac, err)
		}

		// Delete the dynamic entry in the bridge
		if nlnh.Family > 0 {
			nlnh := &netlink.Neigh{
				IP:     dstIP,
				Family: nh.family,
			}

			nlnh.HardwareAddr = dstMac
			nlnh.Flags = netlink.NTF_MASTER
			if nh.linkDst != "" {
				nlnh.LinkIndex = iface.Attrs().Index
			}
			nlh.NeighDel(nlnh)
		}
	}

	n.Lock()
	for i, nh := range n.neighbors {
		if nh.dstIP.Equal(dstIP) && bytes.Equal(nh.dstMac, dstMac) {
			n.neighbors = append(n.neighbors[:i], n.neighbors[i+1:]...)
			break
		}
	}
	n.Unlock()
	logrus.Debugf("Neighbor entry deleted for IP %v, mac %v", dstIP, dstMac)

	return nil
}

func (n *networkNamespace) AddNeighbor(dstIP net.IP, dstMac net.HardwareAddr, force bool, options ...NeighOption) error {
	var (
		iface netlink.Link
		err   error
	)

	// If the namespace already has the neighbor entry but the AddNeighbor is called
	// because of a miss notification (force flag) program the kernel anyway.
	nh := n.findNeighbor(dstIP, dstMac)
	if nh != nil {
		if !force {
			logrus.Warnf("Neighbor entry already present for IP %v, mac %v", dstIP, dstMac)
			return nil
		}
		logrus.Warnf("Force kernel update, Neighbor entry already present for IP %v, mac %v", dstIP, dstMac)
	}

	nh = &neigh{
		dstIP:  dstIP,
		dstMac: dstMac,
	}

	nh.processNeighOptions(options...)

	if nh.linkName != "" {
		nh.linkDst = n.findDst(nh.linkName, false)
		if nh.linkDst == "" {
			return fmt.Errorf("could not find the interface with name %s", nh.linkName)
		}
	}

	n.Lock()
	nlh := n.nlHandle
	n.Unlock()

	if nh.linkDst != "" {
		iface, err = nlh.LinkByName(nh.linkDst)
		if err != nil {
			return fmt.Errorf("could not find interface with destination name %s: %v",
				nh.linkDst, err)
		}
	}

	nlnh := &netlink.Neigh{
		IP:           dstIP,
		HardwareAddr: dstMac,
		State:        netlink.NUD_PERMANENT,
		Family:       nh.family,
	}

	if nlnh.Family > 0 {
		nlnh.Flags = netlink.NTF_SELF
	}

	if nh.linkDst != "" {
		nlnh.LinkIndex = iface.Attrs().Index
	}

	if err := nlh.NeighSet(nlnh); err != nil {
		return fmt.Errorf("could not add neighbor entry: %v", err)
	}

	n.Lock()
	n.neighbors = append(n.neighbors, nh)
	n.Unlock()
	logrus.Debugf("Neighbor entry added for IP %v, mac %v", dstIP, dstMac)

	return nil
}
