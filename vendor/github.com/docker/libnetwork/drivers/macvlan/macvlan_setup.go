package macvlan

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/docker/libnetwork/ns"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

const (
	dummyPrefix      = "dm-" // macvlan prefix for dummy parent interface
	macvlanKernelVer = 3     // minimum macvlan kernel support
	macvlanMajorVer  = 9     // minimum macvlan major kernel support
)

// Create the macvlan slave specifying the source name
func createMacVlan(containerIfName, parent, macvlanMode string) (string, error) {
	// Set the macvlan mode. Default is bridge mode
	mode, err := setMacVlanMode(macvlanMode)
	if err != nil {
		return "", fmt.Errorf("Unsupported %s macvlan mode: %v", macvlanMode, err)
	}
	// verify the Docker host interface acting as the macvlan parent iface exists
	if !parentExists(parent) {
		return "", fmt.Errorf("the requested parent interface %s was not found on the Docker host", parent)
	}
	// Get the link for the master index (Example: the docker host eth iface)
	parentLink, err := ns.NlHandle().LinkByName(parent)
	if err != nil {
		return "", fmt.Errorf("error occoured looking up the %s parent iface %s error: %s", macvlanType, parent, err)
	}
	// Create a macvlan link
	macvlan := &netlink.Macvlan{
		LinkAttrs: netlink.LinkAttrs{
			Name:        containerIfName,
			ParentIndex: parentLink.Attrs().Index,
		},
		Mode: mode,
	}
	if err := ns.NlHandle().LinkAdd(macvlan); err != nil {
		// If a user creates a macvlan and ipvlan on same parent, only one slave iface can be active at a time.
		return "", fmt.Errorf("failed to create the %s port: %v", macvlanType, err)
	}

	return macvlan.Attrs().Name, nil
}

// setMacVlanMode setter for one of the four macvlan port types
func setMacVlanMode(mode string) (netlink.MacvlanMode, error) {
	switch mode {
	case modePrivate:
		return netlink.MACVLAN_MODE_PRIVATE, nil
	case modeVepa:
		return netlink.MACVLAN_MODE_VEPA, nil
	case modeBridge:
		return netlink.MACVLAN_MODE_BRIDGE, nil
	case modePassthru:
		return netlink.MACVLAN_MODE_PASSTHRU, nil
	default:
		return 0, fmt.Errorf("unknown macvlan mode: %s", mode)
	}
}

// parentExists checks if the specified interface exists in the default namespace
func parentExists(ifaceStr string) bool {
	_, err := ns.NlHandle().LinkByName(ifaceStr)
	if err != nil {
		return false
	}

	return true
}

// createVlanLink parses sub-interfaces and vlan id for creation
func createVlanLink(parentName string) error {
	if strings.Contains(parentName, ".") {
		parent, vidInt, err := parseVlan(parentName)
		if err != nil {
			return err
		}
		// VLAN identifier or VID is a 12-bit field specifying the VLAN to which the frame belongs
		if vidInt > 4094 || vidInt < 1 {
			return fmt.Errorf("vlan id must be between 1-4094, received: %d", vidInt)
		}
		// get the parent link to attach a vlan subinterface
		parentLink, err := ns.NlHandle().LinkByName(parent)
		if err != nil {
			return fmt.Errorf("failed to find master interface %s on the Docker host: %v", parent, err)
		}
		vlanLink := &netlink.Vlan{
			LinkAttrs: netlink.LinkAttrs{
				Name:        parentName,
				ParentIndex: parentLink.Attrs().Index,
			},
			VlanId: vidInt,
		}
		// create the subinterface
		if err := ns.NlHandle().LinkAdd(vlanLink); err != nil {
			return fmt.Errorf("failed to create %s vlan link: %v", vlanLink.Name, err)
		}
		// Bring the new netlink iface up
		if err := ns.NlHandle().LinkSetUp(vlanLink); err != nil {
			return fmt.Errorf("failed to enable %s the macvlan parent link %v", vlanLink.Name, err)
		}
		logrus.Debugf("Added a vlan tagged netlink subinterface: %s with a vlan id: %d", parentName, vidInt)
		return nil
	}

	return fmt.Errorf("invalid subinterface vlan name %s, example formatting is eth0.10", parentName)
}

// delVlanLink verifies only sub-interfaces with a vlan id get deleted
func delVlanLink(linkName string) error {
	if strings.Contains(linkName, ".") {
		_, _, err := parseVlan(linkName)
		if err != nil {
			return err
		}
		// delete the vlan subinterface
		vlanLink, err := ns.NlHandle().LinkByName(linkName)
		if err != nil {
			return fmt.Errorf("failed to find interface %s on the Docker host : %v", linkName, err)
		}
		// verify a parent interface isn't being deleted
		if vlanLink.Attrs().ParentIndex == 0 {
			return fmt.Errorf("interface %s does not appear to be a slave device: %v", linkName, err)
		}
		// delete the macvlan slave device
		if err := ns.NlHandle().LinkDel(vlanLink); err != nil {
			return fmt.Errorf("failed to delete  %s link: %v", linkName, err)
		}
		logrus.Debugf("Deleted a vlan tagged netlink subinterface: %s", linkName)
	}
	// if the subinterface doesn't parse to iface.vlan_id leave the interface in
	// place since it could be a user specified name not created by the driver.
	return nil
}

// parseVlan parses and verifies a slave interface name: -o parent=eth0.10
func parseVlan(linkName string) (string, int, error) {
	// parse -o parent=eth0.10
	splitName := strings.Split(linkName, ".")
	if len(splitName) != 2 {
		return "", 0, fmt.Errorf("required interface name format is: name.vlan_id, ex. eth0.10 for vlan 10, instead received %s", linkName)
	}
	parent, vidStr := splitName[0], splitName[1]
	// validate type and convert vlan id to int
	vidInt, err := strconv.Atoi(vidStr)
	if err != nil {
		return "", 0, fmt.Errorf("unable to parse a valid vlan id from: %s (ex. eth0.10 for vlan 10)", vidStr)
	}
	// Check if the interface exists
	if !parentExists(parent) {
		return "", 0, fmt.Errorf("-o parent interface does was not found on the host: %s", parent)
	}

	return parent, vidInt, nil
}

// createDummyLink creates a dummy0 parent link
func createDummyLink(dummyName, truncNetID string) error {
	// create a parent interface since one was not specified
	parent := &netlink.Dummy{
		LinkAttrs: netlink.LinkAttrs{
			Name: dummyName,
		},
	}
	if err := ns.NlHandle().LinkAdd(parent); err != nil {
		return err
	}
	parentDummyLink, err := ns.NlHandle().LinkByName(dummyName)
	if err != nil {
		return fmt.Errorf("error occoured looking up the %s parent iface %s error: %s", macvlanType, dummyName, err)
	}
	// bring the new netlink iface up
	if err := ns.NlHandle().LinkSetUp(parentDummyLink); err != nil {
		return fmt.Errorf("failed to enable %s the macvlan parent link: %v", dummyName, err)
	}

	return nil
}

// delDummyLink deletes the link type dummy used when -o parent is not passed
func delDummyLink(linkName string) error {
	// delete the vlan subinterface
	dummyLink, err := ns.NlHandle().LinkByName(linkName)
	if err != nil {
		return fmt.Errorf("failed to find link %s on the Docker host : %v", linkName, err)
	}
	// verify a parent interface is being deleted
	if dummyLink.Attrs().ParentIndex != 0 {
		return fmt.Errorf("link %s is not a parent dummy interface", linkName)
	}
	// delete the macvlan dummy device
	if err := ns.NlHandle().LinkDel(dummyLink); err != nil {
		return fmt.Errorf("failed to delete the dummy %s link: %v", linkName, err)
	}
	logrus.Debugf("Deleted a dummy parent link: %s", linkName)

	return nil
}

// getDummyName returns the name of a dummy parent with truncated net ID and driver prefix
func getDummyName(netID string) string {
	return dummyPrefix + netID
}
