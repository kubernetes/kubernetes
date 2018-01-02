package bridge

import (
	"fmt"

	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/libnetwork/netutils"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

// SetupDevice create a new bridge interface/
func setupDevice(config *networkConfiguration, i *bridgeInterface) error {
	var setMac bool

	// We only attempt to create the bridge when the requested device name is
	// the default one.
	if config.BridgeName != DefaultBridgeName && config.DefaultBridge {
		return NonDefaultBridgeExistError(config.BridgeName)
	}

	// Set the bridgeInterface netlink.Bridge.
	i.Link = &netlink.Bridge{
		LinkAttrs: netlink.LinkAttrs{
			Name: config.BridgeName,
		},
	}

	// Only set the bridge's MAC address if the kernel version is > 3.3, as it
	// was not supported before that.
	kv, err := kernel.GetKernelVersion()
	if err != nil {
		logrus.Errorf("Failed to check kernel versions: %v. Will not assign a MAC address to the bridge interface", err)
	} else {
		setMac = kv.Kernel > 3 || (kv.Kernel == 3 && kv.Major >= 3)
	}

	if err = i.nlh.LinkAdd(i.Link); err != nil {
		logrus.Debugf("Failed to create bridge %s via netlink. Trying ioctl", config.BridgeName)
		return ioctlCreateBridge(config.BridgeName, setMac)
	}

	if setMac {
		hwAddr := netutils.GenerateRandomMAC()
		if err = i.nlh.LinkSetHardwareAddr(i.Link, hwAddr); err != nil {
			return fmt.Errorf("failed to set bridge mac-address %s : %s", hwAddr, err.Error())
		}
		logrus.Debugf("Setting bridge mac address to %s", hwAddr)
	}
	return err
}

// SetupDeviceUp ups the given bridge interface.
func setupDeviceUp(config *networkConfiguration, i *bridgeInterface) error {
	err := i.nlh.LinkSetUp(i.Link)
	if err != nil {
		return fmt.Errorf("Failed to set link up for %s: %v", config.BridgeName, err)
	}

	// Attempt to update the bridge interface to refresh the flags status,
	// ignoring any failure to do so.
	if lnk, err := i.nlh.LinkByName(config.BridgeName); err == nil {
		i.Link = lnk
	} else {
		logrus.Warnf("Failed to retrieve link for interface (%s): %v", config.BridgeName, err)
	}
	return nil
}
