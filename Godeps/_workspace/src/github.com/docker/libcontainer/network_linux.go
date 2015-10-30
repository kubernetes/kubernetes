// +build linux

package libcontainer

import (
	"fmt"
	"io/ioutil"
	"net"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/docker/libcontainer/configs"
	"github.com/docker/libcontainer/netlink"
	"github.com/docker/libcontainer/utils"
)

var strategies = map[string]networkStrategy{
	"veth":     &veth{},
	"loopback": &loopback{},
}

// networkStrategy represents a specific network configuration for
// a container's networking stack
type networkStrategy interface {
	create(*network, int) error
	initialize(*network) error
	detach(*configs.Network) error
	attach(*configs.Network) error
}

// getStrategy returns the specific network strategy for the
// provided type.
func getStrategy(tpe string) (networkStrategy, error) {
	s, exists := strategies[tpe]
	if !exists {
		return nil, fmt.Errorf("unknown strategy type %q", tpe)
	}
	return s, nil
}

// Returns the network statistics for the network interfaces represented by the NetworkRuntimeInfo.
func getNetworkInterfaceStats(interfaceName string) (*NetworkInterface, error) {
	out := &NetworkInterface{Name: interfaceName}
	// This can happen if the network runtime information is missing - possible if the
	// container was created by an old version of libcontainer.
	if interfaceName == "" {
		return out, nil
	}
	type netStatsPair struct {
		// Where to write the output.
		Out *uint64
		// The network stats file to read.
		File string
	}
	// Ingress for host veth is from the container. Hence tx_bytes stat on the host veth is actually number of bytes received by the container.
	netStats := []netStatsPair{
		{Out: &out.RxBytes, File: "tx_bytes"},
		{Out: &out.RxPackets, File: "tx_packets"},
		{Out: &out.RxErrors, File: "tx_errors"},
		{Out: &out.RxDropped, File: "tx_dropped"},

		{Out: &out.TxBytes, File: "rx_bytes"},
		{Out: &out.TxPackets, File: "rx_packets"},
		{Out: &out.TxErrors, File: "rx_errors"},
		{Out: &out.TxDropped, File: "rx_dropped"},
	}
	for _, netStat := range netStats {
		data, err := readSysfsNetworkStats(interfaceName, netStat.File)
		if err != nil {
			return nil, err
		}
		*(netStat.Out) = data
	}
	return out, nil
}

// Reads the specified statistics available under /sys/class/net/<EthInterface>/statistics
func readSysfsNetworkStats(ethInterface, statsFile string) (uint64, error) {
	data, err := ioutil.ReadFile(filepath.Join("/sys/class/net", ethInterface, "statistics", statsFile))
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

// loopback is a network strategy that provides a basic loopback device
type loopback struct {
}

func (l *loopback) create(n *network, nspid int) error {
	return nil
}

func (l *loopback) initialize(config *network) error {
	iface, err := net.InterfaceByName("lo")
	if err != nil {
		return err
	}
	return netlink.NetworkLinkUp(iface)
}

func (l *loopback) attach(n *configs.Network) (err error) {
	return nil
}

func (l *loopback) detach(n *configs.Network) (err error) {
	return nil
}

// veth is a network strategy that uses a bridge and creates
// a veth pair, one that is attached to the bridge on the host and the other
// is placed inside the container's namespace
type veth struct {
}

func (v *veth) detach(n *configs.Network) (err error) {
	bridge, err := net.InterfaceByName(n.Bridge)
	if err != nil {
		return err
	}
	host, err := net.InterfaceByName(n.HostInterfaceName)
	if err != nil {
		return err
	}
	if err := netlink.DelFromBridge(host, bridge); err != nil {
		return err
	}
	return nil
}

// attach a container network interface to an external network
func (v *veth) attach(n *configs.Network) (err error) {
	bridge, err := net.InterfaceByName(n.Bridge)
	if err != nil {
		return err
	}
	host, err := net.InterfaceByName(n.HostInterfaceName)
	if err != nil {
		return err
	}
	if err := netlink.AddToBridge(host, bridge); err != nil {
		return err
	}
	if err := netlink.NetworkSetMTU(host, n.Mtu); err != nil {
		return err
	}
	if n.HairpinMode {
		if err := netlink.SetHairpinMode(host, true); err != nil {
			return err
		}
	}
	if err := netlink.NetworkLinkUp(host); err != nil {
		return err
	}

	return nil
}

func (v *veth) create(n *network, nspid int) (err error) {
	tmpName, err := v.generateTempPeerName()
	if err != nil {
		return err
	}
	n.TempVethPeerName = tmpName
	defer func() {
		if err != nil {
			netlink.NetworkLinkDel(n.HostInterfaceName)
			netlink.NetworkLinkDel(n.TempVethPeerName)
		}
	}()
	if n.Bridge == "" {
		return fmt.Errorf("bridge is not specified")
	}
	if err := netlink.NetworkCreateVethPair(n.HostInterfaceName, n.TempVethPeerName, n.TxQueueLen); err != nil {
		return err
	}
	if err := v.attach(&n.Network); err != nil {
		return err
	}
	child, err := net.InterfaceByName(n.TempVethPeerName)
	if err != nil {
		return err
	}
	return netlink.NetworkSetNsPid(child, nspid)
}

func (v *veth) generateTempPeerName() (string, error) {
	return utils.GenerateRandomName("veth", 7)
}

func (v *veth) initialize(config *network) error {
	peer := config.TempVethPeerName
	if peer == "" {
		return fmt.Errorf("peer is not specified")
	}
	child, err := net.InterfaceByName(peer)
	if err != nil {
		return err
	}
	if err := netlink.NetworkLinkDown(child); err != nil {
		return err
	}
	if err := netlink.NetworkChangeName(child, config.Name); err != nil {
		return err
	}
	// get the interface again after we changed the name as the index also changes.
	if child, err = net.InterfaceByName(config.Name); err != nil {
		return err
	}
	if config.MacAddress != "" {
		if err := netlink.NetworkSetMacAddress(child, config.MacAddress); err != nil {
			return err
		}
	}
	ip, ipNet, err := net.ParseCIDR(config.Address)
	if err != nil {
		return err
	}
	if err := netlink.NetworkLinkAddIp(child, ip, ipNet); err != nil {
		return err
	}
	if config.IPv6Address != "" {
		if ip, ipNet, err = net.ParseCIDR(config.IPv6Address); err != nil {
			return err
		}
		if err := netlink.NetworkLinkAddIp(child, ip, ipNet); err != nil {
			return err
		}
	}
	if err := netlink.NetworkSetMTU(child, config.Mtu); err != nil {
		return err
	}
	if err := netlink.NetworkLinkUp(child); err != nil {
		return err
	}
	if config.Gateway != "" {
		if err := netlink.AddDefaultGw(config.Gateway, config.Name); err != nil {
			return err
		}
	}
	if config.IPv6Gateway != "" {
		if err := netlink.AddDefaultGw(config.IPv6Gateway, config.Name); err != nil {
			return err
		}
	}
	return nil
}
