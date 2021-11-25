// +build linux

package libcontainer

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/types"
	"github.com/vishvananda/netlink"
)

var strategies = map[string]networkStrategy{
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
func getNetworkInterfaceStats(interfaceName string) (*types.NetworkInterface, error) {
	out := &types.NetworkInterface{Name: interfaceName}
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
	return strconv.ParseUint(string(bytes.TrimSpace(data)), 10, 64)
}

// loopback is a network strategy that provides a basic loopback device
type loopback struct{}

func (l *loopback) create(n *network, nspid int) error {
	return nil
}

func (l *loopback) initialize(config *network) error {
	return netlink.LinkSetUp(&netlink.Device{LinkAttrs: netlink.LinkAttrs{Name: "lo"}})
}

func (l *loopback) attach(n *configs.Network) (err error) {
	return nil
}

func (l *loopback) detach(n *configs.Network) (err error) {
	return nil
}
