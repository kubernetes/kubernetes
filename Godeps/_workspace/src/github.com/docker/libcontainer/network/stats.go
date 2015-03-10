package network

import (
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

type NetworkStats struct {
	RxBytes   uint64 `json:"rx_bytes"`
	RxPackets uint64 `json:"rx_packets"`
	RxErrors  uint64 `json:"rx_errors"`
	RxDropped uint64 `json:"rx_dropped"`
	TxBytes   uint64 `json:"tx_bytes"`
	TxPackets uint64 `json:"tx_packets"`
	TxErrors  uint64 `json:"tx_errors"`
	TxDropped uint64 `json:"tx_dropped"`
}

// Returns the network statistics for the network interfaces represented by the NetworkRuntimeInfo.
func GetStats(networkState *NetworkState) (*NetworkStats, error) {
	// This can happen if the network runtime information is missing - possible if the container was created by an old version of libcontainer.
	if networkState.VethHost == "" {
		return &NetworkStats{}, nil
	}

	out := &NetworkStats{}

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
		data, err := readSysfsNetworkStats(networkState.VethHost, netStat.File)
		if err != nil {
			return nil, err
		}
		*(netStat.Out) = data
	}

	return out, nil
}

// Reads the specified statistics available under /sys/class/net/<EthInterface>/statistics
func readSysfsNetworkStats(ethInterface, statsFile string) (uint64, error) {
	fullPath := filepath.Join("/sys/class/net", ethInterface, "statistics", statsFile)
	data, err := ioutil.ReadFile(fullPath)
	if err != nil {
		return 0, err
	}
	value, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, err
	}

	return value, err
}
