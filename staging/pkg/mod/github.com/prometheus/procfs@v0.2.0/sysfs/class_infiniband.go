// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package sysfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

const infinibandClassPath = "class/infiniband"

// InfiniBandCounters contains counter values from files in
// /sys/class/infiniband/<Name>/ports/<Port>/counters or
// /sys/class/infiniband/<Name>/ports/<Port>/counters_ext
// for a single port of one InfiniBand device.
type InfiniBandCounters struct {
	LegacyPortMulticastRcvPackets  *uint64 // counters_ext/port_multicast_rcv_packets
	LegacyPortMulticastXmitPackets *uint64 // counters_ext/port_multicast_xmit_packets
	LegacyPortRcvData64            *uint64 // counters_ext/port_rcv_data_64
	LegacyPortRcvPackets64         *uint64 // counters_ext/port_rcv_packets_64
	LegacyPortUnicastRcvPackets    *uint64 // counters_ext/port_unicast_rcv_packets
	LegacyPortUnicastXmitPackets   *uint64 // counters_ext/port_unicast_xmit_packets
	LegacyPortXmitData64           *uint64 // counters_ext/port_xmit_data_64
	LegacyPortXmitPackets64        *uint64 // counters_ext/port_xmit_packets_64

	LinkDowned                  *uint64 // counters/link_downed
	LinkErrorRecovery           *uint64 // counters/link_error_recovery
	MulticastRcvPackets         *uint64 // counters/multicast_rcv_packets
	MulticastXmitPackets        *uint64 // counters/multicast_xmit_packets
	PortRcvConstraintErrors     *uint64 // counters/port_rcv_constraint_errors
	PortRcvData                 *uint64 // counters/port_rcv_data
	PortRcvDiscards             *uint64 // counters/port_rcv_discards
	PortRcvErrors               *uint64 // counters/port_rcv_errors
	PortRcvPackets              *uint64 // counters/port_rcv_packets
	PortRcvRemotePhysicalErrors *uint64 // counters/port_rcv_remote_physical_errors
	PortRcvSwitchRelayErrors    *uint64 // counters/port_rcv_switch_relay_errors
	PortXmitConstraintErrors    *uint64 // counters/port_xmit_constraint_errors
	PortXmitData                *uint64 // counters/port_xmit_data
	PortXmitDiscards            *uint64 // counters/port_xmit_discards
	PortXmitPackets             *uint64 // counters/port_xmit_packets
	PortXmitWait                *uint64 // counters/port_xmit_wait
	SymbolError                 *uint64 // counters/symbol_error
	UnicastRcvPackets           *uint64 // counters/unicast_rcv_packets
	UnicastXmitPackets          *uint64 // counters/unicast_xmit_packets
	VL15Dropped                 *uint64 // counters/VL15_dropped
}

// InfiniBandPort contains info from files in
// /sys/class/infiniband/<Name>/ports/<Port>
// for a single port of one InfiniBand device.
type InfiniBandPort struct {
	Name        string
	Port        uint
	State       string // String representation from /sys/class/infiniband/<Name>/ports/<Port>/state
	StateID     uint   // ID from /sys/class/infiniband/<Name>/ports/<Port>/state
	PhysState   string // String representation from /sys/class/infiniband/<Name>/ports/<Port>/phys_state
	PhysStateID uint   // String representation from /sys/class/infiniband/<Name>/ports/<Port>/phys_state
	Rate        uint64 // in bytes/second from /sys/class/infiniband/<Name>/ports/<Port>/rate
	Counters    InfiniBandCounters
}

// InfiniBandDevice contains info from files in /sys/class/infiniband for a
// single InfiniBand device.
type InfiniBandDevice struct {
	Name            string
	BoardID         string // /sys/class/infiniband/<Name>/board_id
	FirmwareVersion string // /sys/class/infiniband/<Name>/fw_ver
	HCAType         string // /sys/class/infiniband/<Name>/hca_type
	Ports           map[uint]InfiniBandPort
}

// InfiniBandClass is a collection of every InfiniBand device in
// /sys/class/infiniband.
//
// The map keys are the names of the InfiniBand devices.
type InfiniBandClass map[string]InfiniBandDevice

// InfiniBandClass returns info for all InfiniBand devices read from
// /sys/class/infiniband.
func (fs FS) InfiniBandClass() (InfiniBandClass, error) {
	path := fs.sys.Path(infinibandClassPath)

	dirs, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	ibc := make(InfiniBandClass, len(dirs))
	for _, d := range dirs {
		device, err := fs.parseInfiniBandDevice(d.Name())
		if err != nil {
			return nil, err
		}

		ibc[device.Name] = *device
	}

	return ibc, nil
}

// Parse one InfiniBand device.
func (fs FS) parseInfiniBandDevice(name string) (*InfiniBandDevice, error) {
	path := fs.sys.Path(infinibandClassPath, name)
	device := InfiniBandDevice{Name: name}

	for _, f := range [...]string{"board_id", "fw_ver", "hca_type"} {
		name := filepath.Join(path, f)
		value, err := util.SysReadFile(name)
		if err != nil {
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		switch f {
		case "board_id":
			device.BoardID = value
		case "fw_ver":
			device.FirmwareVersion = value
		case "hca_type":
			device.HCAType = value
		}
	}

	portsPath := filepath.Join(path, "ports")
	ports, err := ioutil.ReadDir(portsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to list InfiniBand ports at %q: %v", portsPath, err)
	}

	device.Ports = make(map[uint]InfiniBandPort, len(ports))
	for _, d := range ports {
		port, err := fs.parseInfiniBandPort(name, d.Name())
		if err != nil {
			return nil, err
		}

		device.Ports[port.Port] = *port
	}

	return &device, nil
}

// Parse InfiniBand state. Expected format: "<id>: <string-representation>"
func parseState(s string) (uint, string, error) {
	parts := strings.Split(s, ":")
	if len(parts) != 2 {
		return 0, "", fmt.Errorf("failed to split %s into 'ID: NAME'", s)
	}
	name := strings.TrimSpace(parts[1])
	value, err := strconv.ParseUint(strings.TrimSpace(parts[0]), 10, 32)
	if err != nil {
		return 0, name, fmt.Errorf("failed to convert %s into uint", strings.TrimSpace(parts[0]))
	}
	id := uint(value)
	return id, name, nil
}

// Parse rate (example: "100 Gb/sec (4X EDR)") and return it as bytes/second
func parseRate(s string) (uint64, error) {
	parts := strings.Split(s, "Gb/sec")
	if len(parts) != 2 {
		return 0, fmt.Errorf("failed to split '%s' by 'Gb/sec'", s)
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 32)
	if err != nil {
		return 0, fmt.Errorf("failed to convert %s into uint", strings.TrimSpace(parts[0]))
	}
	// Convert Gb/s into bytes/s
	rate := uint64(value * 125000000)
	return rate, nil
}

// parseInfiniBandPort scans predefined files in /sys/class/infiniband/<device>/ports/<port>
// directory and gets their contents.
func (fs FS) parseInfiniBandPort(name string, port string) (*InfiniBandPort, error) {
	portNumber, err := strconv.ParseUint(port, 10, 32)
	if err != nil {
		return nil, fmt.Errorf("failed to convert %s into uint", port)
	}
	ibp := InfiniBandPort{Name: name, Port: uint(portNumber)}

	portPath := fs.sys.Path(infinibandClassPath, name, "ports", port)
	content, err := ioutil.ReadFile(filepath.Join(portPath, "state"))
	if err != nil {
		return nil, err
	}
	id, name, err := parseState(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse state file in %s: %s", portPath, err)
	}
	ibp.State = name
	ibp.StateID = id

	content, err = ioutil.ReadFile(filepath.Join(portPath, "phys_state"))
	if err != nil {
		return nil, err
	}
	id, name, err = parseState(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse phys_state file in %s: %s", portPath, err)
	}
	ibp.PhysState = name
	ibp.PhysStateID = id

	content, err = ioutil.ReadFile(filepath.Join(portPath, "rate"))
	if err != nil {
		return nil, err
	}
	ibp.Rate, err = parseRate(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse rate file in %s: %s", portPath, err)
	}

	counters, err := parseInfiniBandCounters(portPath)
	if err != nil {
		return nil, err
	}
	ibp.Counters = *counters

	return &ibp, nil
}

func parseInfiniBandCounters(portPath string) (*InfiniBandCounters, error) {
	var counters InfiniBandCounters

	path := filepath.Join(portPath, "counters")
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	for _, f := range files {
		if !f.Mode().IsRegular() {
			continue
		}

		name := filepath.Join(path, f.Name())
		value, err := util.SysReadFile(name)
		if err != nil {
			if os.IsNotExist(err) || os.IsPermission(err) || err.Error() == "operation not supported" || err.Error() == "invalid argument" {
				continue
			}
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		// According to Mellanox, the metrics port_rcv_data, port_xmit_data,
		// port_rcv_data_64, and port_xmit_data_64 "are divided by 4 unconditionally"
		// as they represent the amount of data being transmitted and received per lane.
		// Mellanox cards have 4 lanes per port, so all values must be multiplied by 4
		// to get the expected value.

		vp := util.NewValueParser(value)

		switch f.Name() {
		case "link_downed":
			counters.LinkDowned = vp.PUInt64()
		case "link_error_recovery":
			counters.LinkErrorRecovery = vp.PUInt64()
		case "multicast_rcv_packets":
			counters.MulticastRcvPackets = vp.PUInt64()
		case "multicast_xmit_packets":
			counters.MulticastXmitPackets = vp.PUInt64()
		case "port_rcv_constraint_errors":
			counters.PortRcvConstraintErrors = vp.PUInt64()
		case "port_rcv_data":
			counters.PortRcvData = vp.PUInt64()
			if counters.PortRcvData != nil {
				*counters.PortRcvData *= 4
			}
		case "port_rcv_discards":
			counters.PortRcvDiscards = vp.PUInt64()
		case "port_rcv_errors":
			counters.PortRcvErrors = vp.PUInt64()
		case "port_rcv_packets":
			counters.PortRcvPackets = vp.PUInt64()
		case "port_rcv_remote_physical_errors":
			counters.PortRcvRemotePhysicalErrors = vp.PUInt64()
		case "port_rcv_switch_relay_errors":
			counters.PortRcvSwitchRelayErrors = vp.PUInt64()
		case "port_xmit_constraint_errors":
			counters.PortXmitConstraintErrors = vp.PUInt64()
		case "port_xmit_data":
			counters.PortXmitData = vp.PUInt64()
			if counters.PortXmitData != nil {
				*counters.PortXmitData *= 4
			}
		case "port_xmit_discards":
			counters.PortXmitDiscards = vp.PUInt64()
		case "port_xmit_packets":
			counters.PortXmitPackets = vp.PUInt64()
		case "port_xmit_wait":
			counters.PortXmitWait = vp.PUInt64()
		case "symbol_error":
			counters.SymbolError = vp.PUInt64()
		case "unicast_rcv_packets":
			counters.UnicastRcvPackets = vp.PUInt64()
		case "unicast_xmit_packets":
			counters.UnicastXmitPackets = vp.PUInt64()
		case "VL15_dropped":
			counters.VL15Dropped = vp.PUInt64()
		}

		if err := vp.Err(); err != nil {
			// Ugly workaround for handling https://github.com/prometheus/node_exporter/issues/966
			// when counters are `N/A (not available)`.
			// This was already patched and submitted, see
			// https://www.spinics.net/lists/linux-rdma/msg68596.html
			// Remove this as soon as the fix lands in the enterprise distros.
			if strings.Contains(value, "N/A (no PMA)") {
				continue
			}
			return nil, err
		}
	}

	// Parse legacy counters
	path = filepath.Join(portPath, "counters_ext")
	files, err = ioutil.ReadDir(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	for _, f := range files {
		if !f.Mode().IsRegular() {
			continue
		}

		name := filepath.Join(path, f.Name())
		value, err := util.SysReadFile(name)
		if err != nil {
			if os.IsNotExist(err) || os.IsPermission(err) || err.Error() == "operation not supported" || err.Error() == "invalid argument" {
				continue
			}
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		vp := util.NewValueParser(value)

		switch f.Name() {
		case "port_multicast_rcv_packets":
			counters.LegacyPortMulticastRcvPackets = vp.PUInt64()
		case "port_multicast_xmit_packets":
			counters.LegacyPortMulticastXmitPackets = vp.PUInt64()
		case "port_rcv_data_64":
			counters.LegacyPortRcvData64 = vp.PUInt64()
			if counters.LegacyPortRcvData64 != nil {
				*counters.LegacyPortRcvData64 *= 4
			}
		case "port_rcv_packets_64":
			counters.LegacyPortRcvPackets64 = vp.PUInt64()
		case "port_unicast_rcv_packets":
			counters.LegacyPortUnicastRcvPackets = vp.PUInt64()
		case "port_unicast_xmit_packets":
			counters.LegacyPortUnicastXmitPackets = vp.PUInt64()
		case "port_xmit_data_64":
			counters.LegacyPortXmitData64 = vp.PUInt64()
			if counters.LegacyPortXmitData64 != nil {
				*counters.LegacyPortXmitData64 *= 4
			}
		case "port_xmit_packets_64":
			counters.LegacyPortXmitPackets64 = vp.PUInt64()
		}

		if err := vp.Err(); err != nil {
			// Ugly workaround for handling https://github.com/prometheus/node_exporter/issues/966
			// when counters are `N/A (not available)`.
			// This was already patched and submitted, see
			// https://www.spinics.net/lists/linux-rdma/msg68596.html
			// Remove this as soon as the fix lands in the enterprise distros.
			if strings.Contains(value, "N/A (no PMA)") {
				continue
			}
			return nil, err
		}
	}

	return &counters, nil
}
