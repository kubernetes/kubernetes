// Copyright 2020 The Prometheus Authors
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

package procfs

import (
	"bufio"
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// NetProtocolStats stores the contents from /proc/net/protocols
type NetProtocolStats map[string]NetProtocolStatLine

// NetProtocolStatLine contains a single line parsed from /proc/net/protocols. We
// only care about the first six columns as the rest are not likely to change
// and only serve to provide a set of capabilities for each protocol.
type NetProtocolStatLine struct {
	Name         string // 0 The name of the protocol
	Size         uint64 // 1 The size, in bytes, of a given protocol structure. e.g. sizeof(struct tcp_sock) or sizeof(struct unix_sock)
	Sockets      int64  // 2 Number of sockets in use by this protocol
	Memory       int64  // 3 Number of 4KB pages allocated by all sockets of this protocol
	Pressure     int    // 4 This is either yes, no, or NI (not implemented). For the sake of simplicity we treat NI as not experiencing memory pressure.
	MaxHeader    uint64 // 5 Protocol specific max header size
	Slab         bool   // 6 Indicates whether or not memory is allocated from the SLAB
	ModuleName   string // 7 The name of the module that implemented this protocol or "kernel" if not from a module
	Capabilities NetProtocolCapabilities
}

// NetProtocolCapabilities contains a list of capabilities for each protocol
type NetProtocolCapabilities struct {
	Close               bool // 8
	Connect             bool // 9
	Disconnect          bool // 10
	Accept              bool // 11
	IoCtl               bool // 12
	Init                bool // 13
	Destroy             bool // 14
	Shutdown            bool // 15
	SetSockOpt          bool // 16
	GetSockOpt          bool // 17
	SendMsg             bool // 18
	RecvMsg             bool // 19
	SendPage            bool // 20
	Bind                bool // 21
	BacklogRcv          bool // 22
	Hash                bool // 23
	UnHash              bool // 24
	GetPort             bool // 25
	EnterMemoryPressure bool // 26
}

// NetProtocols reads stats from /proc/net/protocols and returns a map of
// PortocolStatLine entries. As of this writing no official Linux Documentation
// exists, however the source is fairly self-explanatory and the format seems
// stable since its introduction in 2.6.12-rc2
// Linux 2.6.12-rc2 - https://elixir.bootlin.com/linux/v2.6.12-rc2/source/net/core/sock.c#L1452
// Linux 5.10 - https://elixir.bootlin.com/linux/v5.10.4/source/net/core/sock.c#L3586
func (fs FS) NetProtocols() (NetProtocolStats, error) {
	data, err := util.ReadFileNoStat(fs.proc.Path("net/protocols"))
	if err != nil {
		return NetProtocolStats{}, err
	}
	return parseNetProtocols(bufio.NewScanner(bytes.NewReader(data)))
}

func parseNetProtocols(s *bufio.Scanner) (NetProtocolStats, error) {
	nps := NetProtocolStats{}

	// Skip the header line
	s.Scan()

	for s.Scan() {
		line, err := nps.parseLine(s.Text())
		if err != nil {
			return NetProtocolStats{}, err
		}

		nps[line.Name] = *line
	}
	return nps, nil
}

func (ps NetProtocolStats) parseLine(rawLine string) (*NetProtocolStatLine, error) {
	line := &NetProtocolStatLine{Capabilities: NetProtocolCapabilities{}}
	var err error
	const enabled = "yes"
	const disabled = "no"

	fields := strings.Fields(rawLine)
	line.Name = fields[0]
	line.Size, err = strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return nil, err
	}
	line.Sockets, err = strconv.ParseInt(fields[2], 10, 64)
	if err != nil {
		return nil, err
	}
	line.Memory, err = strconv.ParseInt(fields[3], 10, 64)
	if err != nil {
		return nil, err
	}
	if fields[4] == enabled {
		line.Pressure = 1
	} else if fields[4] == disabled {
		line.Pressure = 0
	} else {
		line.Pressure = -1
	}
	line.MaxHeader, err = strconv.ParseUint(fields[5], 10, 64)
	if err != nil {
		return nil, err
	}
	if fields[6] == enabled {
		line.Slab = true
	} else if fields[6] == disabled {
		line.Slab = false
	} else {
		return nil, fmt.Errorf("unable to parse capability for protocol: %s", line.Name)
	}
	line.ModuleName = fields[7]

	err = line.Capabilities.parseCapabilities(fields[8:])
	if err != nil {
		return nil, err
	}

	return line, nil
}

func (pc *NetProtocolCapabilities) parseCapabilities(capabilities []string) error {
	// The capabilities are all bools so we can loop over to map them
	capabilityFields := [...]*bool{
		&pc.Close,
		&pc.Connect,
		&pc.Disconnect,
		&pc.Accept,
		&pc.IoCtl,
		&pc.Init,
		&pc.Destroy,
		&pc.Shutdown,
		&pc.SetSockOpt,
		&pc.GetSockOpt,
		&pc.SendMsg,
		&pc.RecvMsg,
		&pc.SendPage,
		&pc.Bind,
		&pc.BacklogRcv,
		&pc.Hash,
		&pc.UnHash,
		&pc.GetPort,
		&pc.EnterMemoryPressure,
	}

	for i := 0; i < len(capabilities); i++ {
		if capabilities[i] == "y" {
			*capabilityFields[i] = true
		} else if capabilities[i] == "n" {
			*capabilityFields[i] = false
		} else {
			return fmt.Errorf("unable to parse capability block for protocol: position %d", i)
		}
	}
	return nil
}
