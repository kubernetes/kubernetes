// Copyright The Prometheus Authors
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
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
)

// Learned from include/uapi/linux/if_arp.h.
const (
	// Completed entry (ha valid).
	ATFComplete = 0x02
	// Permanent entry.
	ATFPermanent = 0x04
	// Publish entry.
	ATFPublish = 0x08
	// Has requested trailers.
	ATFUseTrailers = 0x10
	// Obsoleted: Want to use a netmask (only for proxy entries).
	ATFNetmask = 0x20
	// Don't answer this addresses.
	ATFDontPublish = 0x40
)

// ARPEntry contains a single row of the columnar data represented in
// /proc/net/arp.
type ARPEntry struct {
	// IP address
	IPAddr net.IP
	// MAC address
	HWAddr net.HardwareAddr
	// Name of the device
	Device string
	// Flags
	Flags byte
}

// GatherARPEntries retrieves all the ARP entries, parse the relevant columns,
// and then return a slice of ARPEntry's.
func (fs FS) GatherARPEntries() ([]ARPEntry, error) {
	data, err := os.ReadFile(fs.proc.Path("net/arp"))
	if err != nil {
		return nil, fmt.Errorf("%w: error reading arp %s: %w", ErrFileRead, fs.proc.Path("net/arp"), err)
	}

	return parseARPEntries(data)
}

func parseARPEntries(data []byte) ([]ARPEntry, error) {
	lines := strings.Split(string(data), "\n")
	entries := make([]ARPEntry, 0)
	var err error
	const (
		expectedDataWidth   = 6
		expectedHeaderWidth = 9
	)
	for _, line := range lines {
		columns := strings.Fields(line)
		width := len(columns)

		switch width {
		case expectedHeaderWidth, 0:
			continue
		case expectedDataWidth:
			entry, err := parseARPEntry(columns)
			if err != nil {
				return []ARPEntry{}, fmt.Errorf("%w: Failed to parse ARP entry: %v: %w", ErrFileParse, entry, err)
			}
			entries = append(entries, entry)
		default:
			return []ARPEntry{}, fmt.Errorf("%w: %d columns found, but expected %d: %w", ErrFileParse, width, expectedDataWidth, err)
		}

	}

	return entries, err
}

func parseARPEntry(columns []string) (ARPEntry, error) {
	entry := ARPEntry{Device: columns[5]}
	ip := net.ParseIP(columns[0])
	entry.IPAddr = ip

	if mac, err := net.ParseMAC(columns[3]); err == nil {
		entry.HWAddr = mac
	} else {
		return ARPEntry{}, err
	}

	if flags, err := strconv.ParseUint(columns[2], 0, 8); err == nil {
		entry.Flags = byte(flags)
	} else {
		return ARPEntry{}, err
	}

	return entry, nil
}

// IsComplete returns true if ARP entry is marked with complete flag.
func (entry *ARPEntry) IsComplete() bool {
	return entry.Flags&ATFComplete != 0
}
