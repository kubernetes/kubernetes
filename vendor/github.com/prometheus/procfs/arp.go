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

package procfs

import (
	"fmt"
	"io/ioutil"
	"net"
	"strings"
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
}

// GatherARPEntries retrieves all the ARP entries, parse the relevant columns,
// and then return a slice of ARPEntry's.
func (fs FS) GatherARPEntries() ([]ARPEntry, error) {
	data, err := ioutil.ReadFile(fs.proc.Path("net/arp"))
	if err != nil {
		return nil, fmt.Errorf("error reading arp %q: %w", fs.proc.Path("net/arp"), err)
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

		if width == expectedHeaderWidth || width == 0 {
			continue
		} else if width == expectedDataWidth {
			entry, err := parseARPEntry(columns)
			if err != nil {
				return []ARPEntry{}, fmt.Errorf("failed to parse ARP entry: %w", err)
			}
			entries = append(entries, entry)
		} else {
			return []ARPEntry{}, fmt.Errorf("%d columns were detected, but %d were expected", width, expectedDataWidth)
		}

	}

	return entries, err
}

func parseARPEntry(columns []string) (ARPEntry, error) {
	ip := net.ParseIP(columns[0])
	mac := net.HardwareAddr(columns[3])

	entry := ARPEntry{
		IPAddr: ip,
		HWAddr: mac,
		Device: columns[5],
	}

	return entry, nil
}
