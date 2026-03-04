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
	"bufio"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// NetDevSNMP6 is parsed from files in /proc/net/dev_snmp6/ or /proc/<PID>/net/dev_snmp6/.
// The outer map's keys are interface names and the inner map's keys are stat names.
//
// If you'd like a total across all interfaces, please use the Snmp6() method of the Proc type.
type NetDevSNMP6 map[string]map[string]uint64

// Returns kernel/system statistics read from interface files within the /proc/net/dev_snmp6/
// directory.
func (fs FS) NetDevSNMP6() (NetDevSNMP6, error) {
	return newNetDevSNMP6(fs.proc.Path("net/dev_snmp6"))
}

// Returns kernel/system statistics read from interface files within the /proc/<PID>/net/dev_snmp6/
// directory.
func (p Proc) NetDevSNMP6() (NetDevSNMP6, error) {
	return newNetDevSNMP6(p.path("net/dev_snmp6"))
}

// newNetDevSNMP6 creates a new NetDevSNMP6 from the contents of the given directory.
func newNetDevSNMP6(dir string) (NetDevSNMP6, error) {
	netDevSNMP6 := make(NetDevSNMP6)

	// The net/dev_snmp6 folders contain one file per interface
	ifaceFiles, err := os.ReadDir(dir)
	if err != nil {
		// On systems with IPv6 disabled, this directory won't exist.
		// Do nothing.
		if errors.Is(err, os.ErrNotExist) {
			return netDevSNMP6, err
		}
		return netDevSNMP6, err
	}

	for _, iFaceFile := range ifaceFiles {
		filePath := filepath.Join(dir, iFaceFile.Name())

		f, err := os.Open(filePath)
		if err != nil {
			return netDevSNMP6, err
		}
		defer f.Close()

		netDevSNMP6[iFaceFile.Name()], err = parseNetDevSNMP6Stats(f)
		if err != nil {
			return netDevSNMP6, err
		}
	}

	return netDevSNMP6, nil
}

func parseNetDevSNMP6Stats(r io.Reader) (map[string]uint64, error) {
	m := make(map[string]uint64)

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		stat := strings.Fields(scanner.Text())
		if len(stat) < 2 {
			continue
		}
		key, val := stat[0], stat[1]

		// Expect stat name to contain "6" or be "ifIndex"
		if strings.Contains(key, "6") || key == "ifIndex" {
			v, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return m, err
			}

			m[key] = v
		}
	}
	return m, scanner.Err()
}
