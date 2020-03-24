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
	"strconv"
	"strings"
)

// For the proc file format details,
// see https://elixir.bootlin.com/linux/v4.17/source/net/core/net-procfs.c#L162
// and https://elixir.bootlin.com/linux/v4.17/source/include/linux/netdevice.h#L2810.

// SoftnetEntry contains a single row of data from /proc/net/softnet_stat
type SoftnetEntry struct {
	// Number of processed packets
	Processed uint
	// Number of dropped packets
	Dropped uint
	// Number of times processing packets ran out of quota
	TimeSqueezed uint
}

// GatherSoftnetStats reads /proc/net/softnet_stat, parse the relevant columns,
// and then return a slice of SoftnetEntry's.
func (fs FS) GatherSoftnetStats() ([]SoftnetEntry, error) {
	data, err := ioutil.ReadFile(fs.proc.Path("net/softnet_stat"))
	if err != nil {
		return nil, fmt.Errorf("error reading softnet %s: %s", fs.proc.Path("net/softnet_stat"), err)
	}

	return parseSoftnetEntries(data)
}

func parseSoftnetEntries(data []byte) ([]SoftnetEntry, error) {
	lines := strings.Split(string(data), "\n")
	entries := make([]SoftnetEntry, 0)
	var err error
	const (
		expectedColumns = 11
	)
	for _, line := range lines {
		columns := strings.Fields(line)
		width := len(columns)
		if width == 0 {
			continue
		}
		if width != expectedColumns {
			return []SoftnetEntry{}, fmt.Errorf("%d columns were detected, but %d were expected", width, expectedColumns)
		}
		var entry SoftnetEntry
		if entry, err = parseSoftnetEntry(columns); err != nil {
			return []SoftnetEntry{}, err
		}
		entries = append(entries, entry)
	}

	return entries, nil
}

func parseSoftnetEntry(columns []string) (SoftnetEntry, error) {
	var err error
	var processed, dropped, timeSqueezed uint64
	if processed, err = strconv.ParseUint(columns[0], 16, 32); err != nil {
		return SoftnetEntry{}, fmt.Errorf("Unable to parse column 0: %s", err)
	}
	if dropped, err = strconv.ParseUint(columns[1], 16, 32); err != nil {
		return SoftnetEntry{}, fmt.Errorf("Unable to parse column 1: %s", err)
	}
	if timeSqueezed, err = strconv.ParseUint(columns[2], 16, 32); err != nil {
		return SoftnetEntry{}, fmt.Errorf("Unable to parse column 2: %s", err)
	}
	return SoftnetEntry{
		Processed:    uint(processed),
		Dropped:      uint(dropped),
		TimeSqueezed: uint(timeSqueezed),
	}, nil
}
