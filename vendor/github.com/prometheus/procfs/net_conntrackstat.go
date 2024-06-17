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
	"io"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// A ConntrackStatEntry represents one line from net/stat/nf_conntrack
// and contains netfilter conntrack statistics at one CPU core.
type ConntrackStatEntry struct {
	Entries       uint64
	Searched      uint64
	Found         uint64
	New           uint64
	Invalid       uint64
	Ignore        uint64
	Delete        uint64
	DeleteList    uint64
	Insert        uint64
	InsertFailed  uint64
	Drop          uint64
	EarlyDrop     uint64
	SearchRestart uint64
}

// ConntrackStat retrieves netfilter's conntrack statistics, split by CPU cores.
func (fs FS) ConntrackStat() ([]ConntrackStatEntry, error) {
	return readConntrackStat(fs.proc.Path("net", "stat", "nf_conntrack"))
}

// Parses a slice of ConntrackStatEntries from the given filepath.
func readConntrackStat(path string) ([]ConntrackStatEntry, error) {
	// This file is small and can be read with one syscall.
	b, err := util.ReadFileNoStat(path)
	if err != nil {
		// Do not wrap this error so the caller can detect os.IsNotExist and
		// similar conditions.
		return nil, err
	}

	stat, err := parseConntrackStat(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("%s: Cannot read file: %v: %w", ErrFileRead, path, err)
	}

	return stat, nil
}

// Reads the contents of a conntrack statistics file and parses a slice of ConntrackStatEntries.
func parseConntrackStat(r io.Reader) ([]ConntrackStatEntry, error) {
	var entries []ConntrackStatEntry

	scanner := bufio.NewScanner(r)
	scanner.Scan()
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		conntrackEntry, err := parseConntrackStatEntry(fields)
		if err != nil {
			return nil, err
		}
		entries = append(entries, *conntrackEntry)
	}

	return entries, nil
}

// Parses a ConntrackStatEntry from given array of fields.
func parseConntrackStatEntry(fields []string) (*ConntrackStatEntry, error) {
	entries, err := util.ParseHexUint64s(fields)
	if err != nil {
		return nil, fmt.Errorf("%s: Cannot parse entry: %d: %w", ErrFileParse, entries, err)
	}
	numEntries := len(entries)
	if numEntries < 16 || numEntries > 17 {
		return nil,
			fmt.Errorf("%w: invalid conntrackstat entry, invalid number of fields: %d", ErrFileParse, numEntries)
	}

	stats := &ConntrackStatEntry{
		Entries:      *entries[0],
		Searched:     *entries[1],
		Found:        *entries[2],
		New:          *entries[3],
		Invalid:      *entries[4],
		Ignore:       *entries[5],
		Delete:       *entries[6],
		DeleteList:   *entries[7],
		Insert:       *entries[8],
		InsertFailed: *entries[9],
		Drop:         *entries[10],
		EarlyDrop:    *entries[11],
	}

	// Ignore missing search_restart on Linux < 2.6.35.
	if numEntries == 17 {
		stats.SearchRestart = *entries[16]
	}

	return stats, nil
}
