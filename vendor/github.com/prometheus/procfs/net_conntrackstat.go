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
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// A ConntrackStatEntry represents one line from net/stat/nf_conntrack
// and contains netfilter conntrack statistics at one CPU core.
type ConntrackStatEntry struct {
	Entries       uint64
	Found         uint64
	Invalid       uint64
	Ignore        uint64
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
		return nil, fmt.Errorf("failed to read conntrack stats from %q: %w", path, err)
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
	if len(fields) != 17 {
		return nil, fmt.Errorf("invalid conntrackstat entry, missing fields")
	}
	entry := &ConntrackStatEntry{}

	entries, err := parseConntrackStatField(fields[0])
	if err != nil {
		return nil, err
	}
	entry.Entries = entries

	found, err := parseConntrackStatField(fields[2])
	if err != nil {
		return nil, err
	}
	entry.Found = found

	invalid, err := parseConntrackStatField(fields[4])
	if err != nil {
		return nil, err
	}
	entry.Invalid = invalid

	ignore, err := parseConntrackStatField(fields[5])
	if err != nil {
		return nil, err
	}
	entry.Ignore = ignore

	insert, err := parseConntrackStatField(fields[8])
	if err != nil {
		return nil, err
	}
	entry.Insert = insert

	insertFailed, err := parseConntrackStatField(fields[9])
	if err != nil {
		return nil, err
	}
	entry.InsertFailed = insertFailed

	drop, err := parseConntrackStatField(fields[10])
	if err != nil {
		return nil, err
	}
	entry.Drop = drop

	earlyDrop, err := parseConntrackStatField(fields[11])
	if err != nil {
		return nil, err
	}
	entry.EarlyDrop = earlyDrop

	searchRestart, err := parseConntrackStatField(fields[16])
	if err != nil {
		return nil, err
	}
	entry.SearchRestart = searchRestart

	return entry, nil
}

// Parses a uint64 from given hex in string.
func parseConntrackStatField(field string) (uint64, error) {
	val, err := strconv.ParseUint(field, 16, 64)
	if err != nil {
		return 0, fmt.Errorf("couldn't parse %q field: %w", field, err)
	}
	return val, err
}
