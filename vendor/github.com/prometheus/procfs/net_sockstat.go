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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// A NetSockstat contains the output of /proc/net/sockstat{,6} for IPv4 or IPv6,
// respectively.
type NetSockstat struct {
	// Used is non-nil for IPv4 sockstat results, but nil for IPv6.
	Used      *int
	Protocols []NetSockstatProtocol
}

// A NetSockstatProtocol contains statistics about a given socket protocol.
// Pointer fields indicate that the value may or may not be present on any
// given protocol.
type NetSockstatProtocol struct {
	Protocol string
	InUse    int
	Orphan   *int
	TW       *int
	Alloc    *int
	Mem      *int
	Memory   *int
}

// NetSockstat retrieves IPv4 socket statistics.
func (fs FS) NetSockstat() (*NetSockstat, error) {
	return readSockstat(fs.proc.Path("net", "sockstat"))
}

// NetSockstat6 retrieves IPv6 socket statistics.
//
// If IPv6 is disabled on this kernel, the returned error can be checked with
// os.IsNotExist.
func (fs FS) NetSockstat6() (*NetSockstat, error) {
	return readSockstat(fs.proc.Path("net", "sockstat6"))
}

// readSockstat opens and parses a NetSockstat from the input file.
func readSockstat(name string) (*NetSockstat, error) {
	// This file is small and can be read with one syscall.
	b, err := util.ReadFileNoStat(name)
	if err != nil {
		// Do not wrap this error so the caller can detect os.IsNotExist and
		// similar conditions.
		return nil, err
	}

	stat, err := parseSockstat(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("%s: sockstats from %q: %w", ErrFileRead, name, err)
	}

	return stat, nil
}

// parseSockstat reads the contents of a sockstat file and parses a NetSockstat.
func parseSockstat(r io.Reader) (*NetSockstat, error) {
	var stat NetSockstat
	s := bufio.NewScanner(r)
	for s.Scan() {
		// Expect a minimum of a protocol and one key/value pair.
		fields := strings.Split(s.Text(), " ")
		if len(fields) < 3 {
			return nil, fmt.Errorf("%w: Malformed sockstat line: %q", ErrFileParse, s.Text())
		}

		// The remaining fields are key/value pairs.
		kvs, err := parseSockstatKVs(fields[1:])
		if err != nil {
			return nil, fmt.Errorf("%s: sockstat key/value pairs from %q: %w", ErrFileParse, s.Text(), err)
		}

		// The first field is the protocol. We must trim its colon suffix.
		proto := strings.TrimSuffix(fields[0], ":")
		switch proto {
		case "sockets":
			// Special case: IPv4 has a sockets "used" key/value pair that we
			// embed at the top level of the structure.
			used := kvs["used"]
			stat.Used = &used
		default:
			// Parse all other lines as individual protocols.
			nsp := parseSockstatProtocol(kvs)
			nsp.Protocol = proto
			stat.Protocols = append(stat.Protocols, nsp)
		}
	}

	if err := s.Err(); err != nil {
		return nil, err
	}

	return &stat, nil
}

// parseSockstatKVs parses a string slice into a map of key/value pairs.
func parseSockstatKVs(kvs []string) (map[string]int, error) {
	if len(kvs)%2 != 0 {
		return nil, fmt.Errorf("%w:: Odd number of fields in key/value pairs %q", ErrFileParse, kvs)
	}

	// Iterate two values at a time to gather key/value pairs.
	out := make(map[string]int, len(kvs)/2)
	for i := 0; i < len(kvs); i += 2 {
		vp := util.NewValueParser(kvs[i+1])
		out[kvs[i]] = vp.Int()

		if err := vp.Err(); err != nil {
			return nil, err
		}
	}

	return out, nil
}

// parseSockstatProtocol parses a NetSockstatProtocol from the input kvs map.
func parseSockstatProtocol(kvs map[string]int) NetSockstatProtocol {
	var nsp NetSockstatProtocol
	for k, v := range kvs {
		// Capture the range variable to ensure we get unique pointers for
		// each of the optional fields.
		v := v
		switch k {
		case "inuse":
			nsp.InUse = v
		case "orphan":
			nsp.Orphan = &v
		case "tw":
			nsp.TW = &v
		case "alloc":
			nsp.Alloc = &v
		case "mem":
			nsp.Mem = &v
		case "memory":
			nsp.Memory = &v
		}
	}

	return nsp
}
