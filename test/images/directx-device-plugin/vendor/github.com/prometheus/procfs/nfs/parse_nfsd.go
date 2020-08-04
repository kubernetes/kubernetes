// Copyright 2018 The Prometheus Authors
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

package nfs

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// ParseServerRPCStats returns stats read from /proc/net/rpc/nfsd
func ParseServerRPCStats(r io.Reader) (*ServerRPCStats, error) {
	stats := &ServerRPCStats{}

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(scanner.Text())
		// require at least <key> <value>
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid NFSd metric line %q", line)
		}
		label := parts[0]

		var values []uint64
		var err error
		if label == "th" {
			if len(parts) < 3 {
				return nil, fmt.Errorf("invalid NFSd th metric line %q", line)
			}
			values, err = util.ParseUint64s(parts[1:3])
		} else {
			values, err = util.ParseUint64s(parts[1:])
		}
		if err != nil {
			return nil, fmt.Errorf("error parsing NFSd metric line: %s", err)
		}

		switch metricLine := parts[0]; metricLine {
		case "rc":
			stats.ReplyCache, err = parseReplyCache(values)
		case "fh":
			stats.FileHandles, err = parseFileHandles(values)
		case "io":
			stats.InputOutput, err = parseInputOutput(values)
		case "th":
			stats.Threads, err = parseThreads(values)
		case "ra":
			stats.ReadAheadCache, err = parseReadAheadCache(values)
		case "net":
			stats.Network, err = parseNetwork(values)
		case "rpc":
			stats.ServerRPC, err = parseServerRPC(values)
		case "proc2":
			stats.V2Stats, err = parseV2Stats(values)
		case "proc3":
			stats.V3Stats, err = parseV3Stats(values)
		case "proc4":
			stats.ServerV4Stats, err = parseServerV4Stats(values)
		case "proc4ops":
			stats.V4Ops, err = parseV4Ops(values)
		default:
			return nil, fmt.Errorf("unknown NFSd metric line %q", metricLine)
		}
		if err != nil {
			return nil, fmt.Errorf("errors parsing NFSd metric line: %s", err)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning NFSd file: %s", err)
	}

	return stats, nil
}
