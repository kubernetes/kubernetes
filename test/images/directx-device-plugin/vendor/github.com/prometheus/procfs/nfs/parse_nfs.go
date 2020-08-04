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

// ParseClientRPCStats returns stats read from /proc/net/rpc/nfs
func ParseClientRPCStats(r io.Reader) (*ClientRPCStats, error) {
	stats := &ClientRPCStats{}

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(scanner.Text())
		// require at least <key> <value>
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid NFS metric line %q", line)
		}

		values, err := util.ParseUint64s(parts[1:])
		if err != nil {
			return nil, fmt.Errorf("error parsing NFS metric line: %s", err)
		}

		switch metricLine := parts[0]; metricLine {
		case "net":
			stats.Network, err = parseNetwork(values)
		case "rpc":
			stats.ClientRPC, err = parseClientRPC(values)
		case "proc2":
			stats.V2Stats, err = parseV2Stats(values)
		case "proc3":
			stats.V3Stats, err = parseV3Stats(values)
		case "proc4":
			stats.ClientV4Stats, err = parseClientV4Stats(values)
		default:
			return nil, fmt.Errorf("unknown NFS metric line %q", metricLine)
		}
		if err != nil {
			return nil, fmt.Errorf("errors parsing NFS metric line: %s", err)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning NFS file: %s", err)
	}

	return stats, nil
}
