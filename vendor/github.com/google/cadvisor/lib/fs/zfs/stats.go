// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package zfs

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// GetZfsStats returns (capacity, free, available) byte counts for the given ZFS
// dataset/pool. It shells out to `zfs list` -- the same mechanism the
// mistifyio/go-zfs library used internally -- reduced to the three properties
// cAdvisor consumes, so that heavyweight dependency is no longer required.
func GetZfsStats(poolName string) (uint64, uint64, uint64, error) {
	// `-Hp`: scripted output (no header), parsable exact byte values.
	out, err := exec.Command("zfs", "list", "-Hp", "-o", "used,available,usedbydataset", poolName).Output()
	if err != nil {
		return 0, 0, 0, err
	}
	return parseZfsListUsage(out)
}

// parseZfsListUsage parses one line of
// `zfs list -Hp -o used,available,usedbydataset` output (tab-separated). A "-"
// value is treated as 0 (matching go-zfs's setUint). It returns
// (capacity, free, available) preserving cAdvisor's original arithmetic.
func parseZfsListUsage(out []byte) (uint64, uint64, uint64, error) {
	fields := strings.Fields(string(out))
	if len(fields) != 3 {
		return 0, 0, 0, fmt.Errorf("unexpected `zfs list` output, want 3 fields, got %q", out)
	}
	used, err := parseZfsUint(fields[0])
	if err != nil {
		return 0, 0, 0, err
	}
	avail, err := parseZfsUint(fields[1])
	if err != nil {
		return 0, 0, 0, err
	}
	usedByDataset, err := parseZfsUint(fields[2])
	if err != nil {
		return 0, 0, 0, err
	}
	return used + avail + usedByDataset, avail, avail, nil
}

func parseZfsUint(s string) (uint64, error) {
	if s == "-" {
		return 0, nil
	}
	return strconv.ParseUint(s, 10, 64)
}
