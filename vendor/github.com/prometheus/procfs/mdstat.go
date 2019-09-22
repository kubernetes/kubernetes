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

package procfs

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"
)

var (
	statuslineRE = regexp.MustCompile(`(\d+) blocks .*\[(\d+)/(\d+)\] \[[U_]+\]`)
	buildlineRE  = regexp.MustCompile(`\((\d+)/\d+\)`)
)

// MDStat holds info parsed from /proc/mdstat.
type MDStat struct {
	// Name of the device.
	Name string
	// activity-state of the device.
	ActivityState string
	// Number of active disks.
	DisksActive int64
	// Total number of disks the device consists of.
	DisksTotal int64
	// Number of blocks the device holds.
	BlocksTotal int64
	// Number of blocks on the device that are in sync.
	BlocksSynced int64
}

// MDStat parses an mdstat-file (/proc/mdstat) and returns a slice of
// structs containing the relevant info.  More information available here:
// https://raid.wiki.kernel.org/index.php/Mdstat
func (fs FS) MDStat() ([]MDStat, error) {
	data, err := ioutil.ReadFile(fs.proc.Path("mdstat"))
	if err != nil {
		return nil, fmt.Errorf("error parsing mdstat %s: %s", fs.proc.Path("mdstat"), err)
	}
	mdstat, err := parseMDStat(data)
	if err != nil {
		return nil, fmt.Errorf("error parsing mdstat %s: %s", fs.proc.Path("mdstat"), err)
	}
	return mdstat, nil
}

// parseMDStat parses data from mdstat file (/proc/mdstat) and returns a slice of
// structs containing the relevant info.
func parseMDStat(mdstatData []byte) ([]MDStat, error) {
	mdStats := []MDStat{}
	lines := strings.Split(string(mdstatData), "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) == "" || l[0] == ' ' ||
			strings.HasPrefix(l, "Personalities") || strings.HasPrefix(l, "unused") {
			continue
		}

		deviceFields := strings.Fields(l)
		if len(deviceFields) < 3 {
			return nil, fmt.Errorf("not enough fields in mdline (expected at least 3): %s", l)
		}
		mdName := deviceFields[0]
		activityState := deviceFields[2]

		if len(lines) <= i+3 {
			return mdStats, fmt.Errorf("missing lines for md device %s", mdName)
		}

		active, total, size, err := evalStatusLine(lines[i+1])
		if err != nil {
			return nil, err
		}

		syncLineIdx := i + 2
		if strings.Contains(lines[i+2], "bitmap") { // skip bitmap line
			syncLineIdx++
		}

		// If device is recovering/syncing at the moment, get the number of currently
		// synced bytes, otherwise that number equals the size of the device.
		syncedBlocks := size
		if strings.Contains(lines[syncLineIdx], "recovery") || strings.Contains(lines[syncLineIdx], "resync") {
			syncedBlocks, err = evalRecoveryLine(lines[syncLineIdx])
			if err != nil {
				return nil, err
			}
		}

		mdStats = append(mdStats, MDStat{
			Name:          mdName,
			ActivityState: activityState,
			DisksActive:   active,
			DisksTotal:    total,
			BlocksTotal:   size,
			BlocksSynced:  syncedBlocks,
		})
	}

	return mdStats, nil
}

func evalStatusLine(statusline string) (active, total, size int64, err error) {
	matches := statuslineRE.FindStringSubmatch(statusline)
	if len(matches) != 4 {
		return 0, 0, 0, fmt.Errorf("unexpected statusline: %s", statusline)
	}

	size, err = strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("unexpected statusline %s: %s", statusline, err)
	}

	total, err = strconv.ParseInt(matches[2], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("unexpected statusline %s: %s", statusline, err)
	}

	active, err = strconv.ParseInt(matches[3], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("unexpected statusline %s: %s", statusline, err)
	}

	return active, total, size, nil
}

func evalRecoveryLine(buildline string) (syncedBlocks int64, err error) {
	matches := buildlineRE.FindStringSubmatch(buildline)
	if len(matches) != 2 {
		return 0, fmt.Errorf("unexpected buildline: %s", buildline)
	}

	syncedBlocks, err = strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s in buildline: %s", err, buildline)
	}

	return syncedBlocks, nil
}
