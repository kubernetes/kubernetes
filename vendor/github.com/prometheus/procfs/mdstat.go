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

// ParseMDStat parses an mdstat-file and returns a struct with the relevant infos.
func (fs FS) ParseMDStat() (mdstates []MDStat, err error) {
	mdStatusFilePath := fs.Path("mdstat")
	content, err := ioutil.ReadFile(mdStatusFilePath)
	if err != nil {
		return []MDStat{}, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
	}

	mdStates := []MDStat{}
	lines := strings.Split(string(content), "\n")
	for i, l := range lines {
		if l == "" {
			continue
		}
		if l[0] == ' ' {
			continue
		}
		if strings.HasPrefix(l, "Personalities") || strings.HasPrefix(l, "unused") {
			continue
		}

		mainLine := strings.Split(l, " ")
		if len(mainLine) < 3 {
			return mdStates, fmt.Errorf("error parsing mdline: %s", l)
		}
		mdName := mainLine[0]
		activityState := mainLine[2]

		if len(lines) <= i+3 {
			return mdStates, fmt.Errorf(
				"error parsing %s: too few lines for md device %s",
				mdStatusFilePath,
				mdName,
			)
		}

		active, total, size, err := evalStatusline(lines[i+1])
		if err != nil {
			return mdStates, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
		}

		// j is the line number of the syncing-line.
		j := i + 2
		if strings.Contains(lines[i+2], "bitmap") { // skip bitmap line
			j = i + 3
		}

		// If device is syncing at the moment, get the number of currently
		// synced bytes, otherwise that number equals the size of the device.
		syncedBlocks := size
		if strings.Contains(lines[j], "recovery") || strings.Contains(lines[j], "resync") {
			syncedBlocks, err = evalBuildline(lines[j])
			if err != nil {
				return mdStates, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
			}
		}

		mdStates = append(mdStates, MDStat{
			Name:          mdName,
			ActivityState: activityState,
			DisksActive:   active,
			DisksTotal:    total,
			BlocksTotal:   size,
			BlocksSynced:  syncedBlocks,
		})
	}

	return mdStates, nil
}

func evalStatusline(statusline string) (active, total, size int64, err error) {
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

func evalBuildline(buildline string) (syncedBlocks int64, err error) {
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
