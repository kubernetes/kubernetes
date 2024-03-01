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
	"os"
	"regexp"
	"strconv"
	"strings"
)

var (
	statusLineRE         = regexp.MustCompile(`(\d+) blocks .*\[(\d+)/(\d+)\] \[([U_]+)\]`)
	recoveryLineBlocksRE = regexp.MustCompile(`\((\d+)/\d+\)`)
	recoveryLinePctRE    = regexp.MustCompile(`= (.+)%`)
	recoveryLineFinishRE = regexp.MustCompile(`finish=(.+)min`)
	recoveryLineSpeedRE  = regexp.MustCompile(`speed=(.+)[A-Z]`)
	componentDeviceRE    = regexp.MustCompile(`(.*)\[\d+\]`)
)

// MDStat holds info parsed from /proc/mdstat.
type MDStat struct {
	// Name of the device.
	Name string
	// activity-state of the device.
	ActivityState string
	// Number of active disks.
	DisksActive int64
	// Total number of disks the device requires.
	DisksTotal int64
	// Number of failed disks.
	DisksFailed int64
	// Number of "down" disks. (the _ indicator in the status line)
	DisksDown int64
	// Spare disks in the device.
	DisksSpare int64
	// Number of blocks the device holds.
	BlocksTotal int64
	// Number of blocks on the device that are in sync.
	BlocksSynced int64
	// progress percentage of current sync
	BlocksSyncedPct float64
	// estimated finishing time for current sync (in minutes)
	BlocksSyncedFinishTime float64
	// current sync speed (in Kilobytes/sec)
	BlocksSyncedSpeed float64
	// Name of md component devices
	Devices []string
}

// MDStat parses an mdstat-file (/proc/mdstat) and returns a slice of
// structs containing the relevant info.  More information available here:
// https://raid.wiki.kernel.org/index.php/Mdstat
func (fs FS) MDStat() ([]MDStat, error) {
	data, err := os.ReadFile(fs.proc.Path("mdstat"))
	if err != nil {
		return nil, err
	}
	mdstat, err := parseMDStat(data)
	if err != nil {
		return nil, fmt.Errorf("%s: Cannot parse %v: %w", ErrFileParse, fs.proc.Path("mdstat"), err)
	}
	return mdstat, nil
}

// parseMDStat parses data from mdstat file (/proc/mdstat) and returns a slice of
// structs containing the relevant info.
func parseMDStat(mdStatData []byte) ([]MDStat, error) {
	mdStats := []MDStat{}
	lines := strings.Split(string(mdStatData), "\n")

	for i, line := range lines {
		if strings.TrimSpace(line) == "" || line[0] == ' ' ||
			strings.HasPrefix(line, "Personalities") ||
			strings.HasPrefix(line, "unused") {
			continue
		}

		deviceFields := strings.Fields(line)
		if len(deviceFields) < 3 {
			return nil, fmt.Errorf("%s: Expected 3+ lines, got %q", ErrFileParse, line)
		}
		mdName := deviceFields[0] // mdx
		state := deviceFields[2]  // active or inactive

		if len(lines) <= i+3 {
			return nil, fmt.Errorf("%w: Too few lines for md device: %q", ErrFileParse, mdName)
		}

		// Failed disks have the suffix (F) & Spare disks have the suffix (S).
		fail := int64(strings.Count(line, "(F)"))
		spare := int64(strings.Count(line, "(S)"))
		active, total, down, size, err := evalStatusLine(lines[i], lines[i+1])

		if err != nil {
			return nil, fmt.Errorf("%s: Cannot parse md device lines: %v: %w", ErrFileParse, active, err)
		}

		syncLineIdx := i + 2
		if strings.Contains(lines[i+2], "bitmap") { // skip bitmap line
			syncLineIdx++
		}

		// If device is syncing at the moment, get the number of currently
		// synced bytes, otherwise that number equals the size of the device.
		syncedBlocks := size
		speed := float64(0)
		finish := float64(0)
		pct := float64(0)
		recovering := strings.Contains(lines[syncLineIdx], "recovery")
		resyncing := strings.Contains(lines[syncLineIdx], "resync")
		checking := strings.Contains(lines[syncLineIdx], "check")

		// Append recovery and resyncing state info.
		if recovering || resyncing || checking {
			if recovering {
				state = "recovering"
			} else if checking {
				state = "checking"
			} else {
				state = "resyncing"
			}

			// Handle case when resync=PENDING or resync=DELAYED.
			if strings.Contains(lines[syncLineIdx], "PENDING") ||
				strings.Contains(lines[syncLineIdx], "DELAYED") {
				syncedBlocks = 0
			} else {
				syncedBlocks, pct, finish, speed, err = evalRecoveryLine(lines[syncLineIdx])
				if err != nil {
					return nil, fmt.Errorf("%s: Cannot parse sync line in md device: %q: %w", ErrFileParse, mdName, err)
				}
			}
		}

		mdStats = append(mdStats, MDStat{
			Name:                   mdName,
			ActivityState:          state,
			DisksActive:            active,
			DisksFailed:            fail,
			DisksDown:              down,
			DisksSpare:             spare,
			DisksTotal:             total,
			BlocksTotal:            size,
			BlocksSynced:           syncedBlocks,
			BlocksSyncedPct:        pct,
			BlocksSyncedFinishTime: finish,
			BlocksSyncedSpeed:      speed,
			Devices:                evalComponentDevices(deviceFields),
		})
	}

	return mdStats, nil
}

func evalStatusLine(deviceLine, statusLine string) (active, total, down, size int64, err error) {
	statusFields := strings.Fields(statusLine)
	if len(statusFields) < 1 {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
	}

	sizeStr := statusFields[0]
	size, err = strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
	}

	if strings.Contains(deviceLine, "raid0") || strings.Contains(deviceLine, "linear") {
		// In the device deviceLine, only disks have a number associated with them in [].
		total = int64(strings.Count(deviceLine, "["))
		return total, total, 0, size, nil
	}

	if strings.Contains(deviceLine, "inactive") {
		return 0, 0, 0, size, nil
	}

	matches := statusLineRE.FindStringSubmatch(statusLine)
	if len(matches) != 5 {
		return 0, 0, 0, 0, fmt.Errorf("%s: Could not fild all substring matches %s: %w", ErrFileParse, statusLine, err)
	}

	total, err = strconv.ParseInt(matches[2], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
	}

	active, err = strconv.ParseInt(matches[3], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected active %d: %w", ErrFileParse, active, err)
	}
	down = int64(strings.Count(matches[4], "_"))

	return active, total, down, size, nil
}

func evalRecoveryLine(recoveryLine string) (syncedBlocks int64, pct float64, finish float64, speed float64, err error) {
	matches := recoveryLineBlocksRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected recoveryLine %s: %w", ErrFileParse, recoveryLine, err)
	}

	syncedBlocks, err = strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%s: Unexpected parsing of recoveryLine %q: %w", ErrFileParse, recoveryLine, err)
	}

	// Get percentage complete
	matches = recoveryLinePctRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return syncedBlocks, 0, 0, 0, fmt.Errorf("%w: Unexpected recoveryLine matching percentage %s", ErrFileParse, recoveryLine)
	}
	pct, err = strconv.ParseFloat(strings.TrimSpace(matches[1]), 64)
	if err != nil {
		return syncedBlocks, 0, 0, 0, fmt.Errorf("%w: Error parsing float from recoveryLine %q", ErrFileParse, recoveryLine)
	}

	// Get time expected left to complete
	matches = recoveryLineFinishRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return syncedBlocks, pct, 0, 0, fmt.Errorf("%w: Unexpected recoveryLine matching est. finish time: %s", ErrFileParse, recoveryLine)
	}
	finish, err = strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return syncedBlocks, pct, 0, 0, fmt.Errorf("%w: Unable to parse float from recoveryLine: %q", ErrFileParse, recoveryLine)
	}

	// Get recovery speed
	matches = recoveryLineSpeedRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return syncedBlocks, pct, finish, 0, fmt.Errorf("%w: Unexpected recoveryLine value: %s", ErrFileParse, recoveryLine)
	}
	speed, err = strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return syncedBlocks, pct, finish, 0, fmt.Errorf("%s: Error parsing float from recoveryLine: %q: %w", ErrFileParse, recoveryLine, err)
	}

	return syncedBlocks, pct, finish, speed, nil
}

func evalComponentDevices(deviceFields []string) []string {
	mdComponentDevices := make([]string, 0)
	if len(deviceFields) > 3 {
		for _, field := range deviceFields[4:] {
			match := componentDeviceRE.FindStringSubmatch(field)
			if match == nil {
				continue
			}
			mdComponentDevices = append(mdComponentDevices, match[1])
		}
	}

	return mdComponentDevices
}
