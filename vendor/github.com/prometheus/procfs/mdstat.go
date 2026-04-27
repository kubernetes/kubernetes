// Copyright The Prometheus Authors
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
	recoveryLineBlocksRE = regexp.MustCompile(`\((\d+/\d+)\)`)
	recoveryLinePctRE    = regexp.MustCompile(`= (.+)%`)
	recoveryLineFinishRE = regexp.MustCompile(`finish=(.+)min`)
	recoveryLineSpeedRE  = regexp.MustCompile(`speed=(.+)[A-Z]`)
	componentDeviceRE    = regexp.MustCompile(`(.*)\[(\d+)\](\([SF]+\))?`)
	personalitiesPrefix  = "Personalities : "
)

type MDStatComponent struct {
	// Name of the component device.
	Name string
	// DescriptorIndex number of component device, e.g. the order in the superblock.
	DescriptorIndex int32
	// Flags per Linux drivers/md/md.[ch] as of v6.12-rc1
	// Subset that are exposed in mdstat
	WriteMostly bool
	Journal     bool
	Faulty      bool // "Faulty" is what kernel source uses for "(F)"
	Spare       bool
	Replacement bool
	// Some additional flags that are NOT exposed in procfs today; they may
	// be available via sysfs.
	// In_sync, Bitmap_sync, Blocked, WriteErrorSeen, FaultRecorded,
	// BlockedBadBlocks, WantReplacement, Candidate, ...
}

// MDStat holds info parsed from /proc/mdstat.
type MDStat struct {
	// Name of the device.
	Name string
	// raid type of the device.
	Type string
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
	// Number of blocks on the device that need to be synced.
	BlocksToBeSynced int64
	// progress percentage of current sync
	BlocksSyncedPct float64
	// estimated finishing time for current sync (in minutes)
	BlocksSyncedFinishTime float64
	// current sync speed (in Kilobytes/sec)
	BlocksSyncedSpeed float64
	// component devices
	Devices []MDStatComponent
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
		return nil, fmt.Errorf("%w: Cannot parse %v: %w", ErrFileParse, fs.proc.Path("mdstat"), err)
	}
	return mdstat, nil
}

// parseMDStat parses data from mdstat file (/proc/mdstat) and returns a slice of
// structs containing the relevant info.
func parseMDStat(mdStatData []byte) ([]MDStat, error) {
	// TODO:
	// - parse global hotspares from the "unused devices" line.
	mdStats := []MDStat{}
	lines := strings.Split(string(mdStatData), "\n")
	knownRaidTypes := make(map[string]bool)

	for i, line := range lines {
		if strings.TrimSpace(line) == "" || line[0] == ' ' ||
			strings.HasPrefix(line, "unused") {
			continue
		}
		// Personalities : [linear] [multipath] [raid0] [raid1] [raid6] [raid5] [raid4] [raid10]
		if len(knownRaidTypes) == 0 && strings.HasPrefix(line, personalitiesPrefix) {
			personalities := strings.Fields(line[len(personalitiesPrefix):])
			for _, word := range personalities {
				word := word[1 : len(word)-1]
				knownRaidTypes[word] = true
			}
			continue
		}

		deviceFields := strings.Fields(line)
		if len(deviceFields) < 3 {
			return nil, fmt.Errorf("%w: Expected 3+ lines, got %q", ErrFileParse, line)
		}
		mdName := deviceFields[0] // mdx
		state := deviceFields[2]  // active, inactive, broken

		mdType := "unknown" // raid1, raid5, etc.
		var deviceStartIndex int
		if len(deviceFields) > 3 { // mdType may be in the 3rd or 4th field
			if isRaidType(deviceFields[3], knownRaidTypes) {
				mdType = deviceFields[3]
				deviceStartIndex = 4
			} else if len(deviceFields) > 4 && isRaidType(deviceFields[4], knownRaidTypes) {
				// if the 3rd field is (...), the 4th field is the mdType
				mdType = deviceFields[4]
				deviceStartIndex = 5
			}
		}

		if len(lines) <= i+3 {
			return nil, fmt.Errorf("%w: Too few lines for md device: %q", ErrFileParse, mdName)
		}

		// Failed (Faulty) disks have the suffix (F) & Spare disks have the suffix (S).
		fail := int64(strings.Count(line, "(F)"))
		spare := int64(strings.Count(line, "(S)"))
		active, total, down, size, err := evalStatusLine(lines[i], lines[i+1])

		if err != nil {
			return nil, fmt.Errorf("%w: Cannot parse md device lines: %v: %w", ErrFileParse, active, err)
		}

		syncLineIdx := i + 2
		if strings.Contains(lines[i+2], "bitmap") { // skip bitmap line
			syncLineIdx++
		}

		// If device is syncing at the moment, get the number of currently
		// synced bytes, otherwise that number equals the size of the device.
		blocksSynced := size
		blocksToBeSynced := size
		speed := float64(0)
		finish := float64(0)
		pct := float64(0)
		recovering := strings.Contains(lines[syncLineIdx], "recovery")
		reshaping := strings.Contains(lines[syncLineIdx], "reshape")
		resyncing := strings.Contains(lines[syncLineIdx], "resync")
		checking := strings.Contains(lines[syncLineIdx], "check")

		// Append recovery and resyncing state info.
		if recovering || resyncing || checking || reshaping {
			switch {
			case recovering:
				state = "recovering"
			case reshaping:
				state = "reshaping"
			case checking:
				state = "checking"
			default:
				state = "resyncing"
			}

			// Handle case when resync=PENDING or resync=DELAYED.
			if strings.Contains(lines[syncLineIdx], "PENDING") ||
				strings.Contains(lines[syncLineIdx], "DELAYED") {
				blocksSynced = 0
			} else {
				blocksSynced, blocksToBeSynced, pct, finish, speed, err = evalRecoveryLine(lines[syncLineIdx])
				if err != nil {
					return nil, fmt.Errorf("%w: Cannot parse sync line in md device: %q: %w", ErrFileParse, mdName, err)
				}
			}
		}

		devices, err := evalComponentDevices(deviceFields[deviceStartIndex:])
		if err != nil {
			return nil, fmt.Errorf("error parsing components in md device %q: %w", mdName, err)
		}

		mdStats = append(mdStats, MDStat{
			Name:                   mdName,
			Type:                   mdType,
			ActivityState:          state,
			DisksActive:            active,
			DisksFailed:            fail,
			DisksDown:              down,
			DisksSpare:             spare,
			DisksTotal:             total,
			BlocksTotal:            size,
			BlocksSynced:           blocksSynced,
			BlocksToBeSynced:       blocksToBeSynced,
			BlocksSyncedPct:        pct,
			BlocksSyncedFinishTime: finish,
			BlocksSyncedSpeed:      speed,
			Devices:                devices,
		})
	}

	return mdStats, nil
}

// check if a string's format is like the mdType
// Rule 1: mdType should not be like (...)
// Rule 2: mdType should not be like sda[0]
// .
func isRaidType(mdType string, knownRaidTypes map[string]bool) bool {
	_, ok := knownRaidTypes[mdType]
	return !strings.ContainsAny(mdType, "([") && ok
}

func evalStatusLine(deviceLine, statusLine string) (active, total, down, size int64, err error) {
	// e.g. 523968 blocks super 1.2 [4/4] [UUUU]
	statusFields := strings.Fields(statusLine)
	if len(statusFields) < 1 {
		return 0, 0, 0, 0, fmt.Errorf("%w: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
	}

	sizeStr := statusFields[0]
	size, err = strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%w: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
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
		return 0, 0, 0, 0, fmt.Errorf("%w: Could not fild all substring matches %s: %w", ErrFileParse, statusLine, err)
	}

	total, err = strconv.ParseInt(matches[2], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%w: Unexpected statusline %q: %w", ErrFileParse, statusLine, err)
	}

	active, err = strconv.ParseInt(matches[3], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("%w: Unexpected active %d: %w", ErrFileParse, active, err)
	}
	down = int64(strings.Count(matches[4], "_"))

	return active, total, down, size, nil
}

func evalRecoveryLine(recoveryLine string) (blocksSynced int64, blocksToBeSynced int64, pct float64, finish float64, speed float64, err error) {
	matches := recoveryLineBlocksRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return 0, 0, 0, 0, 0, fmt.Errorf("%w: Unexpected recoveryLine blocks %s: %w", ErrFileParse, recoveryLine, err)
	}

	blocks := strings.Split(matches[1], "/")
	blocksSynced, err = strconv.ParseInt(blocks[0], 10, 64)
	if err != nil {
		return 0, 0, 0, 0, 0, fmt.Errorf("%w: Unable to parse recovery blocks synced %q: %w", ErrFileParse, matches[1], err)
	}

	blocksToBeSynced, err = strconv.ParseInt(blocks[1], 10, 64)
	if err != nil {
		return blocksSynced, 0, 0, 0, 0, fmt.Errorf("%w: Unable to parse recovery to be synced blocks %q: %w", ErrFileParse, matches[2], err)
	}

	// Get percentage complete
	matches = recoveryLinePctRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return blocksSynced, blocksToBeSynced, 0, 0, 0, fmt.Errorf("%w: Unexpected recoveryLine matching percentage %s", ErrFileParse, recoveryLine)
	}
	pct, err = strconv.ParseFloat(strings.TrimSpace(matches[1]), 64)
	if err != nil {
		return blocksSynced, blocksToBeSynced, 0, 0, 0, fmt.Errorf("%w: Error parsing float from recoveryLine %q", ErrFileParse, recoveryLine)
	}

	// Get time expected left to complete
	matches = recoveryLineFinishRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return blocksSynced, blocksToBeSynced, pct, 0, 0, fmt.Errorf("%w: Unexpected recoveryLine matching est. finish time: %s", ErrFileParse, recoveryLine)
	}
	finish, err = strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return blocksSynced, blocksToBeSynced, pct, 0, 0, fmt.Errorf("%w: Unable to parse float from recoveryLine: %q", ErrFileParse, recoveryLine)
	}

	// Get recovery speed
	matches = recoveryLineSpeedRE.FindStringSubmatch(recoveryLine)
	if len(matches) != 2 {
		return blocksSynced, blocksToBeSynced, pct, finish, 0, fmt.Errorf("%w: Unexpected recoveryLine value: %s", ErrFileParse, recoveryLine)
	}
	speed, err = strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return blocksSynced, blocksToBeSynced, pct, finish, 0, fmt.Errorf("%w: Error parsing float from recoveryLine: %q: %w", ErrFileParse, recoveryLine, err)
	}

	return blocksSynced, blocksToBeSynced, pct, finish, speed, nil
}

func evalComponentDevices(deviceFields []string) ([]MDStatComponent, error) {
	mdComponentDevices := make([]MDStatComponent, 0)
	for _, field := range deviceFields {
		match := componentDeviceRE.FindStringSubmatch(field)
		if match == nil {
			continue
		}
		descriptorIndex, err := strconv.ParseInt(match[2], 10, 32)
		if err != nil {
			return mdComponentDevices, fmt.Errorf("error parsing int from device %q: %w", match[2], err)
		}
		mdComponentDevices = append(mdComponentDevices, MDStatComponent{
			Name:            match[1],
			DescriptorIndex: int32(descriptorIndex),
			// match may contain one or more of these
			// https://github.com/torvalds/linux/blob/7ec462100ef9142344ddbf86f2c3008b97acddbe/drivers/md/md.c#L8376-L8392
			Faulty:      strings.Contains(match[3], "(F)"),
			Spare:       strings.Contains(match[3], "(S)"),
			Journal:     strings.Contains(match[3], "(J)"),
			Replacement: strings.Contains(match[3], "(R)"),
			WriteMostly: strings.Contains(match[3], "(W)"),
		})
	}

	return mdComponentDevices, nil
}
