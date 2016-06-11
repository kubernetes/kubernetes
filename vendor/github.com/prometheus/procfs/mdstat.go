package procfs

import (
	"fmt"
	"io/ioutil"
	"path"
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
	mdStatusFilePath := path.Join(string(fs), "mdstat")
	content, err := ioutil.ReadFile(mdStatusFilePath)
	if err != nil {
		return []MDStat{}, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
	}

	mdStatusFile := string(content)

	lines := strings.Split(mdStatusFile, "\n")
	var currentMD string

	// Each md has at least the deviceline, statusline and one empty line afterwards
	// so we will have probably something of the order len(lines)/3 devices
	// so we use that for preallocation.
	estimateMDs := len(lines) / 3
	mdStates := make([]MDStat, 0, estimateMDs)

	for i, l := range lines {
		if l == "" {
			// Skip entirely empty lines.
			continue
		}

		if l[0] == ' ' {
			// Those lines are not the beginning of a md-section.
			continue
		}

		if strings.HasPrefix(l, "Personalities") || strings.HasPrefix(l, "unused") {
			// We aren't interested in lines with general info.
			continue
		}

		mainLine := strings.Split(l, " ")
		if len(mainLine) < 3 {
			return mdStates, fmt.Errorf("error parsing mdline: %s", l)
		}
		currentMD = mainLine[0]      // name of md-device
		activityState := mainLine[2] // activity status of said md-device

		if len(lines) <= i+3 {
			return mdStates, fmt.Errorf("error parsing %s: entry for %s has fewer lines than expected", mdStatusFilePath, currentMD)
		}

		active, total, size, err := evalStatusline(lines[i+1]) // parse statusline, always present
		if err != nil {
			return mdStates, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
		}

		//
		// Now get the number of synced blocks.
		//

		// Get the line number of the syncing-line.
		var j int
		if strings.Contains(lines[i+2], "bitmap") { // then skip the bitmap line
			j = i + 3
		} else {
			j = i + 2
		}

		// If device is syncing at the moment, get the number of currently synced bytes,
		// otherwise that number equals the size of the device.
		syncedBlocks := size
		if strings.Contains(lines[j], "recovery") || strings.Contains(lines[j], "resync") {
			syncedBlocks, err = evalBuildline(lines[j])
			if err != nil {
				return mdStates, fmt.Errorf("error parsing %s: %s", mdStatusFilePath, err)
			}
		}

		mdStates = append(mdStates, MDStat{currentMD, activityState, active, total, size, syncedBlocks})

	}

	return mdStates, nil
}

func evalStatusline(statusline string) (active, total, size int64, err error) {
	matches := statuslineRE.FindStringSubmatch(statusline)

	// +1 to make it more obvious that the whole string containing the info is also returned as matches[0].
	if len(matches) != 3+1 {
		return 0, 0, 0, fmt.Errorf("unexpected number matches found in statusline: %s", statusline)
	}

	size, err = strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("%s in statusline: %s", err, statusline)
	}

	total, err = strconv.ParseInt(matches[2], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("%s in statusline: %s", err, statusline)
	}

	active, err = strconv.ParseInt(matches[3], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("%s in statusline: %s", err, statusline)
	}

	return active, total, size, nil
}

// Gets the size that has already been synced out of the sync-line.
func evalBuildline(buildline string) (int64, error) {
	matches := buildlineRE.FindStringSubmatch(buildline)

	// +1 to make it more obvious that the whole string containing the info is also returned as matches[0].
	if len(matches) < 1+1 {
		return 0, fmt.Errorf("too few matches found in buildline: %s", buildline)
	}

	if len(matches) > 1+1 {
		return 0, fmt.Errorf("too many matches found in buildline: %s", buildline)
	}

	syncedSize, err := strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s in buildline: %s", err, buildline)
	}

	return syncedSize, nil
}
