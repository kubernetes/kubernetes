// +build !windows,!solaris

package stats

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/system"
)

/*
#include <unistd.h>
*/
import "C"

// platformNewStatsCollector performs platform specific initialisation of the
// Collector structure.
func platformNewStatsCollector(s *Collector) {
	s.clockTicksPerSecond = uint64(system.GetClockTicks())
}

const nanoSecondsPerSecond = 1e9

// getSystemCPUUsage returns the host system's cpu usage in
// nanoseconds. An error is returned if the format of the underlying
// file does not match.
//
// Uses /proc/stat defined by POSIX. Looks for the cpu
// statistics line and then sums up the first seven fields
// provided. See `man 5 proc` for details on specific field
// information.
func (s *Collector) getSystemCPUUsage() (uint64, error) {
	var line string
	f, err := os.Open("/proc/stat")
	if err != nil {
		return 0, err
	}
	defer func() {
		s.bufReader.Reset(nil)
		f.Close()
	}()
	s.bufReader.Reset(f)
	err = nil
	for err == nil {
		line, err = s.bufReader.ReadString('\n')
		if err != nil {
			break
		}
		parts := strings.Fields(line)
		switch parts[0] {
		case "cpu":
			if len(parts) < 8 {
				return 0, fmt.Errorf("invalid number of cpu fields")
			}
			var totalClockTicks uint64
			for _, i := range parts[1:8] {
				v, err := strconv.ParseUint(i, 10, 64)
				if err != nil {
					return 0, fmt.Errorf("Unable to convert value %s to int: %s", i, err)
				}
				totalClockTicks += v
			}
			return (totalClockTicks * nanoSecondsPerSecond) /
				s.clockTicksPerSecond, nil
		}
	}
	return 0, fmt.Errorf("invalid stat format. Error trying to parse the '/proc/stat' file")
}

func (s *Collector) getNumberOnlineCPUs() (uint32, error) {
	i, err := C.sysconf(C._SC_NPROCESSORS_ONLN)
	// According to POSIX - errno is undefined after successful
	// sysconf, and can be non-zero in several cases, so look for
	// error in returned value not in errno.
	// (https://sourceware.org/bugzilla/show_bug.cgi?id=21536)
	if i == -1 {
		return 0, err
	}
	return uint32(i), nil
}
