package fs

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

const (
	cgroupCpuacctStat     = "cpuacct.stat"
	cgroupCpuacctUsageAll = "cpuacct.usage_all"

	nanosecondsInSecond = 1000000000

	userModeColumn              = 1
	kernelModeColumn            = 2
	cuacctUsageAllColumnsNumber = 3

	// The value comes from `C.sysconf(C._SC_CLK_TCK)`, and
	// on Linux it's a constant which is safe to be hard coded,
	// so we can avoid using cgo here. For details, see:
	// https://github.com/containerd/cgroups/pull/12
	clockTicks uint64 = 100
)

type CpuacctGroup struct{}

func (s *CpuacctGroup) Name() string {
	return "cpuacct"
}

func (s *CpuacctGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func (s *CpuacctGroup) Set(_ string, _ *configs.Resources) error {
	return nil
}

func (s *CpuacctGroup) GetStats(path string, stats *cgroups.Stats) error {
	if !cgroups.PathExists(path) {
		return nil
	}
	userModeUsage, kernelModeUsage, err := getCpuUsageBreakdown(path)
	if err != nil {
		return err
	}

	totalUsage, err := fscommon.GetCgroupParamUint(path, "cpuacct.usage")
	if err != nil {
		return err
	}

	percpuUsage, err := getPercpuUsage(path)
	if err != nil {
		return err
	}

	percpuUsageInKernelmode, percpuUsageInUsermode, err := getPercpuUsageInModes(path)
	if err != nil {
		return err
	}

	stats.CpuStats.CpuUsage.TotalUsage = totalUsage
	stats.CpuStats.CpuUsage.PercpuUsage = percpuUsage
	stats.CpuStats.CpuUsage.PercpuUsageInKernelmode = percpuUsageInKernelmode
	stats.CpuStats.CpuUsage.PercpuUsageInUsermode = percpuUsageInUsermode
	stats.CpuStats.CpuUsage.UsageInUsermode = userModeUsage
	stats.CpuStats.CpuUsage.UsageInKernelmode = kernelModeUsage
	return nil
}

// Returns user and kernel usage breakdown in nanoseconds.
func getCpuUsageBreakdown(path string) (uint64, uint64, error) {
	var userModeUsage, kernelModeUsage uint64
	const (
		userField   = "user"
		systemField = "system"
		file        = cgroupCpuacctStat
	)

	// Expected format:
	// user <usage in ticks>
	// system <usage in ticks>
	data, err := cgroups.ReadFile(path, file)
	if err != nil {
		return 0, 0, err
	}
	// TODO: use strings.SplitN instead.
	fields := strings.Fields(data)
	if len(fields) < 4 || fields[0] != userField || fields[2] != systemField {
		return 0, 0, malformedLine(path, file, data)
	}
	if userModeUsage, err = strconv.ParseUint(fields[1], 10, 64); err != nil {
		return 0, 0, &parseError{Path: path, File: file, Err: err}
	}
	if kernelModeUsage, err = strconv.ParseUint(fields[3], 10, 64); err != nil {
		return 0, 0, &parseError{Path: path, File: file, Err: err}
	}

	return (userModeUsage * nanosecondsInSecond) / clockTicks, (kernelModeUsage * nanosecondsInSecond) / clockTicks, nil
}

func getPercpuUsage(path string) ([]uint64, error) {
	const file = "cpuacct.usage_percpu"
	percpuUsage := []uint64{}
	data, err := cgroups.ReadFile(path, file)
	if err != nil {
		return percpuUsage, err
	}
	// TODO: use strings.SplitN instead.
	for _, value := range strings.Fields(data) {
		value, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return percpuUsage, &parseError{Path: path, File: file, Err: err}
		}
		percpuUsage = append(percpuUsage, value)
	}
	return percpuUsage, nil
}

func getPercpuUsageInModes(path string) ([]uint64, []uint64, error) {
	usageKernelMode := []uint64{}
	usageUserMode := []uint64{}
	const file = cgroupCpuacctUsageAll

	fd, err := cgroups.OpenFile(path, file, os.O_RDONLY)
	if os.IsNotExist(err) {
		return usageKernelMode, usageUserMode, nil
	} else if err != nil {
		return nil, nil, err
	}
	defer fd.Close()

	scanner := bufio.NewScanner(fd)
	scanner.Scan() // skipping header line

	for scanner.Scan() {
		lineFields := strings.SplitN(scanner.Text(), " ", cuacctUsageAllColumnsNumber+1)
		if len(lineFields) != cuacctUsageAllColumnsNumber {
			continue
		}

		usageInKernelMode, err := strconv.ParseUint(lineFields[kernelModeColumn], 10, 64)
		if err != nil {
			return nil, nil, &parseError{Path: path, File: file, Err: err}
		}
		usageKernelMode = append(usageKernelMode, usageInKernelMode)

		usageInUserMode, err := strconv.ParseUint(lineFields[userModeColumn], 10, 64)
		if err != nil {
			return nil, nil, &parseError{Path: path, File: file, Err: err}
		}
		usageUserMode = append(usageUserMode, usageInUserMode)
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, &parseError{Path: path, File: file, Err: err}
	}

	return usageKernelMode, usageUserMode, nil
}
