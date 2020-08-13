// +build linux

package fs

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
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

type CpuacctGroup struct {
}

func (s *CpuacctGroup) Name() string {
	return "cpuacct"
}

func (s *CpuacctGroup) Apply(d *cgroupData) error {
	// we just want to join this group even though we don't set anything
	if _, err := d.join("cpuacct"); err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	return nil
}

func (s *CpuacctGroup) Set(path string, cgroup *configs.Cgroup) error {
	return nil
}

func (s *CpuacctGroup) Remove(d *cgroupData) error {
	return removePath(d.path("cpuacct"))
}

func (s *CpuacctGroup) GetStats(path string, stats *cgroups.Stats) error {
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
	userModeUsage := uint64(0)
	kernelModeUsage := uint64(0)
	const (
		userField   = "user"
		systemField = "system"
	)

	// Expected format:
	// user <usage in ticks>
	// system <usage in ticks>
	data, err := ioutil.ReadFile(filepath.Join(path, cgroupCpuacctStat))
	if err != nil {
		return 0, 0, err
	}
	fields := strings.Fields(string(data))
	if len(fields) < 4 {
		return 0, 0, fmt.Errorf("failure - %s is expected to have at least 4 fields", filepath.Join(path, cgroupCpuacctStat))
	}
	if fields[0] != userField {
		return 0, 0, fmt.Errorf("unexpected field %q in %q, expected %q", fields[0], cgroupCpuacctStat, userField)
	}
	if fields[2] != systemField {
		return 0, 0, fmt.Errorf("unexpected field %q in %q, expected %q", fields[2], cgroupCpuacctStat, systemField)
	}
	if userModeUsage, err = strconv.ParseUint(fields[1], 10, 64); err != nil {
		return 0, 0, err
	}
	if kernelModeUsage, err = strconv.ParseUint(fields[3], 10, 64); err != nil {
		return 0, 0, err
	}

	return (userModeUsage * nanosecondsInSecond) / clockTicks, (kernelModeUsage * nanosecondsInSecond) / clockTicks, nil
}

func getPercpuUsage(path string) ([]uint64, error) {
	percpuUsage := []uint64{}
	data, err := ioutil.ReadFile(filepath.Join(path, "cpuacct.usage_percpu"))
	if err != nil {
		return percpuUsage, err
	}
	for _, value := range strings.Fields(string(data)) {
		value, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return percpuUsage, fmt.Errorf("Unable to convert param value to uint64: %s", err)
		}
		percpuUsage = append(percpuUsage, value)
	}
	return percpuUsage, nil
}

func getPercpuUsageInModes(path string) ([]uint64, []uint64, error) {
	usageKernelMode := []uint64{}
	usageUserMode := []uint64{}

	file, err := os.Open(filepath.Join(path, cgroupCpuacctUsageAll))
	if os.IsNotExist(err) {
		return usageKernelMode, usageUserMode, nil
	} else if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Scan() //skipping header line

	for scanner.Scan() {
		lineFields := strings.SplitN(scanner.Text(), " ", cuacctUsageAllColumnsNumber+1)
		if len(lineFields) != cuacctUsageAllColumnsNumber {
			continue
		}

		usageInKernelMode, err := strconv.ParseUint(lineFields[kernelModeColumn], 10, 64)
		if err != nil {
			return nil, nil, fmt.Errorf("Unable to convert CPU usage in kernel mode to uint64: %s", err)
		}
		usageKernelMode = append(usageKernelMode, usageInKernelMode)

		usageInUserMode, err := strconv.ParseUint(lineFields[userModeColumn], 10, 64)
		if err != nil {
			return nil, nil, fmt.Errorf("Unable to convert CPU usage in user mode to uint64: %s", err)
		}
		usageUserMode = append(usageUserMode, usageInUserMode)
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, fmt.Errorf("Problem in reading %s line by line, %s", cgroupCpuacctUsageAll, err)
	}

	return usageKernelMode, usageUserMode, nil
}
