// +build linux

package fs2

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
)

func setMemory(dirPath string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.MemorySwap != 0 {
		if err := fscommon.WriteFile(dirPath, "memory.swap.max", strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.Memory != 0 {
		if err := fscommon.WriteFile(dirPath, "memory.max", strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
			return err
		}
	}

	// cgroup.Resources.KernelMemory is ignored

	if cgroup.Resources.MemoryReservation != 0 {
		if err := fscommon.WriteFile(dirPath, "memory.low", strconv.FormatInt(cgroup.Resources.MemoryReservation, 10)); err != nil {
			return err
		}
	}

	return nil
}

func statMemory(dirPath string, stats *cgroups.Stats) error {
	// Set stats from memory.stat.
	statsFile, err := os.Open(filepath.Join(dirPath, "memory.stat"))
	if err != nil {
		return err
	}
	defer statsFile.Close()

	sc := bufio.NewScanner(statsFile)
	for sc.Scan() {
		t, v, err := fscommon.GetCgroupParamKeyValue(sc.Text())
		if err != nil {
			return errors.Wrapf(err, "failed to parse memory.stat (%q)", sc.Text())
		}
		stats.MemoryStats.Stats[t] = v
	}
	stats.MemoryStats.Cache = stats.MemoryStats.Stats["cache"]

	memoryUsage, err := getMemoryDataV2(dirPath, "")
	if err != nil {
		return err
	}
	stats.MemoryStats.Usage = memoryUsage
	swapUsage, err := getMemoryDataV2(dirPath, "swap")
	if err != nil {
		return err
	}
	stats.MemoryStats.SwapUsage = swapUsage

	stats.MemoryStats.UseHierarchy = true
	return nil
}

func getMemoryDataV2(path, name string) (cgroups.MemoryData, error) {
	memoryData := cgroups.MemoryData{}

	moduleName := "memory"
	if name != "" {
		moduleName = strings.Join([]string{"memory", name}, ".")
	}
	usage := strings.Join([]string{moduleName, "current"}, ".")
	limit := strings.Join([]string{moduleName, "max"}, ".")

	value, err := fscommon.GetCgroupParamUint(path, usage)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, errors.Wrapf(err, "failed to parse %s", usage)
	}
	memoryData.Usage = value

	value, err = fscommon.GetCgroupParamUint(path, limit)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, errors.Wrapf(err, "failed to parse %s", limit)
	}
	memoryData.Limit = value

	return memoryData, nil
}
