// +build linux

package fs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

const (
	cgroupMemorySwapLimit = "memory.memsw.limit_in_bytes"
	cgroupMemoryLimit     = "memory.limit_in_bytes"
)

type MemoryGroup struct {
}

func (s *MemoryGroup) Name() string {
	return "memory"
}

func (s *MemoryGroup) Apply(d *cgroupData) (err error) {
	path, err := d.path("memory")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	} else if path == "" {
		return nil
	}
	if memoryAssigned(d.config) {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := os.MkdirAll(path, 0755); err != nil {
				return err
			}
			// Only enable kernel memory accouting when this cgroup
			// is created by libcontainer, otherwise we might get
			// error when people use `cgroupsPath` to join an existed
			// cgroup whose kernel memory is not initialized.
			if err := EnableKernelMemoryAccounting(path); err != nil {
				return err
			}
		}
	}
	defer func() {
		if err != nil {
			os.RemoveAll(path)
		}
	}()

	// We need to join memory cgroup after set memory limits, because
	// kmem.limit_in_bytes can only be set when the cgroup is empty.
	_, err = d.join("memory")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func setMemoryAndSwap(path string, cgroup *configs.Cgroup) error {
	// If the memory update is set to -1 we should also
	// set swap to -1, it means unlimited memory.
	if cgroup.Resources.Memory == -1 {
		// Only set swap if it's enabled in kernel
		if cgroups.PathExists(filepath.Join(path, cgroupMemorySwapLimit)) {
			cgroup.Resources.MemorySwap = -1
		}
	}

	// When memory and swap memory are both set, we need to handle the cases
	// for updating container.
	if cgroup.Resources.Memory != 0 && cgroup.Resources.MemorySwap != 0 {
		memoryUsage, err := getMemoryData(path, "")
		if err != nil {
			return err
		}

		// When update memory limit, we should adapt the write sequence
		// for memory and swap memory, so it won't fail because the new
		// value and the old value don't fit kernel's validation.
		if cgroup.Resources.MemorySwap == -1 || memoryUsage.Limit < uint64(cgroup.Resources.MemorySwap) {
			if err := fscommon.WriteFile(path, cgroupMemorySwapLimit, strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
				return err
			}
			if err := fscommon.WriteFile(path, cgroupMemoryLimit, strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
		} else {
			if err := fscommon.WriteFile(path, cgroupMemoryLimit, strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
			if err := fscommon.WriteFile(path, cgroupMemorySwapLimit, strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
				return err
			}
		}
	} else {
		if cgroup.Resources.Memory != 0 {
			if err := fscommon.WriteFile(path, cgroupMemoryLimit, strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
		}
		if cgroup.Resources.MemorySwap != 0 {
			if err := fscommon.WriteFile(path, cgroupMemorySwapLimit, strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
				return err
			}
		}
	}

	return nil
}

func (s *MemoryGroup) Set(path string, cgroup *configs.Cgroup) error {
	if err := setMemoryAndSwap(path, cgroup); err != nil {
		return err
	}

	if cgroup.Resources.KernelMemory != 0 {
		if err := setKernelMemory(path, cgroup.Resources.KernelMemory); err != nil {
			return err
		}
	}

	if cgroup.Resources.MemoryReservation != 0 {
		if err := fscommon.WriteFile(path, "memory.soft_limit_in_bytes", strconv.FormatInt(cgroup.Resources.MemoryReservation, 10)); err != nil {
			return err
		}
	}

	if cgroup.Resources.KernelMemoryTCP != 0 {
		if err := fscommon.WriteFile(path, "memory.kmem.tcp.limit_in_bytes", strconv.FormatInt(cgroup.Resources.KernelMemoryTCP, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.OomKillDisable {
		if err := fscommon.WriteFile(path, "memory.oom_control", "1"); err != nil {
			return err
		}
	}
	if cgroup.Resources.MemorySwappiness == nil || int64(*cgroup.Resources.MemorySwappiness) == -1 {
		return nil
	} else if *cgroup.Resources.MemorySwappiness <= 100 {
		if err := fscommon.WriteFile(path, "memory.swappiness", strconv.FormatUint(*cgroup.Resources.MemorySwappiness, 10)); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("invalid value:%d. valid memory swappiness range is 0-100", *cgroup.Resources.MemorySwappiness)
	}

	return nil
}

func (s *MemoryGroup) Remove(d *cgroupData) error {
	return removePath(d.path("memory"))
}

func (s *MemoryGroup) GetStats(path string, stats *cgroups.Stats) error {
	// Set stats from memory.stat.
	statsFile, err := os.Open(filepath.Join(path, "memory.stat"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer statsFile.Close()

	sc := bufio.NewScanner(statsFile)
	for sc.Scan() {
		t, v, err := fscommon.GetCgroupParamKeyValue(sc.Text())
		if err != nil {
			return fmt.Errorf("failed to parse memory.stat (%q) - %v", sc.Text(), err)
		}
		stats.MemoryStats.Stats[t] = v
	}
	stats.MemoryStats.Cache = stats.MemoryStats.Stats["cache"]

	memoryUsage, err := getMemoryData(path, "")
	if err != nil {
		return err
	}
	stats.MemoryStats.Usage = memoryUsage
	swapUsage, err := getMemoryData(path, "memsw")
	if err != nil {
		return err
	}
	stats.MemoryStats.SwapUsage = swapUsage
	kernelUsage, err := getMemoryData(path, "kmem")
	if err != nil {
		return err
	}
	stats.MemoryStats.KernelUsage = kernelUsage
	kernelTCPUsage, err := getMemoryData(path, "kmem.tcp")
	if err != nil {
		return err
	}
	stats.MemoryStats.KernelTCPUsage = kernelTCPUsage

	useHierarchy := strings.Join([]string{"memory", "use_hierarchy"}, ".")
	value, err := fscommon.GetCgroupParamUint(path, useHierarchy)
	if err != nil {
		return err
	}
	if value == 1 {
		stats.MemoryStats.UseHierarchy = true
	}
	return nil
}

func memoryAssigned(cgroup *configs.Cgroup) bool {
	return cgroup.Resources.Memory != 0 ||
		cgroup.Resources.MemoryReservation != 0 ||
		cgroup.Resources.MemorySwap > 0 ||
		cgroup.Resources.KernelMemory > 0 ||
		cgroup.Resources.KernelMemoryTCP > 0 ||
		cgroup.Resources.OomKillDisable ||
		(cgroup.Resources.MemorySwappiness != nil && int64(*cgroup.Resources.MemorySwappiness) != -1)
}

func getMemoryData(path, name string) (cgroups.MemoryData, error) {
	memoryData := cgroups.MemoryData{}

	moduleName := "memory"
	if name != "" {
		moduleName = strings.Join([]string{"memory", name}, ".")
	}
	usage := strings.Join([]string{moduleName, "usage_in_bytes"}, ".")
	maxUsage := strings.Join([]string{moduleName, "max_usage_in_bytes"}, ".")
	failcnt := strings.Join([]string{moduleName, "failcnt"}, ".")
	limit := strings.Join([]string{moduleName, "limit_in_bytes"}, ".")

	value, err := fscommon.GetCgroupParamUint(path, usage)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", usage, err)
	}
	memoryData.Usage = value
	value, err = fscommon.GetCgroupParamUint(path, maxUsage)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", maxUsage, err)
	}
	memoryData.MaxUsage = value
	value, err = fscommon.GetCgroupParamUint(path, failcnt)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", failcnt, err)
	}
	memoryData.Failcnt = value
	value, err = fscommon.GetCgroupParamUint(path, limit)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", limit, err)
	}
	memoryData.Limit = value

	return memoryData, nil
}
