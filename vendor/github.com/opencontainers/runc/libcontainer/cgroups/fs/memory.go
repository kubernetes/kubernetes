// +build linux

package fs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
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
	}
	// reset error.
	err = nil
	if path == "" {
		// Invalid input.
		return fmt.Errorf("invalid path for memory cgroups: %+v", d)
	}
	defer func() {
		if err != nil {
			os.RemoveAll(path)
		}
	}()
	if !cgroups.PathExists(path) {
		if err = os.MkdirAll(path, 0755); err != nil {
			return err
		}
	}
	if memoryAssigned(d.config) {
		// We have to set kernel memory here, as we can't change it once
		// processes have been attached to the cgroup.
		if err = s.SetKernelMemory(path, d.config); err != nil {
			return err
		}
	}
	// We need to join memory cgroup after set memory limits, because
	// kmem.limit_in_bytes can only be set when the cgroup is empty.
	if _, jerr := d.join("memory"); jerr != nil && !cgroups.IsNotFound(jerr) {
		err = jerr
		return err
	}
	return nil
}

func getModifyTime(path string) (time.Time, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to get memory cgroups creation time: %v", err)
	}
	return stat.ModTime(), nil
}

func (s *MemoryGroup) SetKernelMemory(path string, cgroup *configs.Cgroup) error {
	// This has to be done separately because it has special
	// constraints (it can only be initialized before setting up a
	// hierarchy or adding a task to the cgroups. However, if
	// sucessfully initialized, it can be updated anytime afterwards)
	if cgroup.Resources.KernelMemory != 0 {
		// Is kmem.limit_in_bytes already set?
		// memory.kmem.max_usage_in_bytes is a read-only file. Use it to get cgroups creation time.
		kmemCreationTime, err := getModifyTime(filepath.Join(path, "memory.kmem.max_usage_in_bytes"))
		if err != nil {
			return err
		}
		kmemLimitsUpdateTime, err := getModifyTime(filepath.Join(path, "memory.kmem.limit_in_bytes"))
		if err != nil {
			return err
		}
		// kmem.limit_in_bytes has already been set if its update time is after that of creation time.
		// We use `!=` op instead of `>` because updates are losing precision compared to creation.
		kmemInitialized := !kmemLimitsUpdateTime.Equal(kmemCreationTime)
		if !kmemInitialized {
			// If there's already tasks in the cgroup, we can't change the limit either
			tasks, err := getCgroupParamString(path, "tasks")
			if err != nil {
				return err
			}
			if tasks != "" {
				return fmt.Errorf("cannot set kmem.limit_in_bytes after task have joined this cgroup")
			}
		}
		if err := writeFile(path, "memory.kmem.limit_in_bytes", strconv.FormatInt(cgroup.Resources.KernelMemory, 10)); err != nil {
			return err
		}
	}
	return nil
}

func setMemoryAndSwap(path string, cgroup *configs.Cgroup) error {
	// When memory and swap memory are both set, we need to handle the cases
	// for updating container.
	if cgroup.Resources.Memory != 0 && cgroup.Resources.MemorySwap > 0 {
		memoryUsage, err := getMemoryData(path, "")
		if err != nil {
			return err
		}

		// When update memory limit, we should adapt the write sequence
		// for memory and swap memory, so it won't fail because the new
		// value and the old value don't fit kernel's validation.
		if memoryUsage.Limit < uint64(cgroup.Resources.MemorySwap) {
			if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
				return err
			}
			if err := writeFile(path, "memory.limit_in_bytes", strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
		} else {
			if err := writeFile(path, "memory.limit_in_bytes", strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
			if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
				return err
			}
		}
	} else {
		if cgroup.Resources.Memory != 0 {
			if err := writeFile(path, "memory.limit_in_bytes", strconv.FormatInt(cgroup.Resources.Memory, 10)); err != nil {
				return err
			}
		}
		if cgroup.Resources.MemorySwap > 0 {
			if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.Resources.MemorySwap, 10)); err != nil {
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

	if err := s.SetKernelMemory(path, cgroup); err != nil {
		return err
	}

	if cgroup.Resources.MemoryReservation != 0 {
		if err := writeFile(path, "memory.soft_limit_in_bytes", strconv.FormatInt(cgroup.Resources.MemoryReservation, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.KernelMemoryTCP != 0 {
		if err := writeFile(path, "memory.kmem.tcp.limit_in_bytes", strconv.FormatInt(cgroup.Resources.KernelMemoryTCP, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.OomKillDisable {
		if err := writeFile(path, "memory.oom_control", "1"); err != nil {
			return err
		}
	}
	if cgroup.Resources.MemorySwappiness == nil || int64(*cgroup.Resources.MemorySwappiness) == -1 {
		return nil
	} else if int64(*cgroup.Resources.MemorySwappiness) >= 0 && int64(*cgroup.Resources.MemorySwappiness) <= 100 {
		if err := writeFile(path, "memory.swappiness", strconv.FormatInt(*cgroup.Resources.MemorySwappiness, 10)); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("invalid value:%d. valid memory swappiness range is 0-100", int64(*cgroup.Resources.MemorySwappiness))
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
		t, v, err := getCgroupParamKeyValue(sc.Text())
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

	return nil
}

func memoryAssigned(cgroup *configs.Cgroup) bool {
	return cgroup.Resources.Memory != 0 ||
		cgroup.Resources.MemoryReservation != 0 ||
		cgroup.Resources.MemorySwap > 0 ||
		cgroup.Resources.KernelMemory > 0 ||
		cgroup.Resources.KernelMemoryTCP > 0 ||
		cgroup.Resources.OomKillDisable ||
		(cgroup.Resources.MemorySwappiness != nil && *cgroup.Resources.MemorySwappiness != -1)
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

	value, err := getCgroupParamUint(path, usage)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", usage, err)
	}
	memoryData.Usage = value
	value, err = getCgroupParamUint(path, maxUsage)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", maxUsage, err)
	}
	memoryData.MaxUsage = value
	value, err = getCgroupParamUint(path, failcnt)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", failcnt, err)
	}
	memoryData.Failcnt = value
	value, err = getCgroupParamUint(path, limit)
	if err != nil {
		if moduleName != "memory" && os.IsNotExist(err) {
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, fmt.Errorf("failed to parse %s - %v", limit, err)
	}
	memoryData.Limit = value

	return memoryData, nil
}
