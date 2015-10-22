// +build linux

package fs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/configs"
)

type MemoryGroup struct {
}

func (s *MemoryGroup) Apply(d *data) error {
	dir, err := d.join("memory")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	defer func() {
		if err != nil {
			os.RemoveAll(dir)
		}
	}()

	if err := s.Set(dir, d.c); err != nil {
		return err
	}

	return nil
}

func (s *MemoryGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Memory != 0 {
		if err := writeFile(path, "memory.limit_in_bytes", strconv.FormatInt(cgroup.Memory, 10)); err != nil {
			return err
		}
	}
	if cgroup.MemoryReservation != 0 {
		if err := writeFile(path, "memory.soft_limit_in_bytes", strconv.FormatInt(cgroup.MemoryReservation, 10)); err != nil {
			return err
		}
	}
	if cgroup.MemorySwap > 0 {
		if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.MemorySwap, 10)); err != nil {
			return err
		}
	}
	if cgroup.KernelMemory > 0 {
		if err := writeFile(path, "memory.kmem.limit_in_bytes", strconv.FormatInt(cgroup.KernelMemory, 10)); err != nil {
			return err
		}
	}

	if cgroup.OomKillDisable {
		if err := writeFile(path, "memory.oom_control", "1"); err != nil {
			return err
		}
	}
	if cgroup.MemorySwappiness >= 0 && cgroup.MemorySwappiness <= 100 {
		if err := writeFile(path, "memory.swappiness", strconv.FormatInt(cgroup.MemorySwappiness, 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *MemoryGroup) Remove(d *data) error {
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

	return nil
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

	return memoryData, nil
}
