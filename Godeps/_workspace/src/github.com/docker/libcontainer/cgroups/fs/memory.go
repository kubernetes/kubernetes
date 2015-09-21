package fs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/configs"
)

type MemoryGroup struct {
}

func (s *MemoryGroup) Apply(d *data) error {
	dir, err := d.join("memory")
	// only return an error for memory if it was specified
	if err != nil && (d.c.Memory != 0 || d.c.MemoryReservation != 0 || d.c.MemorySwap != 0) {
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
	// By default, MemorySwap is set to twice the size of Memory.
	if cgroup.MemorySwap == 0 && cgroup.Memory != 0 {
		if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.Memory*2, 10)); err != nil {
			return err
		}
	}
	if cgroup.MemorySwap > 0 {
		if err := writeFile(path, "memory.memsw.limit_in_bytes", strconv.FormatInt(cgroup.MemorySwap, 10)); err != nil {
			return err
		}
	}

	if cgroup.OomKillDisable {
		if err := writeFile(path, "memory.oom_control", "1"); err != nil {
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

	// Set memory usage and max historical usage.
	value, err := getCgroupParamUint(path, "memory.usage_in_bytes")
	if err != nil {
		return fmt.Errorf("failed to parse memory.usage_in_bytes - %v", err)
	}
	stats.MemoryStats.Usage = value
	stats.MemoryStats.Cache = stats.MemoryStats.Stats["cache"]
	value, err = getCgroupParamUint(path, "memory.max_usage_in_bytes")
	if err != nil {
		return fmt.Errorf("failed to parse memory.max_usage_in_bytes - %v", err)
	}
	stats.MemoryStats.MaxUsage = value
	value, err = getCgroupParamUint(path, "memory.failcnt")
	if err != nil {
		return fmt.Errorf("failed to parse memory.failcnt - %v", err)
	}
	stats.MemoryStats.Failcnt = value

	return nil
}
