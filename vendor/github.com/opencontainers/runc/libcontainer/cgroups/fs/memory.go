package fs

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

const (
	cgroupMemorySwapLimit = "memory.memsw.limit_in_bytes"
	cgroupMemoryLimit     = "memory.limit_in_bytes"
	cgroupMemoryUsage     = "memory.usage_in_bytes"
	cgroupMemoryMaxUsage  = "memory.max_usage_in_bytes"
)

type MemoryGroup struct{}

func (s *MemoryGroup) Name() string {
	return "memory"
}

func (s *MemoryGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func setMemory(path string, val int64) error {
	if val == 0 {
		return nil
	}

	err := cgroups.WriteFile(path, cgroupMemoryLimit, strconv.FormatInt(val, 10))
	if !errors.Is(err, unix.EBUSY) {
		return err
	}

	// EBUSY means the kernel can't set new limit as it's too low
	// (lower than the current usage). Return more specific error.
	usage, err := fscommon.GetCgroupParamUint(path, cgroupMemoryUsage)
	if err != nil {
		return err
	}
	max, err := fscommon.GetCgroupParamUint(path, cgroupMemoryMaxUsage)
	if err != nil {
		return err
	}

	return fmt.Errorf("unable to set memory limit to %d (current usage: %d, peak usage: %d)", val, usage, max)
}

func setSwap(path string, val int64) error {
	if val == 0 {
		return nil
	}

	return cgroups.WriteFile(path, cgroupMemorySwapLimit, strconv.FormatInt(val, 10))
}

func setMemoryAndSwap(path string, r *configs.Resources) error {
	// If the memory update is set to -1 and the swap is not explicitly
	// set, we should also set swap to -1, it means unlimited memory.
	if r.Memory == -1 && r.MemorySwap == 0 {
		// Only set swap if it's enabled in kernel
		if cgroups.PathExists(filepath.Join(path, cgroupMemorySwapLimit)) {
			r.MemorySwap = -1
		}
	}

	// When memory and swap memory are both set, we need to handle the cases
	// for updating container.
	if r.Memory != 0 && r.MemorySwap != 0 {
		curLimit, err := fscommon.GetCgroupParamUint(path, cgroupMemoryLimit)
		if err != nil {
			return err
		}

		// When update memory limit, we should adapt the write sequence
		// for memory and swap memory, so it won't fail because the new
		// value and the old value don't fit kernel's validation.
		if r.MemorySwap == -1 || curLimit < uint64(r.MemorySwap) {
			if err := setSwap(path, r.MemorySwap); err != nil {
				return err
			}
			if err := setMemory(path, r.Memory); err != nil {
				return err
			}
			return nil
		}
	}

	if err := setMemory(path, r.Memory); err != nil {
		return err
	}
	if err := setSwap(path, r.MemorySwap); err != nil {
		return err
	}

	return nil
}

func (s *MemoryGroup) Set(path string, r *configs.Resources) error {
	if err := setMemoryAndSwap(path, r); err != nil {
		return err
	}

	// ignore KernelMemory and KernelMemoryTCP

	if r.MemoryReservation != 0 {
		if err := cgroups.WriteFile(path, "memory.soft_limit_in_bytes", strconv.FormatInt(r.MemoryReservation, 10)); err != nil {
			return err
		}
	}

	if r.OomKillDisable {
		if err := cgroups.WriteFile(path, "memory.oom_control", "1"); err != nil {
			return err
		}
	}
	if r.MemorySwappiness == nil || int64(*r.MemorySwappiness) == -1 {
		return nil
	} else if *r.MemorySwappiness <= 100 {
		if err := cgroups.WriteFile(path, "memory.swappiness", strconv.FormatUint(*r.MemorySwappiness, 10)); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("invalid memory swappiness value: %d (valid range is 0-100)", *r.MemorySwappiness)
	}

	return nil
}

func (s *MemoryGroup) GetStats(path string, stats *cgroups.Stats) error {
	const file = "memory.stat"
	statsFile, err := cgroups.OpenFile(path, file, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer statsFile.Close()

	sc := bufio.NewScanner(statsFile)
	for sc.Scan() {
		t, v, err := fscommon.ParseKeyValue(sc.Text())
		if err != nil {
			return &parseError{Path: path, File: file, Err: err}
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

	value, err := fscommon.GetCgroupParamUint(path, "memory.use_hierarchy")
	if err != nil {
		return err
	}
	if value == 1 {
		stats.MemoryStats.UseHierarchy = true
	}

	pagesByNUMA, err := getPageUsageByNUMA(path)
	if err != nil {
		return err
	}
	stats.MemoryStats.PageUsageByNUMA = pagesByNUMA

	return nil
}

func getMemoryData(path, name string) (cgroups.MemoryData, error) {
	memoryData := cgroups.MemoryData{}

	moduleName := "memory"
	if name != "" {
		moduleName = "memory." + name
	}
	var (
		usage    = moduleName + ".usage_in_bytes"
		maxUsage = moduleName + ".max_usage_in_bytes"
		failcnt  = moduleName + ".failcnt"
		limit    = moduleName + ".limit_in_bytes"
	)

	value, err := fscommon.GetCgroupParamUint(path, usage)
	if err != nil {
		if name != "" && os.IsNotExist(err) {
			// Ignore ENOENT as swap and kmem controllers
			// are optional in the kernel.
			return cgroups.MemoryData{}, nil
		}
		return cgroups.MemoryData{}, err
	}
	memoryData.Usage = value
	value, err = fscommon.GetCgroupParamUint(path, maxUsage)
	if err != nil {
		return cgroups.MemoryData{}, err
	}
	memoryData.MaxUsage = value
	value, err = fscommon.GetCgroupParamUint(path, failcnt)
	if err != nil {
		return cgroups.MemoryData{}, err
	}
	memoryData.Failcnt = value
	value, err = fscommon.GetCgroupParamUint(path, limit)
	if err != nil {
		return cgroups.MemoryData{}, err
	}
	memoryData.Limit = value

	return memoryData, nil
}

func getPageUsageByNUMA(path string) (cgroups.PageUsageByNUMA, error) {
	const (
		maxColumns = math.MaxUint8 + 1
		file       = "memory.numa_stat"
	)
	stats := cgroups.PageUsageByNUMA{}

	fd, err := cgroups.OpenFile(path, file, os.O_RDONLY)
	if os.IsNotExist(err) {
		return stats, nil
	} else if err != nil {
		return stats, err
	}
	defer fd.Close()

	// File format is documented in linux/Documentation/cgroup-v1/memory.txt
	// and it looks like this:
	//
	// total=<total pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// file=<total file pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// anon=<total anon pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// unevictable=<total anon pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// hierarchical_<counter>=<counter pages> N0=<node 0 pages> N1=<node 1 pages> ...

	scanner := bufio.NewScanner(fd)
	for scanner.Scan() {
		var field *cgroups.PageStats

		line := scanner.Text()
		columns := strings.SplitN(line, " ", maxColumns)
		for i, column := range columns {
			byNode := strings.SplitN(column, "=", 2)
			// Some custom kernels have non-standard fields, like
			//   numa_locality 0 0 0 0 0 0 0 0 0 0
			//   numa_exectime 0
			if len(byNode) < 2 {
				if i == 0 {
					// Ignore/skip those.
					break
				} else {
					// The first column was already validated,
					// so be strict to the rest.
					return stats, malformedLine(path, file, line)
				}
			}
			key, val := byNode[0], byNode[1]
			if i == 0 { // First column: key is name, val is total.
				field = getNUMAField(&stats, key)
				if field == nil { // unknown field (new kernel?)
					break
				}
				field.Total, err = strconv.ParseUint(val, 0, 64)
				if err != nil {
					return stats, &parseError{Path: path, File: file, Err: err}
				}
				field.Nodes = map[uint8]uint64{}
			} else { // Subsequent columns: key is N<id>, val is usage.
				if len(key) < 2 || key[0] != 'N' {
					// This is definitely an error.
					return stats, malformedLine(path, file, line)
				}

				n, err := strconv.ParseUint(key[1:], 10, 8)
				if err != nil {
					return stats, &parseError{Path: path, File: file, Err: err}
				}

				usage, err := strconv.ParseUint(val, 10, 64)
				if err != nil {
					return stats, &parseError{Path: path, File: file, Err: err}
				}

				field.Nodes[uint8(n)] = usage
			}

		}
	}
	if err := scanner.Err(); err != nil {
		return cgroups.PageUsageByNUMA{}, &parseError{Path: path, File: file, Err: err}
	}

	return stats, nil
}

func getNUMAField(stats *cgroups.PageUsageByNUMA, name string) *cgroups.PageStats {
	switch name {
	case "total":
		return &stats.Total
	case "file":
		return &stats.File
	case "anon":
		return &stats.Anon
	case "unevictable":
		return &stats.Unevictable
	case "hierarchical_total":
		return &stats.Hierarchical.Total
	case "hierarchical_file":
		return &stats.Hierarchical.File
	case "hierarchical_anon":
		return &stats.Hierarchical.Anon
	case "hierarchical_unevictable":
		return &stats.Hierarchical.Unevictable
	}
	return nil
}
