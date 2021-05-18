// +build linux

package fs

import (
	"bufio"
	"fmt"
	"math"
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

func (s *MemoryGroup) Apply(path string, d *cgroupData) (err error) {
	if path == "" {
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
	return join(path, d.pid)
}

func setMemoryAndSwap(path string, cgroup *configs.Cgroup) error {
	// If the memory update is set to -1 and the swap is not explicitly
	// set, we should also set swap to -1, it means unlimited memory.
	if cgroup.Resources.Memory == -1 && cgroup.Resources.MemorySwap == 0 {
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

func (s *MemoryGroup) GetStats(path string, stats *cgroups.Stats) error {
	// Set stats from memory.stat.
	statsFile, err := fscommon.OpenFile(path, "memory.stat", os.O_RDONLY)
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

func getPageUsageByNUMA(cgroupPath string) (cgroups.PageUsageByNUMA, error) {
	const (
		maxColumns = math.MaxUint8 + 1
		filename   = "memory.numa_stat"
	)
	stats := cgroups.PageUsageByNUMA{}

	file, err := fscommon.OpenFile(cgroupPath, filename, os.O_RDONLY)
	if os.IsNotExist(err) {
		return stats, nil
	} else if err != nil {
		return stats, err
	}

	// File format is documented in linux/Documentation/cgroup-v1/memory.txt
	// and it looks like this:
	//
	// total=<total pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// file=<total file pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// anon=<total anon pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// unevictable=<total anon pages> N0=<node 0 pages> N1=<node 1 pages> ...
	// hierarchical_<counter>=<counter pages> N0=<node 0 pages> N1=<node 1 pages> ...

	scanner := bufio.NewScanner(file)
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
					return stats, fmt.Errorf("malformed line %q in %s",
						line, filename)
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
					return stats, err
				}
				field.Nodes = map[uint8]uint64{}
			} else { // Subsequent columns: key is N<id>, val is usage.
				if len(key) < 2 || key[0] != 'N' {
					// This is definitely an error.
					return stats, fmt.Errorf("malformed line %q in %s",
						line, filename)
				}

				n, err := strconv.ParseUint(key[1:], 10, 8)
				if err != nil {
					return cgroups.PageUsageByNUMA{}, err
				}

				usage, err := strconv.ParseUint(val, 10, 64)
				if err != nil {
					return cgroups.PageUsageByNUMA{}, err
				}

				field.Nodes[uint8(n)] = usage
			}

		}
	}
	err = scanner.Err()
	if err != nil {
		return cgroups.PageUsageByNUMA{}, err
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
