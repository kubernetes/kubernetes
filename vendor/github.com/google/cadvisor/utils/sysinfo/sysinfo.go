// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sysinfo

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/sysfs"

	"k8s.io/klog/v2"
)

var (
	schedulerRegExp      = regexp.MustCompile(`.*\[(.*)\].*`)
	nodeDirRegExp        = regexp.MustCompile(`node/node(\d*)`)
	cpuDirRegExp         = regexp.MustCompile(`/cpu(\d+)`)
	memoryCapacityRegexp = regexp.MustCompile(`MemTotal:\s*([0-9]+) kB`)

	cpusPath = "/sys/devices/system/cpu"
)

const (
	cacheLevel2  = 2
	hugepagesDir = "hugepages/"
)

// Get information about block devices present on the system.
// Uses the passed in system interface to retrieve the low level OS information.
func GetBlockDeviceInfo(sysfs sysfs.SysFs) (map[string]info.DiskInfo, error) {
	disks, err := sysfs.GetBlockDevices()
	if err != nil {
		return nil, err
	}

	diskMap := make(map[string]info.DiskInfo)
	for _, disk := range disks {
		name := disk.Name()
		// Ignore non-disk devices.
		// TODO(rjnagal): Maybe just match hd, sd, and dm prefixes.
		if strings.HasPrefix(name, "loop") || strings.HasPrefix(name, "ram") || strings.HasPrefix(name, "sr") {
			continue
		}
		diskInfo := info.DiskInfo{
			Name: name,
		}
		dev, err := sysfs.GetBlockDeviceNumbers(name)
		if err != nil {
			return nil, err
		}
		n, err := fmt.Sscanf(dev, "%d:%d", &diskInfo.Major, &diskInfo.Minor)
		if err != nil || n != 2 {
			return nil, fmt.Errorf("could not parse device numbers from %s for device %s", dev, name)
		}
		out, err := sysfs.GetBlockDeviceSize(name)
		if err != nil {
			return nil, err
		}
		// Remove trailing newline before conversion.
		size, err := strconv.ParseUint(strings.TrimSpace(out), 10, 64)
		if err != nil {
			return nil, err
		}
		// size is in 512 bytes blocks.
		diskInfo.Size = size * 512

		diskInfo.Scheduler = "none"
		blkSched, err := sysfs.GetBlockDeviceScheduler(name)
		if err == nil {
			matches := schedulerRegExp.FindSubmatch([]byte(blkSched))
			if len(matches) >= 2 {
				diskInfo.Scheduler = string(matches[1])
			}
		}
		device := fmt.Sprintf("%d:%d", diskInfo.Major, diskInfo.Minor)
		diskMap[device] = diskInfo
	}
	return diskMap, nil
}

// Get information about network devices present on the system.
func GetNetworkDevices(sysfs sysfs.SysFs) ([]info.NetInfo, error) {
	devs, err := sysfs.GetNetworkDevices()
	if err != nil {
		return nil, err
	}
	netDevices := []info.NetInfo{}
	for _, dev := range devs {
		name := dev.Name()
		// Ignore docker, loopback, and veth devices.
		ignoredDevices := []string{"lo", "veth", "docker"}
		ignored := false
		for _, prefix := range ignoredDevices {
			if strings.HasPrefix(name, prefix) {
				ignored = true
				break
			}
		}
		if ignored {
			continue
		}
		address, err := sysfs.GetNetworkAddress(name)
		if err != nil {
			return nil, err
		}
		mtuStr, err := sysfs.GetNetworkMtu(name)
		if err != nil {
			return nil, err
		}
		var mtu int64
		n, err := fmt.Sscanf(mtuStr, "%d", &mtu)
		if err != nil || n != 1 {
			return nil, fmt.Errorf("could not parse mtu from %s for device %s", mtuStr, name)
		}
		netInfo := info.NetInfo{
			Name:       name,
			MacAddress: strings.TrimSpace(address),
			Mtu:        mtu,
		}
		speed, err := sysfs.GetNetworkSpeed(name)
		// Some devices don't set speed.
		if err == nil {
			var s int64
			n, err := fmt.Sscanf(speed, "%d", &s)
			if err != nil || n != 1 {
				return nil, fmt.Errorf("could not parse speed from %s for device %s", speed, name)
			}
			netInfo.Speed = s
		}
		netDevices = append(netDevices, netInfo)
	}
	return netDevices, nil
}

// GetHugePagesInfo returns information about pre-allocated huge pages
// hugepagesDirectory should be top directory of hugepages
// Such as: /sys/kernel/mm/hugepages/
func GetHugePagesInfo(sysFs sysfs.SysFs, hugepagesDirectory string) ([]info.HugePagesInfo, error) {
	var hugePagesInfo []info.HugePagesInfo
	files, err := sysFs.GetHugePagesInfo(hugepagesDirectory)
	if err != nil {
		// treat as non-fatal since kernels and machine can be
		// configured to disable hugepage support
		return hugePagesInfo, nil
	}

	for _, st := range files {
		nameArray := strings.Split(st.Name(), "-")
		pageSizeArray := strings.Split(nameArray[1], "kB")
		pageSize, err := strconv.ParseUint(string(pageSizeArray[0]), 10, 64)
		if err != nil {
			return hugePagesInfo, err
		}

		val, err := sysFs.GetHugePagesNr(hugepagesDirectory, st.Name())
		if err != nil {
			return hugePagesInfo, err
		}
		var numPages uint64
		// we use sscanf as the file as a new-line that trips up ParseUint
		// it returns the number of tokens successfully parsed, so if
		// n != 1, it means we were unable to parse a number from the file
		n, err := fmt.Sscanf(string(val), "%d", &numPages)
		if err != nil || n != 1 {
			return hugePagesInfo, fmt.Errorf("could not parse file nr_hugepage for %s, contents %q", st.Name(), string(val))
		}

		hugePagesInfo = append(hugePagesInfo, info.HugePagesInfo{
			NumPages: numPages,
			PageSize: pageSize,
		})
	}
	return hugePagesInfo, nil
}

// GetNodesInfo returns information about NUMA nodes and their topology
func GetNodesInfo(sysFs sysfs.SysFs) ([]info.Node, int, error) {
	nodes := []info.Node{}
	allLogicalCoresCount := 0

	nodesDirs, err := sysFs.GetNodesPaths()
	if err != nil {
		return nil, 0, err
	}

	if len(nodesDirs) == 0 {
		klog.Warningf("Nodes topology is not available, providing CPU topology")
		return getCPUTopology(sysFs)
	}

	for _, nodeDir := range nodesDirs {
		id, err := getMatchedInt(nodeDirRegExp, nodeDir)
		if err != nil {
			return nil, 0, err
		}
		node := info.Node{Id: id}

		cpuDirs, err := sysFs.GetCPUsPaths(nodeDir)
		if len(cpuDirs) == 0 {
			klog.Warningf("Found node without any CPU, nodeDir: %s, number of cpuDirs %d, err: %v", nodeDir, len(cpuDirs), err)
		} else {
			cores, err := getCoresInfo(sysFs, cpuDirs)
			if err != nil {
				return nil, 0, err
			}
			node.Cores = cores
			for _, core := range cores {
				allLogicalCoresCount += len(core.Threads)
			}
		}

		// On some Linux platforms(such as Arm64 guest kernel), cache info may not exist.
		// So, we should ignore error here.
		err = addCacheInfo(sysFs, &node)
		if err != nil {
			klog.V(1).Infof("Found node without cache information, nodeDir: %s", nodeDir)
		}

		node.Memory, err = getNodeMemInfo(sysFs, nodeDir)
		if err != nil {
			return nil, 0, err
		}

		hugepagesDirectory := fmt.Sprintf("%s/%s", nodeDir, hugepagesDir)
		node.HugePages, err = GetHugePagesInfo(sysFs, hugepagesDirectory)
		if err != nil {
			return nil, 0, err
		}

		nodes = append(nodes, node)
	}
	return nodes, allLogicalCoresCount, err
}

func getCPUTopology(sysFs sysfs.SysFs) ([]info.Node, int, error) {
	nodes := []info.Node{}

	cpusPaths, err := sysFs.GetCPUsPaths(cpusPath)
	if err != nil {
		return nil, 0, err
	}
	cpusCount := len(cpusPaths)

	if cpusCount == 0 {
		err = fmt.Errorf("Any CPU is not available, cpusPath: %s", cpusPath)
		return nil, 0, err
	}

	cpusByPhysicalPackageID, err := getCpusByPhysicalPackageID(sysFs, cpusPaths)
	if err != nil {
		return nil, 0, err
	}

	if len(cpusByPhysicalPackageID) == 0 {
		klog.Warningf("Cannot read any physical package id for any CPU")
		return nil, cpusCount, nil
	}

	for physicalPackageID, cpus := range cpusByPhysicalPackageID {
		node := info.Node{Id: physicalPackageID}

		cores, err := getCoresInfo(sysFs, cpus)
		if err != nil {
			return nil, 0, err
		}
		node.Cores = cores

		// On some Linux platforms(such as Arm64 guest kernel), cache info may not exist.
		// So, we should ignore error here.
		err = addCacheInfo(sysFs, &node)
		if err != nil {
			klog.V(1).Infof("Found cpu without cache information, cpuPath: %s", cpus)
		}
		nodes = append(nodes, node)
	}
	return nodes, cpusCount, nil
}

func getCpusByPhysicalPackageID(sysFs sysfs.SysFs, cpusPaths []string) (map[int][]string, error) {
	cpuPathsByPhysicalPackageID := make(map[int][]string)
	for _, cpuPath := range cpusPaths {

		rawPhysicalPackageID, err := sysFs.GetCPUPhysicalPackageID(cpuPath)
		if os.IsNotExist(err) {
			klog.Warningf("Cannot read physical package id for %s, physical_package_id file does not exist, err: %s", cpuPath, err)
			continue
		} else if err != nil {
			return nil, err
		}

		physicalPackageID, err := strconv.Atoi(rawPhysicalPackageID)
		if err != nil {
			return nil, err
		}

		if _, ok := cpuPathsByPhysicalPackageID[physicalPackageID]; !ok {
			cpuPathsByPhysicalPackageID[physicalPackageID] = make([]string, 0)
		}

		cpuPathsByPhysicalPackageID[physicalPackageID] = append(cpuPathsByPhysicalPackageID[physicalPackageID], cpuPath)
	}
	return cpuPathsByPhysicalPackageID, nil
}

// addCacheInfo adds information about cache for NUMA node
func addCacheInfo(sysFs sysfs.SysFs, node *info.Node) error {
	for coreID, core := range node.Cores {
		threadID := core.Threads[0] //get any thread for core
		caches, err := GetCacheInfo(sysFs, threadID)
		if err != nil {
			return err
		}

		numThreadsPerCore := len(core.Threads)
		numThreadsPerNode := len(node.Cores) * numThreadsPerCore

		for _, cache := range caches {
			c := info.Cache{
				Size:  cache.Size,
				Level: cache.Level,
				Type:  cache.Type,
			}
			if cache.Cpus == numThreadsPerNode && cache.Level > cacheLevel2 {
				// Add a node-level cache.
				cacheFound := false
				for _, nodeCache := range node.Caches {
					if nodeCache == c {
						cacheFound = true
					}
				}
				if !cacheFound {
					node.Caches = append(node.Caches, c)
				}
			} else if cache.Cpus == numThreadsPerCore {
				// Add core level cache
				node.Cores[coreID].Caches = append(node.Cores[coreID].Caches, c)
			}
			// Ignore unknown caches.
		}
	}
	return nil
}

// getNodeMemInfo returns information about total memory for NUMA node
func getNodeMemInfo(sysFs sysfs.SysFs, nodeDir string) (uint64, error) {
	rawMem, err := sysFs.GetMemInfo(nodeDir)
	if err != nil {
		//Ignore if per-node info is not available.
		klog.Warningf("Found node without memory information, nodeDir: %s", nodeDir)
		return 0, nil
	}
	matches := memoryCapacityRegexp.FindStringSubmatch(rawMem)
	if len(matches) != 2 {
		return 0, fmt.Errorf("failed to match regexp in output: %q", string(rawMem))
	}
	memory, err := strconv.ParseUint(matches[1], 10, 64)
	if err != nil {
		return 0, err
	}
	memory = memory * 1024 // Convert to bytes
	return uint64(memory), nil
}

// getCoresInfo retruns infromation about physical cores
func getCoresInfo(sysFs sysfs.SysFs, cpuDirs []string) ([]info.Core, error) {
	cores := make([]info.Core, 0, len(cpuDirs))
	for _, cpuDir := range cpuDirs {
		cpuID, err := getMatchedInt(cpuDirRegExp, cpuDir)
		if err != nil {
			return nil, fmt.Errorf("Unexpected format of CPU directory, cpuDirRegExp %s, cpuDir: %s", cpuDirRegExp, cpuDir)
		}
		if !sysFs.IsCPUOnline(cpuDir) {
			continue
		}

		rawPhysicalID, err := sysFs.GetCoreID(cpuDir)
		if os.IsNotExist(err) {
			klog.Warningf("Cannot read core id for %s, core_id file does not exist, err: %s", cpuDir, err)
			continue
		} else if err != nil {
			return nil, err
		}
		physicalID, err := strconv.Atoi(rawPhysicalID)
		if err != nil {
			return nil, err
		}

		coreIDx := -1
		for id, core := range cores {
			if core.Id == physicalID {
				coreIDx = id
			}
		}
		if coreIDx == -1 {
			cores = append(cores, info.Core{})
			coreIDx = len(cores) - 1
		}
		desiredCore := &cores[coreIDx]

		desiredCore.Id = physicalID
		if len(desiredCore.Threads) == 0 {
			desiredCore.Threads = []int{cpuID}
		} else {
			desiredCore.Threads = append(desiredCore.Threads, cpuID)
		}

		rawPhysicalPackageID, err := sysFs.GetCPUPhysicalPackageID(cpuDir)
		if os.IsNotExist(err) {
			klog.Warningf("Cannot read physical package id for %s, physical_package_id file does not exist, err: %s", cpuDir, err)
			continue
		} else if err != nil {
			return nil, err
		}

		physicalPackageID, err := strconv.Atoi(rawPhysicalPackageID)
		if err != nil {
			return nil, err
		}
		desiredCore.SocketID = physicalPackageID
	}
	return cores, nil
}

// GetCacheInfo return information about a cache accessible from the given cpu thread
func GetCacheInfo(sysFs sysfs.SysFs, id int) ([]sysfs.CacheInfo, error) {
	caches, err := sysFs.GetCaches(id)
	if err != nil {
		return nil, err
	}

	info := []sysfs.CacheInfo{}
	for _, cache := range caches {
		if !strings.HasPrefix(cache.Name(), "index") {
			continue
		}
		cacheInfo, err := sysFs.GetCacheInfo(id, cache.Name())
		if err != nil {
			return nil, err
		}
		info = append(info, cacheInfo)
	}
	return info, nil
}

func getNetworkStats(name string, sysFs sysfs.SysFs) (info.InterfaceStats, error) {
	var stats info.InterfaceStats
	var err error
	stats.Name = name
	stats.RxBytes, err = sysFs.GetNetworkStatValue(name, "rx_bytes")
	if err != nil {
		return stats, err
	}
	stats.RxPackets, err = sysFs.GetNetworkStatValue(name, "rx_packets")
	if err != nil {
		return stats, err
	}
	stats.RxErrors, err = sysFs.GetNetworkStatValue(name, "rx_errors")
	if err != nil {
		return stats, err
	}
	stats.RxDropped, err = sysFs.GetNetworkStatValue(name, "rx_dropped")
	if err != nil {
		return stats, err
	}
	stats.TxBytes, err = sysFs.GetNetworkStatValue(name, "tx_bytes")
	if err != nil {
		return stats, err
	}
	stats.TxPackets, err = sysFs.GetNetworkStatValue(name, "tx_packets")
	if err != nil {
		return stats, err
	}
	stats.TxErrors, err = sysFs.GetNetworkStatValue(name, "tx_errors")
	if err != nil {
		return stats, err
	}
	stats.TxDropped, err = sysFs.GetNetworkStatValue(name, "tx_dropped")
	if err != nil {
		return stats, err
	}
	return stats, nil
}

func GetSystemUUID(sysFs sysfs.SysFs) (string, error) {
	return sysFs.GetSystemUUID()
}

func getMatchedInt(rgx *regexp.Regexp, str string) (int, error) {
	matches := rgx.FindStringSubmatch(str)
	if len(matches) != 2 {
		return 0, fmt.Errorf("failed to match regexp, str: %s", str)
	}
	valInt, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, err
	}
	return valInt, nil
}

// GetSocketFromCPU returns Socket ID of passed CPU. If is not present, returns -1.
func GetSocketFromCPU(topology []info.Node, cpu int) int {
	for _, node := range topology {
		found, coreID := node.FindCoreByThread(cpu)
		if found {
			return node.Cores[coreID].SocketID
		}
	}
	return -1
}
