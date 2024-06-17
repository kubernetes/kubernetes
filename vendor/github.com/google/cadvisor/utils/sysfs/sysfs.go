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

package sysfs

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

const (
	blockDir     = "/sys/block"
	cacheDir     = "/sys/devices/system/cpu/cpu"
	netDir       = "/sys/class/net"
	dmiDir       = "/sys/class/dmi"
	ppcDevTree   = "/proc/device-tree"
	s390xDevTree = "/etc" // s390/s390x changes

	meminfoFile = "meminfo"

	distanceFile = "distance"

	sysFsCPUTopology = "topology"

	// CPUPhysicalPackageID is a physical package id of cpu#. Typically corresponds to a physical socket number,
	// but the actual value is architecture and platform dependent.
	CPUPhysicalPackageID = "physical_package_id"
	// CPUCoreID is the CPU core ID of cpu#. Typically it is the hardware platform's identifier
	// (rather than the kernel's). The actual value is architecture and platform dependent.
	CPUCoreID = "core_id"

	coreIDFilePath    = "/" + sysFsCPUTopology + "/core_id"
	packageIDFilePath = "/" + sysFsCPUTopology + "/physical_package_id"

	// memory size calculations

	cpuDirPattern  = "cpu*[0-9]"
	nodeDirPattern = "node*[0-9]"

	//HugePagesNrFile name of nr_hugepages file in sysfs
	HugePagesNrFile = "nr_hugepages"
)

var (
	nodeDir = "/sys/devices/system/node/"
)

type CacheInfo struct {
	// cache id
	Id int
	// size in bytes
	Size uint64
	// cache type - instruction, data, unified
	Type string
	// distance from cpus in a multi-level hierarchy
	Level int
	// number of cpus that can access this cache.
	Cpus int
}

// Abstracts the lowest level calls to sysfs.
type SysFs interface {
	// Get NUMA nodes paths
	GetNodesPaths() ([]string, error)
	// Get paths to CPUs in provided directory e.g. /sys/devices/system/node/node0 or /sys/devices/system/cpu
	GetCPUsPaths(cpusPath string) ([]string, error)
	// Get physical core id for specified CPU
	GetCoreID(coreIDFilePath string) (string, error)
	// Get physical package id for specified CPU
	GetCPUPhysicalPackageID(cpuPath string) (string, error)
	// Get total memory for specified NUMA node
	GetMemInfo(nodeDir string) (string, error)
	// Get hugepages from specified directory
	GetHugePagesInfo(hugePagesDirectory string) ([]os.FileInfo, error)
	// Get hugepage_nr from specified directory
	GetHugePagesNr(hugePagesDirectory string, hugePageName string) (string, error)
	// Get directory information for available block devices.
	GetBlockDevices() ([]os.FileInfo, error)
	// Get Size of a given block device.
	GetBlockDeviceSize(string) (string, error)
	// Get scheduler type for the block device.
	GetBlockDeviceScheduler(string) (string, error)
	// Get device major:minor number string.
	GetBlockDeviceNumbers(string) (string, error)
	// Is the device "hidden" (meaning will not have a device handle)
	// This is the case with native nvme multipathing.
	IsBlockDeviceHidden(string) (bool, error)

	GetNetworkDevices() ([]os.FileInfo, error)
	GetNetworkAddress(string) (string, error)
	GetNetworkMtu(string) (string, error)
	GetNetworkSpeed(string) (string, error)
	GetNetworkStatValue(dev string, stat string) (uint64, error)

	// Get directory information for available caches accessible to given cpu.
	GetCaches(id int) ([]os.FileInfo, error)
	// Get information for a cache accessible from the given cpu.
	GetCacheInfo(cpu int, cache string) (CacheInfo, error)

	GetSystemUUID() (string, error)

	// GetDistances returns distance array
	GetDistances(string) (string, error)

	// IsCPUOnline determines if CPU status from kernel hotplug machanism standpoint.
	// See: https://www.kernel.org/doc/html/latest/core-api/cpu_hotplug.html
	IsCPUOnline(dir string) bool
}

type realSysFs struct {
	cpuPath string
}

func NewRealSysFs() SysFs {
	return &realSysFs{
		cpuPath: "/sys/devices/system/cpu",
	}
}

func (fs *realSysFs) GetNodesPaths() ([]string, error) {
	pathPattern := fmt.Sprintf("%s%s", nodeDir, nodeDirPattern)
	return filepath.Glob(pathPattern)
}

func (fs *realSysFs) GetCPUsPaths(cpusPath string) ([]string, error) {
	pathPattern := fmt.Sprintf("%s/%s", cpusPath, cpuDirPattern)
	return filepath.Glob(pathPattern)
}

func (fs *realSysFs) GetCoreID(cpuPath string) (string, error) {
	coreIDFilePath := fmt.Sprintf("%s%s", cpuPath, coreIDFilePath)
	coreID, err := os.ReadFile(coreIDFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(coreID)), err
}

func (fs *realSysFs) GetCPUPhysicalPackageID(cpuPath string) (string, error) {
	packageIDFilePath := fmt.Sprintf("%s%s", cpuPath, packageIDFilePath)
	packageID, err := os.ReadFile(packageIDFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(packageID)), err
}

func (fs *realSysFs) GetMemInfo(nodePath string) (string, error) {
	meminfoPath := fmt.Sprintf("%s/%s", nodePath, meminfoFile)
	meminfo, err := os.ReadFile(meminfoPath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(meminfo)), err
}

func (fs *realSysFs) GetDistances(nodePath string) (string, error) {
	distancePath := fmt.Sprintf("%s/%s", nodePath, distanceFile)
	distance, err := os.ReadFile(distancePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(distance)), err
}

func (fs *realSysFs) GetHugePagesInfo(hugePagesDirectory string) ([]os.FileInfo, error) {
	dirs, err := os.ReadDir(hugePagesDirectory)
	if err != nil {
		return nil, err
	}
	return toFileInfo(dirs)
}

func (fs *realSysFs) GetHugePagesNr(hugepagesDirectory string, hugePageName string) (string, error) {
	hugePageFilePath := fmt.Sprintf("%s%s/%s", hugepagesDirectory, hugePageName, HugePagesNrFile)
	hugePageFile, err := os.ReadFile(hugePageFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(hugePageFile)), err
}

func (fs *realSysFs) GetBlockDevices() ([]os.FileInfo, error) {
	dirs, err := os.ReadDir(blockDir)
	if err != nil {
		return nil, err
	}
	return toFileInfo(dirs)
}

func (fs *realSysFs) GetBlockDeviceNumbers(name string) (string, error) {
	dev, err := os.ReadFile(path.Join(blockDir, name, "/dev"))
	if err != nil {
		return "", err
	}
	return string(dev), nil
}

func (fs *realSysFs) IsBlockDeviceHidden(name string) (bool, error) {
	// See: https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-block
	//      https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git
	//        - c8487d854ba5 ("lsblk: Ignore hidden devices")
	devHiddenPath := path.Join(blockDir, name, "/hidden")
	hidden, err := os.ReadFile(devHiddenPath)
	if err != nil && os.IsNotExist(err) {
		// older OS may not have /hidden sysfs entry, so for sure
		// it is not a hidden device...
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to read %s: %w", devHiddenPath, err)
	}
	if string(hidden) == "1" {
		return true, nil
	}
	return false, nil
}

func (fs *realSysFs) GetBlockDeviceScheduler(name string) (string, error) {
	sched, err := os.ReadFile(path.Join(blockDir, name, "/queue/scheduler"))
	if err != nil {
		return "", err
	}
	return string(sched), nil
}

func (fs *realSysFs) GetBlockDeviceSize(name string) (string, error) {
	size, err := os.ReadFile(path.Join(blockDir, name, "/size"))
	if err != nil {
		return "", err
	}
	return string(size), nil
}

func (fs *realSysFs) GetNetworkDevices() ([]os.FileInfo, error) {
	dirs, err := os.ReadDir(netDir)
	if err != nil {
		return nil, err
	}
	files, err := toFileInfo(dirs)
	if err != nil {
		return nil, err
	}

	// Filter out non-directory & non-symlink files
	filtered := []os.FileInfo{}
	for _, f := range files {
		if f.Mode()|os.ModeSymlink != 0 {
			f, err = os.Stat(path.Join(netDir, f.Name()))
			if err != nil {
				continue
			}
		}
		if f.IsDir() {
			filtered = append(filtered, f)
		}
	}
	return filtered, nil
}

func (fs *realSysFs) GetNetworkAddress(name string) (string, error) {
	address, err := os.ReadFile(path.Join(netDir, name, "/address"))
	if err != nil {
		return "", err
	}
	return string(address), nil
}

func (fs *realSysFs) GetNetworkMtu(name string) (string, error) {
	mtu, err := os.ReadFile(path.Join(netDir, name, "/mtu"))
	if err != nil {
		return "", err
	}
	return string(mtu), nil
}

func (fs *realSysFs) GetNetworkSpeed(name string) (string, error) {
	speed, err := os.ReadFile(path.Join(netDir, name, "/speed"))
	if err != nil {
		return "", err
	}
	return string(speed), nil
}

func (fs *realSysFs) GetNetworkStatValue(dev string, stat string) (uint64, error) {
	statPath := path.Join(netDir, dev, "/statistics", stat)
	out, err := os.ReadFile(statPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read stat from %q for device %q", statPath, dev)
	}
	var s uint64
	n, err := fmt.Sscanf(string(out), "%d", &s)
	if err != nil || n != 1 {
		return 0, fmt.Errorf("could not parse value from %q for file %s", string(out), statPath)
	}
	return s, nil
}

func (fs *realSysFs) GetCaches(id int) ([]os.FileInfo, error) {
	cpuPath := fmt.Sprintf("%s%d/cache", cacheDir, id)
	dir, err := os.ReadDir(cpuPath)
	if err != nil {
		return nil, err
	}
	return toFileInfo(dir)
}

func toFileInfo(dirs []os.DirEntry) ([]os.FileInfo, error) {
	info := []os.FileInfo{}
	for _, dir := range dirs {
		fI, err := dir.Info()
		if err != nil {
			return nil, err
		}
		info = append(info, fI)
	}
	return info, nil
}

func bitCount(i uint64) (count int) {
	for i != 0 {
		if i&1 == 1 {
			count++
		}
		i >>= 1
	}
	return
}

func getCPUCount(cache string) (count int, err error) {
	out, err := os.ReadFile(path.Join(cache, "/shared_cpu_map"))
	if err != nil {
		return 0, err
	}
	masks := strings.Split(string(out), ",")
	for _, mask := range masks {
		// convert hex string to uint64
		m, err := strconv.ParseUint(strings.TrimSpace(mask), 16, 64)
		if err != nil {
			return 0, fmt.Errorf("failed to parse cpu map %q: %v", string(out), err)
		}
		count += bitCount(m)
	}
	return
}

func (fs *realSysFs) GetCacheInfo(cpu int, name string) (CacheInfo, error) {
	cachePath := fmt.Sprintf("%s%d/cache/%s", cacheDir, cpu, name)
	out, err := os.ReadFile(path.Join(cachePath, "/id"))
	if err != nil {
		return CacheInfo{}, err
	}
	var id int
	n, err := fmt.Sscanf(string(out), "%d", &id)
	if err != nil || n != 1 {
		return CacheInfo{}, err
	}

	out, err = os.ReadFile(path.Join(cachePath, "/size"))
	if err != nil {
		return CacheInfo{}, err
	}
	var size uint64
	n, err = fmt.Sscanf(string(out), "%dK", &size)
	if err != nil || n != 1 {
		return CacheInfo{}, err
	}
	// convert to bytes
	size = size * 1024
	out, err = os.ReadFile(path.Join(cachePath, "/level"))
	if err != nil {
		return CacheInfo{}, err
	}
	var level int
	n, err = fmt.Sscanf(string(out), "%d", &level)
	if err != nil || n != 1 {
		return CacheInfo{}, err
	}

	out, err = os.ReadFile(path.Join(cachePath, "/type"))
	if err != nil {
		return CacheInfo{}, err
	}
	cacheType := strings.TrimSpace(string(out))
	cpuCount, err := getCPUCount(cachePath)
	if err != nil {
		return CacheInfo{}, err
	}
	return CacheInfo{
		Id:    id,
		Size:  size,
		Level: level,
		Type:  cacheType,
		Cpus:  cpuCount,
	}, nil
}

func (fs *realSysFs) GetSystemUUID() (string, error) {
	if id, err := os.ReadFile(path.Join(dmiDir, "id", "product_uuid")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else if id, err = os.ReadFile(path.Join(ppcDevTree, "system-id")); err == nil {
		return strings.TrimSpace(strings.TrimRight(string(id), "\000")), nil
	} else if id, err = os.ReadFile(path.Join(ppcDevTree, "vm,uuid")); err == nil {
		return strings.TrimSpace(strings.TrimRight(string(id), "\000")), nil
	} else if id, err = os.ReadFile(path.Join(s390xDevTree, "machine-id")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else {
		return "", err
	}
}

func (fs *realSysFs) IsCPUOnline(cpuPath string) bool {
	cpuOnlinePath, err := filepath.Abs(fs.cpuPath + "/online")
	if err != nil {
		klog.V(1).Infof("Unable to get absolute path for %s", cpuPath)
		return false
	}

	// Quick check to determine if file exists: if it does not then kernel CPU hotplug is disabled and all CPUs are online.
	_, err = os.Stat(cpuOnlinePath)
	if err != nil && os.IsNotExist(err) {
		return true
	}
	if err != nil {
		klog.V(1).Infof("Unable to stat %s: %s", cpuOnlinePath, err)
	}

	cpuID, err := getCPUID(cpuPath)
	if err != nil {
		klog.V(1).Infof("Unable to get CPU ID from path %s: %s", cpuPath, err)
		return false
	}

	isOnline, err := isCPUOnline(cpuOnlinePath, cpuID)
	if err != nil {
		klog.V(1).Infof("Unable to get online CPUs list: %s", err)
		return false
	}
	return isOnline
}

func getCPUID(dir string) (uint16, error) {
	regex := regexp.MustCompile("cpu([0-9]+)")
	matches := regex.FindStringSubmatch(dir)
	if len(matches) == 2 {
		id, err := strconv.Atoi(matches[1])
		if err != nil {
			return 0, err
		}
		return uint16(id), nil
	}
	return 0, fmt.Errorf("can't get CPU ID from %s", dir)
}

// isCPUOnline is copied from github.com/opencontainers/runc/libcontainer/cgroups/fs and modified to suite cAdvisor
// needs as Apache 2.0 license allows.
// It parses CPU list (such as: 0,3-5,10) into a struct that allows to determine quickly if CPU or particular ID is online.
// see: https://github.com/opencontainers/runc/blob/ab27e12cebf148aa5d1ee3ad13d9fc7ae12bf0b6/libcontainer/cgroups/fs/cpuset.go#L45
func isCPUOnline(path string, cpuID uint16) (bool, error) {
	fileContent, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}
	if len(fileContent) == 0 {
		return false, fmt.Errorf("%s found to be empty", path)
	}

	cpuList := strings.TrimSpace(string(fileContent))
	for _, s := range strings.Split(cpuList, ",") {
		splitted := strings.SplitN(s, "-", 3)
		switch len(splitted) {
		case 3:
			return false, fmt.Errorf("invalid values in %s", path)
		case 2:
			min, err := strconv.ParseUint(splitted[0], 10, 16)
			if err != nil {
				return false, err
			}
			max, err := strconv.ParseUint(splitted[1], 10, 16)
			if err != nil {
				return false, err
			}
			if min > max {
				return false, fmt.Errorf("invalid values in %s", path)
			}
			// Return true, if the CPU under consideration is in the range of online CPUs.
			if cpuID >= uint16(min) && cpuID <= uint16(max) {
				return true, nil
			}
		case 1:
			value, err := strconv.ParseUint(s, 10, 16)
			if err != nil {
				return false, err
			}
			if uint16(value) == cpuID {
				return true, nil
			}
		}
	}

	return false, nil
}

// Looks for sysfs cpu path containing given CPU property, e.g. core_id or physical_package_id
// and returns number of unique values of given property, exemplary usage: getting number of CPU physical cores
func GetUniqueCPUPropertyCount(cpuAttributesPath string, propertyName string) int {
	absCPUAttributesPath, err := filepath.Abs(cpuAttributesPath)
	if err != nil {
		klog.Errorf("Cannot make %s absolute", cpuAttributesPath)
		return 0
	}
	pathPattern := absCPUAttributesPath + "/cpu*[0-9]"
	sysCPUPaths, err := filepath.Glob(pathPattern)
	if err != nil {
		klog.Errorf("Cannot find files matching pattern (pathPattern: %s),  number of unique %s set to 0", pathPattern, propertyName)
		return 0
	}
	cpuOnlinePath, err := filepath.Abs(cpuAttributesPath + "/online")
	if err != nil {
		klog.V(1).Infof("Unable to get absolute path for %s", cpuAttributesPath+"/../online")
		return 0
	}

	if err != nil {
		klog.V(1).Infof("Unable to get online CPUs list: %s", err)
		return 0
	}
	uniques := make(map[string]bool)
	for _, sysCPUPath := range sysCPUPaths {
		cpuID, err := getCPUID(sysCPUPath)
		if err != nil {
			klog.V(1).Infof("Unable to get CPU ID from path %s: %s", sysCPUPath, err)
			return 0
		}
		isOnline, err := isCPUOnline(cpuOnlinePath, cpuID)
		if err != nil && !os.IsNotExist(err) {
			klog.V(1).Infof("Unable to determine CPU online state: %s", err)
			continue
		}
		if !isOnline && !os.IsNotExist(err) {
			continue
		}
		propertyPath := filepath.Join(sysCPUPath, sysFsCPUTopology, propertyName)
		propertyVal, err := os.ReadFile(propertyPath)
		if err != nil {
			klog.Warningf("Cannot open %s, assuming 0 for %s of CPU %d", propertyPath, propertyName, cpuID)
			propertyVal = []byte("0")
		}
		packagePath := filepath.Join(sysCPUPath, sysFsCPUTopology, CPUPhysicalPackageID)
		packageVal, err := os.ReadFile(packagePath)
		if err != nil {
			klog.Warningf("Cannot open %s, assuming 0 %s of CPU %d", packagePath, CPUPhysicalPackageID, cpuID)
			packageVal = []byte("0")

		}
		uniques[fmt.Sprintf("%s_%s", bytes.TrimSpace(propertyVal), bytes.TrimSpace(packageVal))] = true
	}
	return len(uniques)
}
