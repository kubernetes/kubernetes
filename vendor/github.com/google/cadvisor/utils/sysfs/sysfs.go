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
	"errors"
	"fmt"
	"io/ioutil"
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

	coreIDFilePath    = "/topology/core_id"
	packageIDFilePath = "/topology/physical_package_id"
	meminfoFile       = "meminfo"

	cpuDirPattern  = "cpu*[0-9]"
	nodeDirPattern = "node*[0-9]"

	//HugePagesNrFile name of nr_hugepages file in sysfs
	HugePagesNrFile = "nr_hugepages"
)

var (
	nodeDir = "/sys/devices/system/node/"
)

type CacheInfo struct {
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
	// IsCPUOnline determines if CPU status from kernel hotplug machanism standpoint.
	// See: https://www.kernel.org/doc/html/latest/core-api/cpu_hotplug.html
	IsCPUOnline(dir string) bool
}

type realSysFs struct{}

func NewRealSysFs() SysFs {
	return &realSysFs{}
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
	coreID, err := ioutil.ReadFile(coreIDFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(coreID)), err
}

func (fs *realSysFs) GetCPUPhysicalPackageID(cpuPath string) (string, error) {
	packageIDFilePath := fmt.Sprintf("%s%s", cpuPath, packageIDFilePath)
	packageID, err := ioutil.ReadFile(packageIDFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(packageID)), err
}

func (fs *realSysFs) GetMemInfo(nodePath string) (string, error) {
	meminfoPath := fmt.Sprintf("%s/%s", nodePath, meminfoFile)
	meminfo, err := ioutil.ReadFile(meminfoPath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(meminfo)), err
}

func (fs *realSysFs) GetHugePagesInfo(hugePagesDirectory string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(hugePagesDirectory)
}

func (fs *realSysFs) GetHugePagesNr(hugepagesDirectory string, hugePageName string) (string, error) {
	hugePageFilePath := fmt.Sprintf("%s%s/%s", hugepagesDirectory, hugePageName, HugePagesNrFile)
	hugePageFile, err := ioutil.ReadFile(hugePageFilePath)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(hugePageFile)), err
}

func (fs *realSysFs) GetBlockDevices() ([]os.FileInfo, error) {
	return ioutil.ReadDir(blockDir)
}

func (fs *realSysFs) GetBlockDeviceNumbers(name string) (string, error) {
	dev, err := ioutil.ReadFile(path.Join(blockDir, name, "/dev"))
	if err != nil {
		return "", err
	}
	return string(dev), nil
}

func (fs *realSysFs) GetBlockDeviceScheduler(name string) (string, error) {
	sched, err := ioutil.ReadFile(path.Join(blockDir, name, "/queue/scheduler"))
	if err != nil {
		return "", err
	}
	return string(sched), nil
}

func (fs *realSysFs) GetBlockDeviceSize(name string) (string, error) {
	size, err := ioutil.ReadFile(path.Join(blockDir, name, "/size"))
	if err != nil {
		return "", err
	}
	return string(size), nil
}

func (fs *realSysFs) GetNetworkDevices() ([]os.FileInfo, error) {
	files, err := ioutil.ReadDir(netDir)
	if err != nil {
		return nil, err
	}

	// Filter out non-directory & non-symlink files
	var dirs []os.FileInfo
	for _, f := range files {
		if f.Mode()|os.ModeSymlink != 0 {
			f, err = os.Stat(path.Join(netDir, f.Name()))
			if err != nil {
				continue
			}
		}
		if f.IsDir() {
			dirs = append(dirs, f)
		}
	}
	return dirs, nil
}

func (fs *realSysFs) GetNetworkAddress(name string) (string, error) {
	address, err := ioutil.ReadFile(path.Join(netDir, name, "/address"))
	if err != nil {
		return "", err
	}
	return string(address), nil
}

func (fs *realSysFs) GetNetworkMtu(name string) (string, error) {
	mtu, err := ioutil.ReadFile(path.Join(netDir, name, "/mtu"))
	if err != nil {
		return "", err
	}
	return string(mtu), nil
}

func (fs *realSysFs) GetNetworkSpeed(name string) (string, error) {
	speed, err := ioutil.ReadFile(path.Join(netDir, name, "/speed"))
	if err != nil {
		return "", err
	}
	return string(speed), nil
}

func (fs *realSysFs) GetNetworkStatValue(dev string, stat string) (uint64, error) {
	statPath := path.Join(netDir, dev, "/statistics", stat)
	out, err := ioutil.ReadFile(statPath)
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
	return ioutil.ReadDir(cpuPath)
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
	out, err := ioutil.ReadFile(path.Join(cache, "/shared_cpu_map"))
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

func (fs *realSysFs) GetCacheInfo(id int, name string) (CacheInfo, error) {
	cachePath := fmt.Sprintf("%s%d/cache/%s", cacheDir, id, name)
	out, err := ioutil.ReadFile(path.Join(cachePath, "/size"))
	if err != nil {
		return CacheInfo{}, err
	}
	var size uint64
	n, err := fmt.Sscanf(string(out), "%dK", &size)
	if err != nil || n != 1 {
		return CacheInfo{}, err
	}
	// convert to bytes
	size = size * 1024
	out, err = ioutil.ReadFile(path.Join(cachePath, "/level"))
	if err != nil {
		return CacheInfo{}, err
	}
	var level int
	n, err = fmt.Sscanf(string(out), "%d", &level)
	if err != nil || n != 1 {
		return CacheInfo{}, err
	}

	out, err = ioutil.ReadFile(path.Join(cachePath, "/type"))
	if err != nil {
		return CacheInfo{}, err
	}
	cacheType := strings.TrimSpace(string(out))
	cpuCount, err := getCPUCount(cachePath)
	if err != nil {
		return CacheInfo{}, err
	}
	return CacheInfo{
		Size:  size,
		Level: level,
		Type:  cacheType,
		Cpus:  cpuCount,
	}, nil
}

func (fs *realSysFs) GetSystemUUID() (string, error) {
	if id, err := ioutil.ReadFile(path.Join(dmiDir, "id", "product_uuid")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else if id, err = ioutil.ReadFile(path.Join(ppcDevTree, "system-id")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else if id, err = ioutil.ReadFile(path.Join(ppcDevTree, "vm,uuid")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else if id, err = ioutil.ReadFile(path.Join(s390xDevTree, "machine-id")); err == nil {
		return strings.TrimSpace(string(id)), nil
	} else {
		return "", err
	}
}

func (fs *realSysFs) IsCPUOnline(dir string) bool {
	cpuPath := fmt.Sprintf("%s/online", dir)
	content, err := ioutil.ReadFile(cpuPath)
	if err != nil {
		pathErr, ok := err.(*os.PathError)
		if ok {
			if errors.Is(pathErr.Unwrap(), os.ErrNotExist) && isZeroCPU(dir) {
				return true
			}
		}
		klog.Warningf("unable to read %s: %s", cpuPath, err.Error())
		return false
	}
	trimmed := bytes.TrimSpace(content)
	return len(trimmed) == 1 && trimmed[0] == 49
}

func isZeroCPU(dir string) bool {
	regex := regexp.MustCompile("cpu([0-9]*)")
	matches := regex.FindStringSubmatch(dir)
	return len(matches) == 2 && matches[1] == "0"
}
