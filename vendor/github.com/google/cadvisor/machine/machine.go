// Copyright 2015 Google Inc. All Rights Reserved.
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

// The machine package contains functions that extract machine-level specs.
package machine

import (
	"fmt"
	"os"
	"path"
	"regexp"

	"strconv"
	"strings"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/utils/sysinfo"

	"k8s.io/klog/v2"

	"golang.org/x/sys/unix"
)

var (
	coreRegExp = regexp.MustCompile(`(?m)^core id\s*:\s*([0-9]+)$`)
	nodeRegExp = regexp.MustCompile(`(?m)^physical id\s*:\s*([0-9]+)$`)
	// Power systems have a different format so cater for both
	cpuClockSpeedMHz     = regexp.MustCompile(`(?:cpu MHz|CPU MHz|clock)\s*:\s*([0-9]+\.[0-9]+)(?:MHz)?`)
	memoryCapacityRegexp = regexp.MustCompile(`MemTotal:\s*([0-9]+) kB`)
	swapCapacityRegexp   = regexp.MustCompile(`SwapTotal:\s*([0-9]+) kB`)
	vendorIDRegexp       = regexp.MustCompile(`vendor_id\s*:\s*(\w+)`)

	cpuAttributesPath  = "/sys/devices/system/cpu/"
	isMemoryController = regexp.MustCompile("mc[0-9]+")
	isDimm             = regexp.MustCompile("dimm[0-9]+")
	machineArch        = getMachineArch()
	maxFreqFile        = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
)

const memTypeFileName = "dimm_mem_type"
const sizeFileName = "size"

// GetCPUVendorID returns "vendor_id" reading /proc/cpuinfo file.
func GetCPUVendorID(procInfo []byte) string {
	vendorID := ""

	matches := vendorIDRegexp.FindSubmatch(procInfo)
	if len(matches) != 2 {
		klog.V(4).Info("Cannot read vendor id correctly, set empty.")
		return vendorID
	}

	vendorID = string(matches[1])

	return vendorID
}

// GetPhysicalCores returns number of CPU cores reading /proc/cpuinfo file or if needed information from sysfs cpu path
func GetPhysicalCores(procInfo []byte) int {
	numCores := getUniqueMatchesCount(string(procInfo), coreRegExp)
	if numCores == 0 {
		// read number of cores from /sys/bus/cpu/devices/cpu*/topology/core_id to deal with processors
		// for which 'core id' is not available in /proc/cpuinfo
		numCores = sysfs.GetUniqueCPUPropertyCount(cpuAttributesPath, sysfs.CPUCoreID)
	}
	if numCores == 0 {
		klog.Errorf("Cannot read number of physical cores correctly, number of cores set to %d", numCores)
	}
	return numCores
}

// GetSockets returns number of CPU sockets reading /proc/cpuinfo file or if needed information from sysfs cpu path
func GetSockets(procInfo []byte) int {
	numSocket := getUniqueMatchesCount(string(procInfo), nodeRegExp)
	if numSocket == 0 {
		// read number of sockets from /sys/bus/cpu/devices/cpu*/topology/physical_package_id to deal with processors
		// for which 'physical id' is not available in /proc/cpuinfo
		numSocket = sysfs.GetUniqueCPUPropertyCount(cpuAttributesPath, sysfs.CPUPhysicalPackageID)
	}
	if numSocket == 0 {
		klog.Errorf("Cannot read number of sockets correctly, number of sockets set to %d", numSocket)
	}
	return numSocket
}

// GetClockSpeed returns the CPU clock speed, given a []byte formatted as the /proc/cpuinfo file.
func GetClockSpeed(procInfo []byte) (uint64, error) {
	// First look through sys to find a max supported cpu frequency.
	if utils.FileExists(maxFreqFile) {
		val, err := os.ReadFile(maxFreqFile)
		if err != nil {
			return 0, err
		}
		var maxFreq uint64
		n, err := fmt.Sscanf(string(val), "%d", &maxFreq)
		if err != nil || n != 1 {
			return 0, fmt.Errorf("could not parse frequency %q", val)
		}
		return maxFreq, nil
	}
	// s390/s390x, mips64, riscv64, aarch64 and arm32 changes
	if isMips64() || isSystemZ() || isAArch64() || isArm32() || isRiscv64() {
		return 0, nil
	}

	// Fall back to /proc/cpuinfo
	matches := cpuClockSpeedMHz.FindSubmatch(procInfo)
	if len(matches) != 2 {
		return 0, fmt.Errorf("could not detect clock speed from output: %q", string(procInfo))
	}

	speed, err := strconv.ParseFloat(string(matches[1]), 64)
	if err != nil {
		return 0, err
	}
	// Convert to kHz
	return uint64(speed * 1000), nil
}

// GetMachineMemoryCapacity returns the machine's total memory from /proc/meminfo.
// Returns the total memory capacity as an uint64 (number of bytes).
func GetMachineMemoryCapacity() (uint64, error) {
	out, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}

	memoryCapacity, err := parseCapacity(out, memoryCapacityRegexp)
	if err != nil {
		return 0, err
	}
	return memoryCapacity, err
}

// GetMachineMemoryByType returns information about memory capacity and number of DIMMs.
// Information is retrieved from sysfs edac per-DIMM API (/sys/devices/system/edac/mc/)
// introduced in kernel 3.6. Documentation can be found at
// https://www.kernel.org/doc/Documentation/admin-guide/ras.rst.
// Full list of memory types can be found in edac_mc.c
// (https://github.com/torvalds/linux/blob/v5.5/drivers/edac/edac_mc.c#L198)
func GetMachineMemoryByType(edacPath string) (map[string]*info.MemoryInfo, error) {
	memory := map[string]*info.MemoryInfo{}
	names, err := os.ReadDir(edacPath)
	// On some architectures (such as ARM) memory controller device may not exist.
	// If this is the case then we ignore error and return empty slice.
	_, ok := err.(*os.PathError)
	if err != nil && ok {
		return memory, nil
	} else if err != nil {
		return memory, err
	}
	for _, controllerDir := range names {
		controller := controllerDir.Name()
		if !isMemoryController.MatchString(controller) {
			continue
		}
		dimms, err := os.ReadDir(path.Join(edacPath, controllerDir.Name()))
		if err != nil {
			return map[string]*info.MemoryInfo{}, err
		}
		for _, dimmDir := range dimms {
			dimm := dimmDir.Name()
			if !isDimm.MatchString(dimm) {
				continue
			}
			memType, err := os.ReadFile(path.Join(edacPath, controller, dimm, memTypeFileName))
			if err != nil {
				return map[string]*info.MemoryInfo{}, err
			}
			readableMemType := strings.TrimSpace(string(memType))
			if _, exists := memory[readableMemType]; !exists {
				memory[readableMemType] = &info.MemoryInfo{}
			}
			size, err := os.ReadFile(path.Join(edacPath, controller, dimm, sizeFileName))
			if err != nil {
				return map[string]*info.MemoryInfo{}, err
			}
			capacity, err := strconv.Atoi(strings.TrimSpace(string(size)))
			if err != nil {
				return map[string]*info.MemoryInfo{}, err
			}
			memory[readableMemType].Capacity += uint64(mbToBytes(capacity))
			memory[readableMemType].DimmCount++
		}
	}

	return memory, nil
}

func mbToBytes(megabytes int) int {
	return megabytes * 1024 * 1024
}

// GetMachineSwapCapacity returns the machine's total swap from /proc/meminfo.
// Returns the total swap capacity as an uint64 (number of bytes).
func GetMachineSwapCapacity() (uint64, error) {
	out, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}

	swapCapacity, err := parseCapacity(out, swapCapacityRegexp)
	if err != nil {
		return 0, err
	}
	return swapCapacity, err
}

// GetTopology returns CPU topology reading information from sysfs
func GetTopology(sysFs sysfs.SysFs) ([]info.Node, int, error) {
	return sysinfo.GetNodesInfo(sysFs)
}

// parseCapacity matches a Regexp in a []byte, returning the resulting value in bytes.
// Assumes that the value matched by the Regexp is in KB.
func parseCapacity(b []byte, r *regexp.Regexp) (uint64, error) {
	matches := r.FindSubmatch(b)
	if len(matches) != 2 {
		return 0, fmt.Errorf("failed to match regexp in output: %q", string(b))
	}
	m, err := strconv.ParseUint(string(matches[1]), 10, 64)
	if err != nil {
		return 0, err
	}

	// Convert to bytes.
	return m * 1024, err
}

// getUniqueMatchesCount returns number of unique matches in given argument using provided regular expression
func getUniqueMatchesCount(s string, r *regexp.Regexp) int {
	matches := r.FindAllString(s, -1)
	uniques := make(map[string]bool)
	for _, match := range matches {
		uniques[match] = true
	}
	return len(uniques)
}

func getMachineArch() string {
	uname := unix.Utsname{}
	err := unix.Uname(&uname)
	if err != nil {
		klog.Errorf("Cannot get machine architecture, err: %v", err)
		return ""
	}
	return string(uname.Machine[:])
}

// arm32 changes
func isArm32() bool {
	return strings.Contains(machineArch, "arm")
}

// aarch64 changes
func isAArch64() bool {
	return strings.Contains(machineArch, "aarch64")
}

// s390/s390x changes
func isSystemZ() bool {
	return strings.Contains(machineArch, "390")
}

// riscv64 changes
func isRiscv64() bool {
	return strings.Contains(machineArch, "riscv64")
}

// mips64 changes
func isMips64() bool {
	return strings.Contains(machineArch, "mips64")
}
