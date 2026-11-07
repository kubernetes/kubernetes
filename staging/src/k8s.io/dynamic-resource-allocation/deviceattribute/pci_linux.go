/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package deviceattribute

import (
	"fmt"
	"io/fs"
	"iter"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// GetPCIeRootAttributeByPCIBusID retrieves the PCIe Root Complex for a given PCI Bus ID.
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7".
//
// It returns a DeviceAttribute with the PCIe Root Complex information("pci<domain>:<bus>")
// as a string value or an error if the PCI Bus ID is invalid or the root complex cannot be determined.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func GetPCIeRootAttributeByPCIBusID(pciBusID string, mods ...MachineModifier) (DeviceAttribute, error) {
	var mc machine
	initDefaultMachine(&mc)
	for _, mod := range mods {
		mod(&mc)
	}

	if err := verifyPCIBDFFormat(pciBusID); err != nil {
		return DeviceAttribute{}, err
	}

	pcieRoot, err := resolvePCIeRoot(mc, pciBusID)
	if err != nil {
		return DeviceAttribute{}, fmt.Errorf("failed to resolve PCIe Root Complex for PCI Bus ID %s: %w", pciBusID, err)
	}

	attr := DeviceAttribute{
		Name:  StandardDeviceAttributePCIeRoot,
		Value: resourceapi.DeviceAttribute{StringValue: &pcieRoot},
	}

	return attr, nil
}

// GetPCIeRootAttributeMapFromCPUId retrieves the PCIe Root Complex information
// for each CPU ID in the system.
//
// It returns a map of CPU ID to DeviceAttribute, where each attribute encapsulates
// the associated PCIe Root Complexes as a sorted string list.
//
// API Design Note:
// A dedicated API for looking up PCIe Roots by a single CPU ID is intentionally not
// provided. While Linux sysfs exposes a PCI-device -> local-CPU relationship via
// /sys/bus/pci/devices/*/local_cpulist, there is no direct CPU -> PCI-device index.
// Since a reverse lookup always requires a full scan of all PCI devices, this
// function builds and returns the complete mapping in a single pass for efficiency.
//
// Feature Gate Requirement:
// String list attributes are an alpha feature. Enabling the "DRAListTypeAttributes"
// feature gate is required for this data to be processed correctly in the cluster.
func GetPCIeRootAttributeMapFromCPUId(mods ...MachineModifier) (map[int]DeviceAttribute, error) {
	var mc machine
	initDefaultMachine(&mc)
	for _, mod := range mods {
		mod(&mc)
	}

	// First, getting the set of all CPU IDs in the system
	// to validate the CPU IDs we find in "local_cpulist" files later.
	cpuIds, err := getCPUIds(mc)
	if err != nil {
		return nil, fmt.Errorf("failed to get CPU IDs: %w", err)
	}

	// Next, focusing on the PCI Bus IDs of all PCIe Bridges in the system,
	// because scanning only PCI Bridges would be enough to resolve the relationship
	// between CPU and PCIe Roots.
	pcieBridgeBusIDs, err := getPCIeBridgeBusIDs(mc)
	if err != nil {
		return nil, fmt.Errorf("failed to get PCIe bridges: %w", err)
	}

	cpuIDToPCIeRoots := map[int]sets.Set[string]{}
	pciDevicesDir := filepath.Join("bus", "pci", "devices")
	for _, pcieBridgeBusID := range pcieBridgeBusIDs {
		pcieRoot, err := resolvePCIeRoot(mc, pcieBridgeBusID)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve PCIe Root Complex for PCI Bus ID %s: %w", pcieBridgeBusID, err)
		}

		localCPUListPaths := filepath.Join(pciDevicesDir, pcieBridgeBusID, "local_cpulist")
		content, err := fs.ReadFile(mc.sysfs, localCPUListPaths)
		if err != nil {
			return nil, fmt.Errorf("failed to read local_cpulist at %s: %w", localCPUListPaths, err)
		}

		localCPUList, err := parseCPUList(string(content))
		if err != nil {
			return nil, fmt.Errorf("failed to parse CPU list %s at %s: %w", string(content), localCPUListPaths, err)
		}
		for cpuID := range localCPUList.Iter() {
			if !cpuIds.Has(cpuID) {
				return nil, fmt.Errorf("CPU ID %d from local_cpulist at %s is not in the CPU ID set", cpuID, localCPUListPaths)
			}
			if _, ok := cpuIDToPCIeRoots[cpuID]; !ok {
				cpuIDToPCIeRoots[cpuID] = sets.New[string]()
			}
			cpuIDToPCIeRoots[cpuID].Insert(pcieRoot)
		}
	}

	results := map[int]DeviceAttribute{}
	for cpuID, pcieRoots := range cpuIDToPCIeRoots {
		if pcieRoots.Len() > 0 {
			results[cpuID] = DeviceAttribute{
				Name:  StandardDeviceAttributePCIeRoot,
				Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice(sets.List(pcieRoots))},
			}
		}
	}

	return results, nil
}

// getCPUIds returns a set of all online/present CPU IDs by scanning the
// /sys/devices/system/cpu/ directory. It filters entries matching
// the 'cpu[0-9]+' pattern to identify individual processor cores.
func getCPUIds(mc machine) (sets.Set[int], error) {
	cpuDir := filepath.Join("devices", "system", "cpu")
	entries, err := fs.ReadDir(mc.sysfs, cpuDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read CPU directory: %w", err)
	}
	cpuIds := sets.New[int]()
	cpuDirEntryRegexp := regexp.MustCompile(`^cpu([0-9]+)$`)
	for _, entry := range entries {
		name := entry.Name()
		matches := cpuDirEntryRegexp.FindStringSubmatch(name)
		if matches == nil {
			continue
		}
		cpuIDStr := matches[1]
		id, err := strconv.Atoi(cpuIDStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse CPU ID from path %q: %w", name, err)
		}
		cpuIds.Insert(id)
	}
	return cpuIds, nil
}

// getPCIeBridgeBusIDs retrieves the PCI Bus IDs (BDF format) for all devices
// identified as PCI Bridges (Class Code 0x06).
//
// It scans /sys/bus/pci/devices/ to find bridges that define the PCIe hierarchy.
// By targeting the entire 0x06 class, it captures not only common Host (0x0600)
// and PCI-to-PCI (0x0604) bridges but also specialized interconnects like
// Non-Transparent Bridges (0x0609) and InfiniBand-to-PCI bridges (0x060a).
//
// This comprehensive scanning is critical for accurate CPU-to-PCIe-root
// mapping in non-standard hardware topologies.
//
// ref: https://wiki.osdev.org/PCI#Class_Codes
func getPCIeBridgeBusIDs(mc machine) ([]string, error) {
	pciDevicesDir := filepath.Join("bus", "pci", "devices")
	pcieDevicePaths, err := fs.ReadDir(mc.sysfs, pciDevicesDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read PCI devices directory: %w", err)
	}

	bridges := []string{}
	for _, pcieDevicePath := range pcieDevicePaths {
		pciBusID := pcieDevicePath.Name()
		if err := verifyPCIBDFFormat(pciBusID); err != nil {
			continue
		}

		classData, err := fs.ReadFile(mc.sysfs, filepath.Join(pciDevicesDir, pciBusID, "class"))
		if err != nil {
			return nil, fmt.Errorf("failed to read PCI class for PCI Bus ID %s: %w", pciBusID, err)
		}
		pciClassInfo := strings.TrimSpace(string(classData))
		if len(pciClassInfo) != 8 { // format: "0xCCSSpp" (Class, Subclass, Programming Interface)
			return nil, fmt.Errorf("invalid PCI Class data for PCI Bus ID %s: %q", pciBusID, pciClassInfo)
		}

		if strings.HasPrefix(pciClassInfo, "0x06") {
			bridges = append(bridges, pciBusID)
		}
	}

	return bridges, nil
}

// cpuList represents a collection of CPU IDs, supporting the fragmented
// range format common in Linux sysfs (e.g., "0-3,5,7-31").
type cpuList []cpuListPart

func (cl cpuList) Iter() iter.Seq[int] {
	return func(yield func(int) bool) {
		for _, part := range cl {
			for e := range part.Iter() {
				if !yield(e) {
					return
				}
			}
		}
	}
}

// cpuListPart represents a continuous range of CPU IDs,
// defined by a start and end ID (inclusive).
type cpuListPart struct {
	start int
	end   int
}

func (p cpuListPart) Iter() iter.Seq[int] {
	return func(yield func(int) bool) {
		for i := p.start; i <= p.end; i++ {
			if !yield(i) {
				return
			}
		}
	}
}

// parseCPUList converts a Linux CPU list string (e.g., from local_cpulist)
// into a structured cpuList. It handles both individual IDs and
// hyphenated ranges (inclusive).
func parseCPUList(cpuListStr string) (cpuList, error) {
	cpuListStr = strings.TrimSpace(cpuListStr)
	if cpuListStr == "" {
		return nil, fmt.Errorf("CPU list string is empty")
	}

	// The CPU list can be in the format of "0-3,5,7-9", so we need to parse it accordingly.
	parts := strings.Split(cpuListStr, ",")
	cpuList := make(cpuList, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("invalid CPU list format: empty part in %q", cpuListStr)
		}
		if strings.Contains(part, "-") {
			bounds := strings.Split(part, "-")
			if len(bounds) != 2 {
				return nil, fmt.Errorf("invalid CPU list format: %s", cpuListStr)
			}
			start, err := strconv.Atoi(bounds[0])
			if err != nil {
				return nil, fmt.Errorf("invalid CPU ID in list: %s", bounds[0])
			}
			end, err := strconv.Atoi(bounds[1])
			if err != nil {
				return nil, fmt.Errorf("invalid CPU ID in list: %s", bounds[1])
			}
			if start > end {
				return nil, fmt.Errorf("invalid CPU range: %d-%d", start, end)
			}
			cpuList = append(cpuList, cpuListPart{start: start, end: end})
		} else {
			cpuID, err := strconv.Atoi(part)
			if err != nil {
				return nil, fmt.Errorf("invalid CPU ID in list: %s", part)
			}
			cpuList = append(cpuList, cpuListPart{start: cpuID, end: cpuID})
		}
	}
	return cpuList, nil
}

// resolvePCIeRoot resolves the PCIe Root for a given PCI Bus ID
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7",
// by inspecting sysfs(/sys/devices).
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
//
// /sys/devices has a directory structure which reflects the hardware hierarchy in the system.
// Therefore, the device path may contain intermediate directories (devices).
// Thus, we can not simply find the device path from the PCI Bus ID.
// But, fortunately, /sys/bus/pci/devices/<address> is a symlink to the actual device path in /sys/devices.
// So we can resolve the actual device path by reading the symlink at /sys/bus/pci/devices/<address>.
//
// For example, if the PCIAddress is "0000:04:1f.0",
// /sys/bus/pci/devices/0000:04:1f.0 points to
// /sys/devices/pci0000:00/...<intermediate PCI devices>.../0000:04:1f.0,
// where "pci0000:00" is the PCIe Root.
func resolvePCIeRoot(mc machine, pciBusID string) (string, error) {
	// e.g. /sys/bus/pci/devices/0000:04:1f.0
	sysBusPath := filepath.Join("bus", "pci", "devices", pciBusID)

	target, err := fs.ReadLink(mc.sysfs, sysBusPath)
	if err != nil {
		return "", fmt.Errorf("failed to read symlink for PCI Bus ID %s: %w", sysBusPath, err)
	}

	// If the target is a relative path, we need to resolve it relative to the symlink's directory.
	if !filepath.IsAbs(target) {
		target = filepath.Clean(filepath.Join(filepath.Dir(sysBusPath), target))
	}

	// Once the symlink was resolved, we must have a path like /sys/devices/pci0000:00/...<intermediate PCI devices>.../0000:04:1f.0
	devicePathPrefix := filepath.Join("devices", "pci")
	if !strings.HasPrefix(target, devicePathPrefix) {
		return "", fmt.Errorf("symlink target for PCI Bus ID %s is invalid: it must start with %s: %s", pciBusID, devicePathPrefix, target)
	}
	if filepath.Base(target) != pciBusID {
		return "", fmt.Errorf("symlink target for PCI Bus ID %s is invalid: it must end with %s: %s", pciBusID, pciBusID, target)
	}

	// We need to extract the PCIe Root part, which is the first part of the path after /sys/devices/.
	target = strings.TrimPrefix(target, "devices"+string(filepath.Separator))
	pcieRootParts := strings.Split(target, string(filepath.Separator))
	return pcieRootParts[0], nil
}

// GetPCIBusIDAttribute returns a DeviceAttribute with the PCI Bus Address("<domain>:<bus>:<device>.<function>")
// of a PCI device as a string value.
//
// It returns an error if the PCI Bus ID is empty or is not in
// extended BDF (Domain:Bus:Device.Function) format, e.g., "0123:45:1e.7"
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func GetPCIBusIDAttribute(pciBusID string) (DeviceAttribute, error) {
	if err := verifyPCIBDFFormat(pciBusID); err != nil {
		return DeviceAttribute{}, err
	}

	attr := DeviceAttribute{
		Name:  StandardDeviceAttributePCIBusID,
		Value: resourceapi.DeviceAttribute{StringValue: &pciBusID},
	}

	return attr, nil
}

// verifyPCIBDFFormat verifies that the PCI Bus ID is in extended BDF (Domain:Bus:Device.Function) format.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func verifyPCIBDFFormat(pciBusID string) error {
	if pciBusID == "" {
		return fmt.Errorf("PCI Bus ID cannot be empty")
	}

	bdfRegexp := regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f]{1})$`)
	if !bdfRegexp.MatchString(pciBusID) {
		return fmt.Errorf("invalid PCI Bus ID format: %s", pciBusID)
	}

	return nil
}
