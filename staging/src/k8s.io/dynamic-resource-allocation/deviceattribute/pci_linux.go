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
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
)

var (
	pcieRootRegexp = regexp.MustCompile(`^pci[0-9a-f]{4}:[0-9a-f]{2}$`)
)

// GetPCIeRootAttributeByPCIBusID retrieves the PCIe Root Complex for a given PCI Bus ID.
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7".
//
// It returns a DeviceAttribute with the PCIe Root Complex information("pci<domain>:<bus>")
// as a string value or an error if the PCI Bus ID is invalid or the root complex cannot be determined.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func GetPCIeRootAttributeByPCIBusID(pciBusID string) (DeviceAttribute, error) {
	if pciBusID == "" {
		return DeviceAttribute{}, fmt.Errorf("PCI Bus ID cannot be empty")
	}

	bdfRegexp := regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f]{1})$`)
	if !bdfRegexp.MatchString(pciBusID) {
		return DeviceAttribute{}, fmt.Errorf("invalid PCI Bus ID format: %s", pciBusID)
	}

	pcieRoot, err := resolvePCIeRoot(pciBusID)
	if err != nil {
		return DeviceAttribute{}, fmt.Errorf("failed to resolve PCIe Root Complex for PCI Bus ID %s: %w", pciBusID, err)
	}

	attr := DeviceAttribute{
		Name:  StandardDeviceAttributePCIeRoot,
		Value: resourceapi.DeviceAttribute{StringValue: &pcieRoot},
	}

	return attr, nil
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
//
// In typical physical machines, /sys/bus/pci/devices/0000:04:1f.0 points to
// /sys/devices/pci0000:00/0000:00:1c.0/0000:04:00.0/0000:04:1f.0,
// where "pci0000:00" is the PCIe Root.
//
// In some virtualized environments(e.g. Hyper-V), they have more complex device hierarchies such that
// pcie devices are behind some virtual bus controllers(e.g. Hyper-V). In such cases,
// /sys/bus/pci/devices/0000:04:1f.0 points to
// /sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/000000c1-0001-0000-3130-444532304235/pci0000:00/0000:04:1f.0
// where "pci0000:00" is the PCIe Root.
func resolvePCIeRoot(pciBusID string) (string, error) {
	// e.g. /sys/bus/pci/devices/0000:04:1f.0
	sysBusPath := sysfs.bus(filepath.Join("pci", "devices", pciBusID))

	target, err := os.Readlink(sysBusPath)
	if err != nil {
		return "", fmt.Errorf("failed to read symlink for PCI Bus ID %s: %w", sysBusPath, err)
	}

	// If the target is a relative path, we need to resolve it relative to the symlink's directory.
	if !filepath.IsAbs(target) {
		target = filepath.Join(filepath.Dir(sysBusPath), target)
	}

	// targetAbs must be of the form /sys/devices/...<intermediate bus controllers>.../pci0000:00/...<intermediate PCI devices>.../0000:04:1f.0
	devicePathPrefix := sysfs.devices("")
	if !strings.HasPrefix(target, devicePathPrefix) {
		return "", fmt.Errorf("symlink target for PCI Bus ID %s is invalid: it must start with %s: %s", pciBusID, devicePathPrefix, target)
	}
	if filepath.Base(target) != pciBusID {
		return "", fmt.Errorf("symlink target for PCI Bus ID %s is invalid: it must end with %s: %s", pciBusID, pciBusID, target)
	}

	// Extract the PCIe Root part(pciXXXX:YY), which is of the form pci<domain>:<bus> from the absolute target path.
	targetAbsParts := strings.Split(strings.TrimPrefix(target, devicePathPrefix+"/"), "/")
	pcieRootPartIdx := slices.IndexFunc(targetAbsParts, func(part string) bool { return pcieRootRegexp.MatchString(part) })
	if pcieRootPartIdx == -1 {
		return "", fmt.Errorf("failed to find PCIe Root Complex part (pciXXXX:YY) in the device path for PCI Bus ID %s: %s", pciBusID, target)
	}

	return targetAbsParts[pcieRootPartIdx], nil
}
