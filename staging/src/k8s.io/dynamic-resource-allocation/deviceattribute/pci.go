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
	"strings"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
)

var (
	bdfRegexp = regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f]{1})$`)

	qualifiedNamePCIeRoot = resourceapi.QualifiedName(resourceapi.StandardDeviceAttributePCIeRoot)
)

type StandardPCIDeviceAttributesOption func(args *StandardPCIDeviceAttributeArgs)

// WithPCIDeviceAddress sets the PCI address for the PCI Bus ID
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7".
//
// It returns a DeviceAttribute with the PCIe Root Complex information("pci<domain>:<bus>")
// as a string value or an error if the PCI Bus ID is invalid or the root complex cannot be determined.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func WithPCIDeviceAddress(pciBusID string) StandardPCIDeviceAttributesOption {
	return func(args *StandardPCIDeviceAttributeArgs) {
		args.pciAddress = pciBusID
	}
}

// StandardPCIDeviceAttributeArgs holds the arguments for generating standard PCI device attributes.
type StandardPCIDeviceAttributeArgs struct {
	pciAddress string
	sysfs      sysfs
}

// StandardPCIDeviceAttributes returns standard device attributes for a PCI device.
func StandardPCIDeviceAttributes(opts ...StandardPCIDeviceAttributesOption) (map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, error) {
	args := &StandardPCIDeviceAttributeArgs{
		sysfs: sysfs(defaultSysfsRoot),
	}
	for _, opt := range opts {
		opt(args)
	}

	attrs := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)

	pciRootAttribute, err := getPCIeRootAttributeByPCIBusID(args.pciAddress, args.sysfs)
	if err != nil {
		return nil, err
	}
	attrs[qualifiedNamePCIeRoot] = pciRootAttribute

	return attrs, nil
}

// This is only for testing purposes.
func withSysfs(sysfs sysfs) StandardPCIDeviceAttributesOption {
	return func(args *StandardPCIDeviceAttributeArgs) {
		args.sysfs = sysfs
	}
}

// GetPCIeRootAttributeByPCI Bus ID retrieves the PCIe Root Complex for a given PCI Bus ID.
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7".
//
// It returns a DeviceAttribute with the PCIe Root Complex information("pci<domain>:<bus>")
// as a string value or an error if the PCI Bus ID is invalid or the root complex cannot be determined.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func getPCIeRootAttributeByPCIBusID(pciBusID string, sysfs sysfs) (resourceapi.DeviceAttribute, error) {
	if pciBusID == "" {
		return resourceapi.DeviceAttribute{}, fmt.Errorf("PCI Bus ID cannot be empty")
	}

	if !bdfRegexp.MatchString(pciBusID) {
		return resourceapi.DeviceAttribute{}, fmt.Errorf("invalid PCI Bus ID format: %s", pciBusID)
	}

	// e.g. /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0,
	sysDevicesPath, err := resolveSysDevicesPath(pciBusID, sysfs)
	if err != nil {
		return resourceapi.DeviceAttribute{}, fmt.Errorf("failed to resolve sysfs path for PCI Bus ID %s: %w", pciBusID, err)
	}

	pciRootPart := strings.Split(strings.TrimPrefix(sysDevicesPath, sysfs.Devices("")+"/"), "/")[0]

	return resourceapi.DeviceAttribute{
		StringValue: &pciRootPart,
	}, nil
}

// resolveSysDevicesPath resolves the /sys/devices path from the PCI Bus ID
// in BDF (Bus-Device-Function) format, e.g., "0123:45:1e.7".
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
//
// /sys/devices has directory structure which reflects the hardware hierarchy in the system.
// Therefore, the device path may contains intermediate directories(devices).
// Thus, we can not simply find the device path from the PCIAddress.
// Fortunately, /sys/bus/pci/devices/<address> is a symlink to the actual device path in /sys/devices.
// So we can resolve the actual device path by reading the symlink at /sys/bus/pci/devices/<address>.
//
// For example, if the PCIAddress is "0000:00:1f.0",
// /sys/bus/pci/devices/0000:00:1f.0 points to
// /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0,
func resolveSysDevicesPath(pciBusID string, sysfs sysfs) (string, error) {
	// e.g. /sys/bus/pci/devices/0000:00:1f.0
	sysBusPath := sysfs.Bus(filepath.Join("pci", "devices", pciBusID))

	targetRelative, err := os.Readlink(sysBusPath)
	if err != nil {
		return "", fmt.Errorf("failed to read symlink for PCI Bus ID %s: %w", sysBusPath, err)
	}
	var targetAbs string
	if filepath.IsAbs(targetRelative) {
		targetAbs = targetRelative
	} else {
		// If the target is a relative path, we need to resolve it relative to the symlink's directory.
		targetAbs = filepath.Join(filepath.Dir(sysBusPath), targetRelative)
	}

	// targetAbs must be /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0
	devicePathPrefix := sysfs.Devices("pci")
	if !strings.HasPrefix(targetAbs, devicePathPrefix) || filepath.Base(targetAbs) != pciBusID {
		return "", fmt.Errorf("invalid symlink target for PCI Bus ID %s: %s", sysBusPath, targetAbs)
	}

	return targetAbs, nil
}
