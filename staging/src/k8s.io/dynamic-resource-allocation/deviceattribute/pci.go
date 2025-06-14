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
	"k8s.io/utils/ptr"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
	utils "k8s.io/dynamic-resource-allocation/utils"
)

var (
	qualifiedNamePCIeRoot = resourceapi.QualifiedName(resourceapi.StandardDeviceAttributePCIeRoot)
)

type StandardPCIDeviceAttributesOption func(args *StandardPCIDeviceAttributeArgs)

// WithPCIDeviceAddress sets the PCI address for the PCI device attributes.
func WithPCIDeviceAddress(pciAddress *utils.PCIAddress) StandardPCIDeviceAttributesOption {
	return func(args *StandardPCIDeviceAttributeArgs) {
		args.pciAddress = pciAddress
	}
}

// StandardPCIDeviceAttributeArgs holds the arguments for generating standard PCI device attributes.
type StandardPCIDeviceAttributeArgs struct {
	pciAddress *utils.PCIAddress
	sysfs      utils.Sysfs
}

// StandardPCIDeviceAttributes returns standard device attributes for a PCI device.
func StandardPCIDeviceAttributes(opts ...StandardPCIDeviceAttributesOption) (map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, error) {
	args := &StandardPCIDeviceAttributeArgs{
		sysfs: utils.NewSysfs(),
	}

	for _, opt := range opts {
		opt(args)
	}

	if args.pciAddress == nil {
		return nil, nil
	}

	attrs := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)

	pciRoot, err := args.pciAddress.ResolvePCIRoot(args.sysfs)
	if err != nil {
		return nil, err
	}

	attrs[qualifiedNamePCIeRoot] = resourceapi.DeviceAttribute{
		StringValue: ptr.To("pci" + pciRoot.String()),
	}

	return attrs, nil
}

// This is only for testing purposes.
func withSysfs(sysfs utils.Sysfs) StandardPCIDeviceAttributesOption {
	return func(args *StandardPCIDeviceAttributeArgs) {
		args.sysfs = sysfs
	}
}
