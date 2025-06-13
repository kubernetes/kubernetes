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
