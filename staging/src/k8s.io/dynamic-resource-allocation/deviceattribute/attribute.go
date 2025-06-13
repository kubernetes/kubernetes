package deviceattribute

import resourceapi "k8s.io/dynamic-resource-allocation/api"

type StandardDeviceAttributesOption func(args *StandardDeviceAttributesOptions)

// WithStandardPCIDeviceAttributesOpts sets the standard PCI device attributes options
func WithStandardPCIDeviceAttributesOpts(opts ...StandardPCIDeviceAttributesOption) StandardDeviceAttributesOption {
	return func(args *StandardDeviceAttributesOptions) {
		args.pciDeviceAttributeOpts = append(args.pciDeviceAttributeOpts, opts...)
	}
}

type StandardDeviceAttributesOptions struct {
	pciDeviceAttributeOpts []StandardPCIDeviceAttributesOption
}

// StandardDeviceAttributes generates standard device attributes based on the provided options.
// It currently supports standard PCI device attributes if options are provided.
func StandardDeviceAttributes(
	opts ...StandardDeviceAttributesOption,
) (map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, error) {
	options := &StandardDeviceAttributesOptions{}
	for _, opt := range opts {
		opt(options)
	}

	attrs := map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}

	if len(options.pciDeviceAttributeOpts) > 0 {
		// Standard PCI device attributes
		pciAttrs, err := StandardPCIDeviceAttributes(options.pciDeviceAttributeOpts...)
		if err != nil {
			return nil, err
		}
		for k, v := range pciAttrs {
			attrs[k] = v
		}
	}

	return attrs, nil
}
