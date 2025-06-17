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
