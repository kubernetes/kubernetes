/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"github.com/spf13/pflag"

	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
)

// NodeIPAMControllerOptions holds the NodeIpamController options.
type NodeIPAMControllerOptions struct {
	*nodeipamconfig.NodeIPAMControllerConfiguration
}

// AddFlags adds flags related to NodeIpamController for controller manager to the specified FlagSet.
func (o *NodeIPAMControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ServiceCIDR, "service-cluster-ip-range", o.ServiceCIDR, "CIDR Range for Services in cluster. Requires --allocate-node-cidrs to be true")
	fs.Int32Var(&o.NodeCIDRMaskSize, "node-cidr-mask-size", o.NodeCIDRMaskSize, "Mask size for node cidr in cluster.")
}

// ApplyTo fills up NodeIpamController config with options.
func (o *NodeIPAMControllerOptions) ApplyTo(cfg *nodeipamconfig.NodeIPAMControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ServiceCIDR = o.ServiceCIDR
	cfg.NodeCIDRMaskSize = o.NodeCIDRMaskSize

	return nil
}

// Validate checks validation of NodeIPAMControllerOptions.
func (o *NodeIPAMControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
