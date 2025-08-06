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

package nodecapabilities

import (
	"fmt"

	"k8s.io/component-helpers/nodecapabilities"
)

// This file contains the list of well-known node capabilities
// and the central registry for them. It is protected by the OWNERS file.

const (
	// NodeCapabilityLifecyclePrefix is the prefix for all node capability keys that are tied to the lifecycle of feature gates.
	NodeCapabilityLifecyclePrefix = "lifecycle.kubernetes.io/"
	// GuaranteedQoSPodCPUResize is the capability key for in-place pod resize for guaranteed QoS pods.
	GuaranteedQoSPodCPUResize = NodeCapabilityLifecyclePrefix + "guaranteed-qos-pod-cpu-resize"
)

// wellKnownCapabilities is the authoritative list of all known node capabilities.
var wellKnownCapabilities = map[string]struct{}{
	GuaranteedQoSPodCPUResize: {},
}

// nodeCapabilitiesRegistry is the authoritative registry of all node capabilities.
var nodeCapabilitiesRegistry = make(map[string]nodecapabilities.Capability)

// Register registers a new capability in the registry.
// This function should be called from the init() function of the package that defines the capability.
// It will panic if the capability is not in the well-known list or if the name is invalid.
func Register(capability nodecapabilities.Capability) {
	if _, ok := wellKnownCapabilities[capability.Name]; !ok {
		panic(fmt.Sprintf("capability %q is not in the well-known list", capability.Name))
	}

	// TODO: Pass a bool if no validation is needed for value. We cannot apply rules for values here.
	if err := ValidateCapability(capability.Name, "true"); err != nil {
		panic(err)
	}

	nodeCapabilitiesRegistry[capability.Name] = capability
}

// NewRegistry returns a new instance of the node capabilities registry.
func NewRegistry() nodecapabilities.Registry {
	return &registry{}
}

type registry struct{}

// Get returns the capability definition for the given name.
func (r *registry) Get(name string) (nodecapabilities.Capability, bool) {
	cap, ok := nodeCapabilitiesRegistry[name]
	return cap, ok
}

// ForEach iterates over all registered capabilities.
func (r *registry) ForEach(f func(name string, cap nodecapabilities.Capability)) {
	for name, cap := range nodeCapabilitiesRegistry {
		f(name, cap)
	}
}
