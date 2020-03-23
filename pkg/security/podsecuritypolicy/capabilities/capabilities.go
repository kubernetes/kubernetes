/*
Copyright 2016 The Kubernetes Authors.

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

package capabilities

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// defaultCapabilities implements the Strategy interface
type defaultCapabilities struct {
	defaultAddCapabilities   []api.Capability
	requiredDropCapabilities []api.Capability
	allowedCaps              []api.Capability
}

var _ Strategy = &defaultCapabilities{}

// NewDefaultCapabilities creates a new defaultCapabilities strategy that will provide defaults and validation
// based on the configured initial caps and allowed caps.
func NewDefaultCapabilities(defaultAddCapabilities, requiredDropCapabilities, allowedCaps []corev1.Capability) (Strategy, error) {
	internalDefaultAddCaps := make([]api.Capability, len(defaultAddCapabilities))
	for i, capability := range defaultAddCapabilities {
		internalDefaultAddCaps[i] = api.Capability(capability)
	}
	internalRequiredDropCaps := make([]api.Capability, len(requiredDropCapabilities))
	for i, capability := range requiredDropCapabilities {
		internalRequiredDropCaps[i] = api.Capability(capability)
	}
	internalAllowedCaps := make([]api.Capability, len(allowedCaps))
	for i, capability := range allowedCaps {
		internalAllowedCaps[i] = api.Capability(capability)
	}
	return &defaultCapabilities{
		defaultAddCapabilities:   internalDefaultAddCaps,
		requiredDropCapabilities: internalRequiredDropCaps,
		allowedCaps:              internalAllowedCaps,
	}, nil
}

// Generate creates the capabilities based on policy rules.  Generate will produce the following:
// 1.  a capabilities.Add set containing all the required adds (unless the
// 		container specifically is dropping the cap) and container requested adds
// 2.  a capabilities.Drop set containing all the required drops and container requested drops
//
// Returns the original container capabilities if no changes are required.
func (s *defaultCapabilities) Generate(pod *api.Pod, container *api.Container) (*api.Capabilities, error) {
	defaultAdd := makeCapSet(s.defaultAddCapabilities)
	requiredDrop := makeCapSet(s.requiredDropCapabilities)
	containerAdd := sets.NewString()
	containerDrop := sets.NewString()

	var containerCapabilities *api.Capabilities
	if container.SecurityContext != nil && container.SecurityContext.Capabilities != nil {
		containerCapabilities = container.SecurityContext.Capabilities
		containerAdd = makeCapSet(container.SecurityContext.Capabilities.Add)
		containerDrop = makeCapSet(container.SecurityContext.Capabilities.Drop)
	}

	// remove any default adds that the container is specifically dropping
	defaultAdd = defaultAdd.Difference(containerDrop)

	combinedAdd := defaultAdd.Union(containerAdd)
	combinedDrop := requiredDrop.Union(containerDrop)

	// no changes? return the original capabilities
	if (len(combinedAdd) == len(containerAdd)) && (len(combinedDrop) == len(containerDrop)) {
		return containerCapabilities, nil
	}

	return &api.Capabilities{
		Add:  capabilityFromStringSlice(combinedAdd.List()),
		Drop: capabilityFromStringSlice(combinedDrop.List()),
	}, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *defaultCapabilities) Validate(fldPath *field.Path, pod *api.Pod, container *api.Container, capabilities *api.Capabilities) field.ErrorList {
	allErrs := field.ErrorList{}

	if capabilities == nil {
		// if container.SC.Caps is nil then nothing was defaulted by the strategy or requested by the pod author
		// if there are no required caps on the strategy and nothing is requested on the pod
		// then we can safely return here without further validation.
		if len(s.defaultAddCapabilities) == 0 && len(s.requiredDropCapabilities) == 0 {
			return allErrs
		}

		// container has no requested caps but we have required caps.  We should have something in
		// at least the drops on the container.
		allErrs = append(allErrs, field.Invalid(fldPath, capabilities,
			"required capabilities are not set on the securityContext"))
		return allErrs
	}

	allowedAdd := makeCapSet(s.allowedCaps)
	allowAllCaps := allowedAdd.Has(string(policy.AllowAllCapabilities))
	if allowAllCaps {
		// skip validation against allowed/defaultAdd/requiredDrop because all capabilities are allowed by a wildcard
		return allErrs
	}

	// validate that anything being added is in the default or allowed sets
	defaultAdd := makeCapSet(s.defaultAddCapabilities)

	for _, cap := range capabilities.Add {
		sCap := string(cap)
		if !defaultAdd.Has(sCap) && !allowedAdd.Has(sCap) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("add"), sCap, "capability may not be added"))
		}
	}

	// validate that anything that is required to be dropped is in the drop set
	containerDrops := makeCapSet(capabilities.Drop)

	for _, requiredDrop := range s.requiredDropCapabilities {
		sDrop := string(requiredDrop)
		if !containerDrops.Has(sDrop) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("drop"), capabilities.Drop,
				fmt.Sprintf("%s is required to be dropped but was not found", sDrop)))
		}
	}

	return allErrs
}

// capabilityFromStringSlice creates a capability slice from a string slice.
func capabilityFromStringSlice(slice []string) []api.Capability {
	if len(slice) == 0 {
		return nil
	}
	caps := []api.Capability{}
	for _, c := range slice {
		caps = append(caps, api.Capability(c))
	}
	return caps
}

// makeCapSet makes a string set from capabilities.
func makeCapSet(caps []api.Capability) sets.String {
	s := sets.NewString()
	for _, c := range caps {
		s.Insert(string(c))
	}
	return s
}
