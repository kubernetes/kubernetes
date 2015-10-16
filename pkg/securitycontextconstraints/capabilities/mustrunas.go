/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// defaultCapabilities implements the CapabilitiesSecurityContextConstraintsStrategy interface
type defaultCapabilities struct {
	defaultAddCapabilities   []api.Capability
	requiredDropCapabilities []api.Capability
	allowedCaps              []api.Capability
}

var _ CapabilitiesSecurityContextConstraintsStrategy = &defaultCapabilities{}

// NewDefaultCapabilities creates a new defaultCapabilities strategy that will provide defaults and validation
// based on the configured initial caps and allowed caps.
func NewDefaultCapabilities(defaultAddCapabilities, requiredDropCapabilities, allowedCaps []api.Capability) (CapabilitiesSecurityContextConstraintsStrategy, error) {
	return &defaultCapabilities{
		defaultAddCapabilities:   defaultAddCapabilities,
		requiredDropCapabilities: requiredDropCapabilities,
		allowedCaps:              allowedCaps,
	}, nil
}

// Generate creates the capabilities based on policy rules.  Generate will produce the following:
// 1.  a capabilities.Add set containing all the required adds (unless the
// 		container specifically is dropping the cap) and container requested adds
// 2.  a capabilities.Drop set containing all the required drops and container requested drops
func (s *defaultCapabilities) Generate(pod *api.Pod, container *api.Container) (*api.Capabilities, error) {
	defaultAdd := makeCapSet(s.defaultAddCapabilities)
	requiredDrop := makeCapSet(s.requiredDropCapabilities)
	containerAdd := sets.NewString()
	containerDrop := sets.NewString()

	if container.SecurityContext != nil && container.SecurityContext.Capabilities != nil {
		containerAdd = makeCapSet(container.SecurityContext.Capabilities.Add)
		containerDrop = makeCapSet(container.SecurityContext.Capabilities.Drop)
	}

	// remove any default adds that the container is specifically dropping
	defaultAdd = defaultAdd.Difference(containerDrop)

	combinedAdd := defaultAdd.Union(containerAdd).List()
	combinedDrop := requiredDrop.Union(containerDrop).List()

	// nothing generated?  return nil
	if len(combinedAdd) == 0 && len(combinedDrop) == 0 {
		return nil, nil
	}

	return &api.Capabilities{
		Add:  capabilityFromStringSlice(combinedAdd),
		Drop: capabilityFromStringSlice(combinedDrop),
	}, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *defaultCapabilities) Validate(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}

	// if the security context isn't set then we haven't generated correctly.  Shouldn't get here
	// if using the provider correctly
	if container.SecurityContext == nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("securityContext"), container.SecurityContext, "no security context is set"))
		return allErrs
	}

	if container.SecurityContext.Capabilities == nil {
		// if container.SC.Caps is nil then nothing was defaulted by the strat or requested by the pod author
		// if there are no required caps on the strategy and nothing is requested on the pod
		// then we can safely return here without further validation.
		if len(s.defaultAddCapabilities) == 0 && len(s.requiredDropCapabilities) == 0 {
			return allErrs
		}

		// container has no requested caps but we have required caps.  We should have something in
		// at least the drops on the container.
		allErrs = append(allErrs, field.Invalid(field.NewPath("capabilities"), container.SecurityContext.Capabilities,
			"required capabilities are not set on the securityContext"))
		return allErrs
	}

	// validate that anything being added is in the default or allowed sets
	defaultAdd := makeCapSet(s.defaultAddCapabilities)
	allowedAdd := makeCapSet(s.allowedCaps)

	for _, cap := range container.SecurityContext.Capabilities.Add {
		sCap := string(cap)
		if !defaultAdd.Has(sCap) && !allowedAdd.Has(sCap) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("capabilities", "add"), sCap, "capability may not be added"))
		}
	}

	// validate that anything that is required to be dropped is in the drop set
	containerDrops := makeCapSet(container.SecurityContext.Capabilities.Drop)

	for _, requiredDrop := range s.requiredDropCapabilities {
		sDrop := string(requiredDrop)
		if !containerDrops.Has(sDrop) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("capabilities", "drop"), container.SecurityContext.Capabilities.Drop,
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

// makeCapSet makes a string set from capabilities and normalizes them to be all lower case to help
// with comparisons.
func makeCapSet(caps []api.Capability) sets.String {
	s := sets.NewString()
	for _, c := range caps {
		s.Insert(string(c))
	}
	return s
}
