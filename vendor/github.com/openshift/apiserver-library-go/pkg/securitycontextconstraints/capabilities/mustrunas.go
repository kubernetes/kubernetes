package capabilities

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
)

// defaultCapabilities implements the CapabilitiesSecurityContextConstraintsStrategy interface
type defaultCapabilities struct {
	defaultAddCapabilities   []corev1.Capability
	requiredDropCapabilities []corev1.Capability
	allowedCaps              []corev1.Capability
}

var _ CapabilitiesSecurityContextConstraintsStrategy = &defaultCapabilities{}

// NewDefaultCapabilities creates a new defaultCapabilities strategy that will provide defaults and validation
// based on the configured initial caps and allowed caps.
func NewDefaultCapabilities(defaultAddCapabilities, requiredDropCapabilities, allowedCaps []corev1.Capability) (CapabilitiesSecurityContextConstraintsStrategy, error) {
	return &defaultCapabilities{
		defaultAddCapabilities:   defaultAddCapabilities,
		requiredDropCapabilities: requiredDropCapabilities,
		allowedCaps:              allowedCaps,
	}, nil
}

// Generate creates the capabilities based on policy rules.  Generate will produce the following:
//  1. a capabilities.Add set containing all the required adds (unless the
//     container specifically is dropping the cap) and container requested adds
//  2. a capabilities.Drop set containing all the required drops and container requested drops
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
		containerAdd = makeCapSetInternal(container.SecurityContext.Capabilities.Add)
		containerDrop = makeCapSetInternal(container.SecurityContext.Capabilities.Drop)
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
		// if container.SC.Caps is nil then nothing was defaulted by the strat or requested by the pod author
		// if there are no required caps on the strategy and nothing is requested on the pod
		// then we can safely return here without further validation.
		if len(s.defaultAddCapabilities) == 0 && len(s.requiredDropCapabilities) == 0 {
			return allErrs
		}

		// container has no requested caps but we have required caps.  We should have something in
		// at least the drops on the container.
		allErrs = append(allErrs, field.Invalid(fldPath.Child("capabilities"), capabilities,
			"required capabilities are not set on the securityContext"))
		return allErrs
	}

	allowedAdd := makeCapSet(s.allowedCaps)
	allowAllCaps := allowedAdd.Has(string(securityv1.AllowAllCapabilities))
	if allowAllCaps {
		// skip validation against allowed/defaultAdd/requiredDrop because all capabilities are allowed by a wildcard
		return allErrs
	}

	// validate that anything being added is in the default or allowed sets
	defaultAdd := makeCapSet(s.defaultAddCapabilities)

	for _, cap := range capabilities.Add {
		sCap := string(cap)
		if !defaultAdd.Has(sCap) && !allowedAdd.Has(sCap) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("capabilities", "add"), sCap, "capability may not be added"))
		}
	}

	// validate that anything that is required to be dropped is in the drop set
	containerDrops := makeCapSetInternal(capabilities.Drop)

	for _, requiredDrop := range s.requiredDropCapabilities {
		sDrop := string(requiredDrop)
		if !containerDrops.Has(sDrop) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("capabilities", "drop"), capabilities.Drop,
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
func makeCapSetInternal(caps []api.Capability) sets.String {
	s := sets.NewString()
	for _, c := range caps {
		s.Insert(string(c))
	}
	return s
}

// makeCapSet makes a string set from capabilities and normalizes them to be all lower case to help
// with comparisons.
func makeCapSet(caps []corev1.Capability) sets.String {
	s := sets.NewString()
	for _, c := range caps {
		s.Insert(string(c))
	}
	return s
}
