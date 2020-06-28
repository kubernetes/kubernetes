/*
Copyright 2019 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"net"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

// ValidateConditionalService validates conditionally valid fields. allowedIPFamilies is an ordered
// list of the valid IP families (IPv4 or IPv6) that are supported. The first family in the slice
// is the cluster default, although the clusterIP here dictates the family defaulting.
func ValidateConditionalService(service, oldService *api.Service, allowedIPFamilies []api.IPFamily) field.ErrorList {
	var errs field.ErrorList
	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(serviceSCTPFields(oldService)) == 0 {
		for _, f := range serviceSCTPFields(service) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}

	errs = append(errs, validateIPFamily(service, oldService, allowedIPFamilies)...)

	return errs
}

// validateIPFamily checks the IPFamily field.
func validateIPFamily(service, oldService *api.Service, allowedIPFamilies []api.IPFamily) field.ErrorList {
	var errs field.ErrorList

	// specifically allow an invalid value to remain in storage as long as the user isn't changing it, regardless of gate
	if oldService != nil && oldService.Spec.IPFamily != nil && service.Spec.IPFamily != nil && *oldService.Spec.IPFamily == *service.Spec.IPFamily {
		return errs
	}

	// If the gate is off, setting or changing IPFamily is not allowed, but clearing it is
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		if service.Spec.IPFamily != nil {
			if oldService != nil {
				errs = append(errs, ValidateImmutableField(service.Spec.IPFamily, oldService.Spec.IPFamily, field.NewPath("spec", "ipFamily"))...)
			} else {
				errs = append(errs, field.Forbidden(field.NewPath("spec", "ipFamily"), "programmer error, must be cleared when the dual-stack feature gate is off"))
			}
		}
		return errs
	}

	// PrepareCreate, PrepareUpdate, and test cases must all set IPFamily when the gate is on
	if service.Spec.IPFamily == nil {
		errs = append(errs, field.Required(field.NewPath("spec", "ipFamily"), "programmer error, must be set or defaulted by other fields"))
		return errs
	}

	// A user is not allowed to change the IPFamily field, except for ExternalName services
	if oldService != nil && oldService.Spec.IPFamily != nil && service.Spec.Type != api.ServiceTypeExternalName {
		errs = append(errs, ValidateImmutableField(service.Spec.IPFamily, oldService.Spec.IPFamily, field.NewPath("spec", "ipFamily"))...)
	}

	// Verify the IPFamily is one of the allowed families
	desiredFamily := *service.Spec.IPFamily
	if hasIPFamily(allowedIPFamilies, desiredFamily) {
		// the IP family is one of the allowed families, verify that it matches cluster IP
		switch ip := net.ParseIP(service.Spec.ClusterIP); {
		case ip == nil:
			// do not need to check anything
		case netutils.IsIPv6(ip) && desiredFamily != api.IPv6Protocol:
			errs = append(errs, field.Invalid(field.NewPath("spec", "ipFamily"), *service.Spec.IPFamily, "does not match IPv6 cluster IP"))
		case !netutils.IsIPv6(ip) && desiredFamily != api.IPv4Protocol:
			errs = append(errs, field.Invalid(field.NewPath("spec", "ipFamily"), *service.Spec.IPFamily, "does not match IPv4 cluster IP"))
		}
	} else {
		errs = append(errs, field.Invalid(field.NewPath("spec", "ipFamily"), desiredFamily, fmt.Sprintf("only the following families are allowed: %s", joinIPFamilies(allowedIPFamilies, ", "))))
	}
	return errs
}

func hasIPFamily(families []api.IPFamily, family api.IPFamily) bool {
	for _, allow := range families {
		if allow == family {
			return true
		}
	}
	return false
}

func joinIPFamilies(families []api.IPFamily, separator string) string {
	var b strings.Builder
	for i, family := range families {
		if i != 0 {
			b.WriteString(separator)
		}
		b.WriteString(string(family))
	}
	return b.String()
}

func serviceSCTPFields(service *api.Service) []*field.Path {
	if service == nil {
		return nil
	}
	fields := []*field.Path{}
	for pIndex, p := range service.Spec.Ports {
		if p.Protocol == api.ProtocolSCTP {
			fields = append(fields, field.NewPath("spec.ports").Index(pIndex).Child("protocol"))
		}
	}
	return fields
}

// ValidateConditionalEndpoints validates conditionally valid fields.
func ValidateConditionalEndpoints(endpoints, oldEndpoints *api.Endpoints) field.ErrorList {
	var errs field.ErrorList
	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(endpointsSCTPFields(oldEndpoints)) == 0 {
		for _, f := range endpointsSCTPFields(endpoints) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}
	return errs
}

func endpointsSCTPFields(endpoints *api.Endpoints) []*field.Path {
	if endpoints == nil {
		return nil
	}
	fields := []*field.Path{}
	for sIndex, s := range endpoints.Subsets {
		for pIndex, p := range s.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, field.NewPath("subsets").Index(sIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	return fields
}

// ValidateConditionalPodTemplate validates conditionally valid fields.
// This should be called from Validate/ValidateUpdate for all resources containing a PodTemplateSpec
func ValidateConditionalPodTemplate(podTemplate, oldPodTemplate *api.PodTemplateSpec, fldPath *field.Path) field.ErrorList {
	var (
		podSpec    *api.PodSpec
		oldPodSpec *api.PodSpec
	)
	if podTemplate != nil {
		podSpec = &podTemplate.Spec
	}
	if oldPodTemplate != nil {
		oldPodSpec = &oldPodTemplate.Spec
	}
	return validateConditionalPodSpec(podSpec, oldPodSpec, fldPath.Child("spec"))
}

// ValidateConditionalPod validates conditionally valid fields.
// This should be called from Validate/ValidateUpdate for all resources containing a Pod
func ValidateConditionalPod(pod, oldPod *api.Pod, fldPath *field.Path) field.ErrorList {
	var (
		podSpec    *api.PodSpec
		oldPodSpec *api.PodSpec
	)
	if pod != nil {
		podSpec = &pod.Spec
	}
	if oldPod != nil {
		oldPodSpec = &oldPod.Spec
	}
	return validateConditionalPodSpec(podSpec, oldPodSpec, fldPath.Child("spec"))
}

func validateConditionalPodSpec(podSpec, oldPodSpec *api.PodSpec, fldPath *field.Path) field.ErrorList {
	// Always make sure we have a non-nil current pod spec
	if podSpec == nil {
		podSpec = &api.PodSpec{}
	}

	errs := field.ErrorList{}

	// If the SCTPSupport feature is disabled, and the old object isn't using the SCTP feature, prevent the new object from using it
	if !utilfeature.DefaultFeatureGate.Enabled(features.SCTPSupport) && len(podSCTPFields(oldPodSpec, nil)) == 0 {
		for _, f := range podSCTPFields(podSpec, fldPath) {
			errs = append(errs, field.NotSupported(f, api.ProtocolSCTP, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
		}
	}

	return errs
}

func podSCTPFields(podSpec *api.PodSpec, fldPath *field.Path) []*field.Path {
	if podSpec == nil {
		return nil
	}
	fields := []*field.Path{}
	for cIndex, c := range podSpec.InitContainers {
		for pIndex, p := range c.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, fldPath.Child("initContainers").Index(cIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	for cIndex, c := range podSpec.Containers {
		for pIndex, p := range c.Ports {
			if p.Protocol == api.ProtocolSCTP {
				fields = append(fields, fldPath.Child("containers").Index(cIndex).Child("ports").Index(pIndex).Child("protocol"))
			}
		}
	}
	return fields
}
