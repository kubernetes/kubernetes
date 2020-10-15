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
