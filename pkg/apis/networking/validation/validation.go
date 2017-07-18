/*
Copyright 2017 The Kubernetes Authors.

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
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/networking"
)

// ValidateNetworkPolicyName can be used to check whether the given networkpolicy
// name is valid.
func ValidateNetworkPolicyName(name string, prefix bool) []string {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidateNetworkPolicySpec tests if required fields in the networkpolicy spec are set.
func ValidateNetworkPolicySpec(spec *networking.NetworkPolicySpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(&spec.PodSelector, fldPath.Child("podSelector"))...)

	// Validate ingress rules.
	for i, ingress := range spec.Ingress {
		ingressPath := fldPath.Child("ingress").Index(i)
		for i, port := range ingress.Ports {
			portPath := ingressPath.Child("ports").Index(i)
			if port.Protocol != nil && *port.Protocol != api.ProtocolTCP && *port.Protocol != api.ProtocolUDP {
				allErrs = append(allErrs, field.NotSupported(portPath.Child("protocol"), *port.Protocol, []string{string(api.ProtocolTCP), string(api.ProtocolUDP)}))
			}
			if port.Port != nil {
				if port.Port.Type == intstr.Int {
					for _, msg := range validation.IsValidPortNum(int(port.Port.IntVal)) {
						allErrs = append(allErrs, field.Invalid(portPath.Child("port"), port.Port.IntVal, msg))
					}
				} else {
					for _, msg := range validation.IsValidPortName(port.Port.StrVal) {
						allErrs = append(allErrs, field.Invalid(portPath.Child("port"), port.Port.StrVal, msg))
					}
				}
			}
		}
		for i, from := range ingress.From {
			fromPath := ingressPath.Child("from").Index(i)
			numFroms := 0
			if from.PodSelector != nil {
				numFroms++
				allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(from.PodSelector, fromPath.Child("podSelector"))...)
			}
			if from.NamespaceSelector != nil {
				numFroms++
				allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(from.NamespaceSelector, fromPath.Child("namespaceSelector"))...)
			}

			if numFroms == 0 {
				allErrs = append(allErrs, field.Required(fromPath, "must specify a from type"))
			} else if numFroms > 1 {
				allErrs = append(allErrs, field.Forbidden(fromPath, "may not specify more than 1 from type"))
			}
		}
	}
	return allErrs
}

// ValidateNetworkPolicy validates a networkpolicy.
func ValidateNetworkPolicy(np *networking.NetworkPolicy) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&np.ObjectMeta, true, ValidateNetworkPolicyName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateNetworkPolicySpec(&np.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateNetworkPolicyUpdate tests if an update to a NetworkPolicy is valid.
func ValidateNetworkPolicyUpdate(update, old *networking.NetworkPolicy) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, ValidateNetworkPolicySpec(&update.Spec, field.NewPath("spec"))...)
	return allErrs
}
