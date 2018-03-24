/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/features"
)

// ValidateSecurityContext ensure the security context contains valid settings
func ValidateSecurityContext(sc *core.SecurityContext, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	//this should only be true for testing since SecurityContext is defaulted by the core
	if sc == nil {
		return allErrs
	}

	if sc.Privileged != nil {
		if *sc.Privileged && !capabilities.Get().AllowPrivileged {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("privileged"), "disallowed by cluster policy"))
		}
	}

	if sc.RunAsUser != nil {
		for _, msg := range validation.IsValidUserID(*sc.RunAsUser) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsUser"), *sc.RunAsUser, msg))
		}
	}

	if sc.RunAsGroup != nil {
		for _, msg := range validation.IsValidGroupID(*sc.RunAsGroup) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsGroup"), *sc.RunAsGroup, msg))
		}
	}

	if sc.AllowPrivilegeEscalation != nil && !*sc.AllowPrivilegeEscalation {
		if sc.Privileged != nil && *sc.Privileged {
			allErrs = append(allErrs, field.Invalid(fldPath, sc, "cannot set `allowPrivilegeEscalation` to false and `privileged` to true"))
		}

		if sc.Capabilities != nil {
			for _, cap := range sc.Capabilities.Add {
				if string(cap) == "CAP_SYS_ADMIN" {
					allErrs = append(allErrs, field.Invalid(fldPath, sc, "cannot set `allowPrivilegeEscalation` to false and `capabilities.Add` CAP_SYS_ADMIN"))
				}
			}
		}
	}

	return allErrs
}

// ValidatePodSecurityContext test that the specified PodSecurityContext has valid data.
func ValidatePodSecurityContext(securityContext *core.PodSecurityContext, spec *core.PodSpec, specPath, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if securityContext != nil {
		allErrs = append(allErrs, validateHostNetwork(securityContext.HostNetwork, spec.Containers, specPath.Child("containers"))...)
		if securityContext.FSGroup != nil {
			for _, msg := range validation.IsValidGroupID(*securityContext.FSGroup) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("fsGroup"), *(securityContext.FSGroup), msg))
			}
		}
		if securityContext.RunAsUser != nil {
			for _, msg := range validation.IsValidUserID(*securityContext.RunAsUser) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsUser"), *(securityContext.RunAsUser), msg))
			}
		}
		if securityContext.RunAsGroup != nil {
			for _, msg := range validation.IsValidGroupID(*securityContext.RunAsGroup) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsGroup"), *(securityContext.RunAsGroup), msg))
			}
		}

		for g, gid := range securityContext.SupplementalGroups {
			for _, msg := range validation.IsValidGroupID(gid) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("supplementalGroups").Index(g), gid, msg))
			}
		}
		if securityContext.ShareProcessNamespace != nil {
			if !utilfeature.DefaultFeatureGate.Enabled(features.PodShareProcessNamespace) {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("shareProcessNamespace"), "Process Namespace Sharing is disabled by PodShareProcessNamespace feature-gate"))
			} else if securityContext.HostPID && *securityContext.ShareProcessNamespace {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("shareProcessNamespace"), *securityContext.ShareProcessNamespace, "ShareProcessNamespace and HostPID cannot both be enabled"))
			}
		}
	}

	return allErrs
}

func validateHostNetwork(hostNetwork bool, containers []core.Container, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if hostNetwork {
		for i, container := range containers {
			portsPath := fldPath.Index(i).Child("ports")
			for i, port := range container.Ports {
				idxPath := portsPath.Index(i)
				if port.HostPort != port.ContainerPort {
					allErrors = append(allErrors, field.Invalid(idxPath.Child("containerPort"), port.ContainerPort, "must match `hostPort` when `hostNetwork` is true"))
				}
			}
		}
	}
	return allErrors
}
