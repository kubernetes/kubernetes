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

package apparmor

import (
	"strings"

	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// Checks whether app armor is required for the pod to run. AppArmor is considered required if any
// non-unconfined profiles are specified.
func isRequired(pod *v1.Pod) bool {
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.AppArmorProfile != nil &&
		pod.Spec.SecurityContext.AppArmorProfile.Type != v1.AppArmorProfileTypeUnconfined {
		return true
	}

	inUse := !podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(c *v1.Container, _ podutil.ContainerType) bool {
		if c.SecurityContext != nil && c.SecurityContext.AppArmorProfile != nil &&
			c.SecurityContext.AppArmorProfile.Type != v1.AppArmorProfileTypeUnconfined {
			return false // is in use; short-circuit
		}
		return true
	})
	if inUse {
		return true
	}

	for key, value := range pod.Annotations {
		if strings.HasPrefix(key, v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix) {
			return value != v1.DeprecatedAppArmorBetaProfileNameUnconfined
		}
	}
	return false
}

// GetProfileName returns the name of the profile to use with the container.
func GetProfile(pod *v1.Pod, container *v1.Container) *v1.AppArmorProfile {
	if container.SecurityContext != nil && container.SecurityContext.AppArmorProfile != nil {
		return container.SecurityContext.AppArmorProfile
	}

	// Static pods may not have had annotations synced to fields, so fallback to annotations before
	// the pod profile.
	if profile := getProfileFromPodAnnotations(pod.Annotations, container.Name); profile != nil {
		return profile
	}

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.AppArmorProfile != nil {
		return pod.Spec.SecurityContext.AppArmorProfile
	}

	return nil
}

// getProfileFromPodAnnotations gets the AppArmor profile to use with container from
// (deprecated) pod annotations.
func getProfileFromPodAnnotations(annotations map[string]string, containerName string) *v1.AppArmorProfile {
	val, ok := annotations[v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix+containerName]
	if !ok {
		return nil
	}

	switch {
	case val == v1.DeprecatedAppArmorBetaProfileRuntimeDefault:
		return &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeRuntimeDefault}

	case val == v1.DeprecatedAppArmorBetaProfileNameUnconfined:
		return &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeUnconfined}

	case strings.HasPrefix(val, v1.DeprecatedAppArmorBetaProfileNamePrefix):
		// Note: an invalid empty localhost profile will be rejected by kubelet admission.
		profileName := strings.TrimPrefix(val, v1.DeprecatedAppArmorBetaProfileNamePrefix)
		return &v1.AppArmorProfile{
			Type:             v1.AppArmorProfileTypeLocalhost,
			LocalhostProfile: &profileName,
		}

	default:
		// Invalid annotation.
		return nil
	}
}
