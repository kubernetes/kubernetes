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

	"k8s.io/api/core/v1"
)

// TODO: Move these values into the API package.
const (
	// The prefix to an annotation key specifying a container profile.
	ContainerAnnotationKeyPrefix = "container.apparmor.security.beta.kubernetes.io/"
	// The annotation key specifying the default AppArmor profile.
	DefaultProfileAnnotationKey = "apparmor.security.beta.kubernetes.io/defaultProfileName"
	// The annotation key specifying the allowed AppArmor profiles.
	AllowedProfilesAnnotationKey = "apparmor.security.beta.kubernetes.io/allowedProfileNames"

	// The profile specifying the runtime default.
	ProfileRuntimeDefault = "runtime/default"
	// The prefix for specifying profiles loaded on the node.
	ProfileNamePrefix = "localhost/"

	// Unconfined profile
	ProfileNameUnconfined = "unconfined"
)

// Checks whether app armor is required for pod to be run.
func isRequired(pod *v1.Pod) bool {
	for key, value := range pod.Annotations {
		if strings.HasPrefix(key, ContainerAnnotationKeyPrefix) {
			return value != ProfileNameUnconfined
		}
	}
	return false
}

// Returns the name of the profile to use with the container.
func GetProfileName(pod *v1.Pod, containerName string) string {
	return GetProfileNameFromPodAnnotations(pod.Annotations, containerName)
}

// GetProfileNameFromPodAnnotations gets the name of the profile to use with container from
// pod annotations
func GetProfileNameFromPodAnnotations(annotations map[string]string, containerName string) string {
	return annotations[ContainerAnnotationKeyPrefix+containerName]
}

// Sets the name of the profile to use with the container.
func SetProfileName(pod *v1.Pod, containerName, profileName string) error {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[ContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}

// Sets the name of the profile to use with the container.
func SetProfileNameFromPodAnnotations(annotations map[string]string, containerName, profileName string) error {
	if annotations == nil {
		return nil
	}
	annotations[ContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}
