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

	"k8s.io/kubernetes/pkg/api"
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
)

// Checks whether app armor is required for pod to be run.
func isRequired(pod *api.Pod) bool {
	for key := range pod.Annotations {
		if strings.HasPrefix(key, ContainerAnnotationKeyPrefix) {
			return true
		}
	}
	return false
}

// Returns the name of the profile to use with the container.
func GetProfileName(pod *api.Pod, containerName string) string {
	return pod.Annotations[ContainerAnnotationKeyPrefix+containerName]
}

// Sets the name of the profile to use with the container.
func SetProfileName(pod *api.Pod, containerName, profileName string) error {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[ContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}
