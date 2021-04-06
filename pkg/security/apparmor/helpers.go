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
	"bufio"
	"fmt"
	"os"
	"path"
	"strings"

	"k8s.io/api/core/v1"
)

// Checks whether app armor is required for pod to be run.
func isRequired(pod *v1.Pod) bool {
	for key, value := range pod.Annotations {
		if strings.HasPrefix(key, v1.AppArmorBetaContainerAnnotationKeyPrefix) {
			return value != v1.AppArmorBetaProfileNameUnconfined
		}
	}
	return false
}

// GetProfileName returns the name of the profile to use with the container.
func GetProfileName(pod *v1.Pod, containerName string) string {
	return GetProfileNameFromPodAnnotations(pod.Annotations, containerName)
}

// GetProfileNameFromPodAnnotations gets the name of the profile to use with container from
// pod annotations
func GetProfileNameFromPodAnnotations(annotations map[string]string, containerName string) string {
	return annotations[v1.AppArmorBetaContainerAnnotationKeyPrefix+containerName]
}

// SetProfileName sets the name of the profile to use with the container.
func SetProfileName(pod *v1.Pod, containerName, profileName string) error {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[v1.AppArmorBetaContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}

// SetProfileNameFromPodAnnotations sets the name of the profile to use with the container.
func SetProfileNameFromPodAnnotations(annotations map[string]string, containerName, profileName string) error {
	if annotations == nil {
		return nil
	}
	annotations[v1.AppArmorBetaContainerAnnotationKeyPrefix+containerName] = profileName
	return nil
}

// GetLoadedProfiles gets profiles under AppArmor filesystem.
func GetLoadedProfiles(appArmorFS string) (map[string]bool, error) {
	profilesPath := path.Join(appArmorFS, "profiles")
	profilesFile, err := os.Open(profilesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %v", profilesPath, err)
	}
	defer profilesFile.Close()

	profiles := map[string]bool{}
	scanner := bufio.NewScanner(profilesFile)
	for scanner.Scan() {
		profileName := parseProfileName(scanner.Text())
		if profileName == "" {
			// Unknown line format; skip it.
			continue
		}
		profiles[profileName] = true
	}
	return profiles, nil
}
