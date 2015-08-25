/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
)

// FormatPodName returns a string representating a pod in a human readable
// format. This function currently is the same as GetPodFullName in
// kubelet/containers, but may differ in the future. As opposed to
// GetPodFullName, FormatPodName is mainly used for logging.
func FormatPodName(pod *api.Pod) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format), while allowed in the container name format.
	return fmt.Sprintf("%s_%s", pod.Name, pod.Namespace)
}

// FormatPodNames returns a string representating a list of pods in a human
// readable format.
func FormatPodNames(pods []*api.Pod) string {
	podStrings := make([]string, 0, len(pods))
	for _, pod := range pods {
		podStrings = append(podStrings, FormatPodName(pod))
	}
	return fmt.Sprintf(strings.Join(podStrings, ", "))
}
