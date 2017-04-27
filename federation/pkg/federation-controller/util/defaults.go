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

package util

import (
	"k8s.io/kubernetes/pkg/api/v1"
)

// setPodSpecDefaults sets optional PodSpec fields to their defaults
// to ensure that specs for older cluster versions can be compared
// with DeepEqual.  Where a resource spec in the federation control
// plane may have a default supplied for an optional field, an older
// cluster would return empty values.
func setPodSpecDefaults(podSpec *v1.PodSpec) {
	v1.SetDefaults_PodSpec(podSpec)
	for i, container := range podSpec.Containers {
		v1.SetDefaults_Container(&container)
		podSpec.Containers[i] = container
	}
}
