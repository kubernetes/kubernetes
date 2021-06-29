/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

func init() {
	addCheck(CheckHostPorts)
}

// CheckHostPorts returns a baseline level check
// that forbids any host ports in 1.0+
func CheckHostPorts() Check {
	return Check{
		ID:    "hostPorts",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       hostPorts_1_0,
			},
		},
	}
}

func hostPorts_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	forbiddenContainers := sets.NewString()
	forbiddenHostPorts := sets.NewInt32()
	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		for _, c := range container.Ports {
			if c.HostPort != 0 {
				forbiddenContainers.Insert(container.Name)
				forbiddenHostPorts.Insert(c.HostPort)
			}
		}
	})

	if len(forbiddenHostPorts) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "forbidden host ports",
			ForbiddenDetail: fmt.Sprintf(
				"containers %q use these host ports %d",
				forbiddenContainers.List(),
				forbiddenHostPorts.List(),
			),
		}
	}
	return CheckResult{Allowed: true}
}
