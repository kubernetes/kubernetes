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

package test

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/pod-security-admission/api"
)

/*
TODO: include field paths in reflect-based unit test

containerFields: []string{
	`securityContext.capabilities.add`,
},
*/

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "hostPort",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			return []*corev1.Pod{
				// no host ports
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12345,
						},
					}
					p.Spec.InitContainers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12346,
						},
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			return []*corev1.Pod{
				// Host Port present
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12345,
							HostPort:      12345,
						},
					}
				}),
				// check initContainer
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12346,
							HostPort:      12346,
						},
					}
				}),
				// mix of hostPorts and regular ports
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12345,
							HostPort:      12345,
						},
						{
							ContainerPort: 12347,
						},
					}
					p.Spec.InitContainers[0].Ports = []corev1.ContainerPort{
						{
							ContainerPort: 12346,
							HostPort:      12346,
						},
						{
							ContainerPort: 12348,
						},
					}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "hostPorts"},
		fixtureData_1_0,
	)
}
