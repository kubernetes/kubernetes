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
	"k8s.io/utils/ptr"
)

/*
TODO: include field paths in reflect-based unit test

containerFields: []string{
	`securityContext.allowPrivilegeEscalation`,
},

*/

func init() {
	fixtureData_1_8 := fixtureGenerator{
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			// minimal valid pod already captures all valid combinations
			return nil
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.OS != nil && p.Spec.OS.Name == corev1.Windows {
				return []*corev1.Pod{}
			}
			return []*corev1.Pod{
				// explicit true
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = ptr.To(true)
				}),
				// ensure initContainers are checked
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].SecurityContext.AllowPrivilegeEscalation = ptr.To(true)
				}),
				// nil AllowPrivilegeEscalation
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = nil }),
				// nil security context
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext = nil }),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 8), check: "allowPrivilegeEscalation"},
		fixtureData_1_8,
	)
}
