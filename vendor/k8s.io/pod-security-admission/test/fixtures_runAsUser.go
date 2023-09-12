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
	"k8s.io/utils/pointer"
)

/*
TODO: include field paths in reflect-based unit test

podFields: []string{
	`securityContext.runAsUser`,
},
containerFields: []string{
	`securityContext.runAsUser`,
},

*/

func init() {

	fixtureData_1_23 := fixtureGenerator{
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.RunAsUser = pointer.Int64Ptr(1000)
					p.Spec.Containers[0].SecurityContext.RunAsUser = pointer.Int64Ptr(1000)
					p.Spec.InitContainers[0].SecurityContext.RunAsUser = pointer.Int64Ptr(1000)
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// explicit 0 on pod
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.RunAsUser = pointer.Int64Ptr(0) }),
				// explicit 0 on containers
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.RunAsUser = pointer.Int64Ptr(0) }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.RunAsUser = pointer.Int64Ptr(0) }),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 23), check: "runAsUser"},
		fixtureData_1_23,
	)
}
