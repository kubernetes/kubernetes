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
Note: these fixtures utilize seccomp helper functions that ensure consistency across the
alpha annotation (up to v.1.19) and the securityContext.seccompProfile field (v1.19+).

The check implementation looks at the appropriate value based on version.
*/

func init() {
	fixtureData_restricted_1_19 := fixtureGenerator{
		expectErrorSubstring: "seccompProfile",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = seccompProfileLocalhost("testing")
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = nil
					p.Spec.Containers[0].SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
					p.Spec.InitContainers[0].SecurityContext.SeccompProfile = seccompProfileLocalhost("testing")
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// unset everywhere
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = nil
				}),
				// unconfined, pod-level
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = seccompProfileUnconfined
				}),
				// unset initContainer
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = nil
					p.Spec.Containers[0].SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
				}),
				// unset container
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = nil
					p.Spec.InitContainers[0].SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
				}),
				// unconfined, container-level
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = nil
					p.Spec.Containers[0].SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
					p.Spec.InitContainers[0].SecurityContext.SeccompProfile = seccompProfileUnconfined
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 19), check: "seccompProfile_restricted"},
		fixtureData_restricted_1_19,
	)
}
