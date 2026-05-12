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
	fixtureData_baseline_1_0 := fixtureGenerator{
		expectErrorSubstring: "seccompProfile",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			// don't generate fixtures if minimal valid pod already has seccomp config
			if val, ok := p.Annotations[annotationKeyPod]; ok &&
				val == corev1.SeccompProfileRuntimeDefault {
				return nil
			}

			p = ensureAnnotation(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					// pod-level default
					p.Annotations[annotationKeyPod] = corev1.SeccompProfileRuntimeDefault
					// container-level localhost
					p.Annotations[annotationKeyContainer(p.Spec.Containers[0])] = corev1.SeccompLocalhostProfileNamePrefix + "testing"
					// init-container unset
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureAnnotation(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Annotations[annotationKeyPod] = corev1.SeccompProfileNameUnconfined
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Annotations[annotationKeyContainer(p.Spec.Containers[0])] = corev1.SeccompProfileNameUnconfined
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Annotations[annotationKeyContainer(p.Spec.InitContainers[0])] = corev1.SeccompProfileNameUnconfined
				}),
			}
		},
	}

	fixtureData_baseline_1_19 := fixtureGenerator{
		expectErrorSubstring: "seccompProfile",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			// don't generate fixtures if minimal valid pod already has seccomp config
			if p.Spec.SecurityContext != nil &&
				p.Spec.SecurityContext.SeccompProfile != nil &&
				p.Spec.SecurityContext.SeccompProfile.Type == corev1.SeccompProfileTypeRuntimeDefault {
				return nil
			}

			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					// pod-level default
					p.Spec.SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
					// container-level localhost
					p.Spec.Containers[0].SecurityContext.SeccompProfile = seccompProfileRuntimeDefault
					// init-container unset
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.SeccompProfile = seccompProfileUnconfined
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.SeccompProfile = seccompProfileUnconfined
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].SecurityContext.SeccompProfile = seccompProfileUnconfined
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "seccompProfile_baseline"},
		fixtureData_baseline_1_0,
	)

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 19), check: "seccompProfile_baseline"},
		fixtureData_baseline_1_19,
	)
}
