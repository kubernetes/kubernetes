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

// ensureCapabilities ensures the pod and all initContainers and containers have a non-nil capabilities.
func ensureCapabilities(p *corev1.Pod) *corev1.Pod {
	p = ensureSecurityContext(p)
	for i := range p.Spec.Containers {
		if p.Spec.Containers[i].SecurityContext.Capabilities == nil {
			p.Spec.Containers[i].SecurityContext.Capabilities = &corev1.Capabilities{}
		}
	}
	for i := range p.Spec.InitContainers {
		if p.Spec.InitContainers[i].SecurityContext.Capabilities == nil {
			p.Spec.InitContainers[i].SecurityContext.Capabilities = &corev1.Capabilities{}
		}
	}
	return p
}

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "non-default capabilities",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			// don't generate fixtures if minimal valid pod drops ALL
			if p.Spec.Containers[0].SecurityContext != nil && p.Spec.Containers[0].SecurityContext.Capabilities != nil {
				for _, capability := range p.Spec.Containers[0].SecurityContext.Capabilities.Drop {
					if capability == corev1.Capability("ALL") {
						return nil
					}
				}
			}

			p = ensureCapabilities(p)
			return []*corev1.Pod{
				// all allowed capabilities
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT",
					}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT",
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureCapabilities(p)
			return []*corev1.Pod{
				// NET_RAW
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"NET_RAW"}
				}),
				// ensure init container is enforced
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"NET_RAW"}
				}),
				// case-difference
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"chown"}
				}),
				// CAP_ prefix
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"CAP_CHOWN"}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "capabilities_baseline"},
		fixtureData_1_0,
	)
}
