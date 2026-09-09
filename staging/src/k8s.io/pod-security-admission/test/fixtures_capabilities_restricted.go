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
	`securityContext.capabilities.drop`,
},
*/

func init() {
	fixtureData_1_22 := fixtureGenerator{
		expectErrorSubstring: "unrestricted capabilities",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			p = ensureCapabilities(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"NET_BIND_SERVICE"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"NET_BIND_SERVICE"}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureCapabilities(p)
			return []*corev1.Pod{
				// test container
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{}
				}),
				// test initContainer
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{}
				}),
				// dropping all individual capabilities is not sufficient
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{
						"SYS_TIME", "SYS_MODULE", "SYS_RAWIO", "SYS_PACCT", "SYS_ADMIN", "SYS_NICE",
						"SYS_RESOURCE", "SYS_TIME", "SYS_TTY_CONFIG", "MKNOD", "AUDIT_WRITE",
						"AUDIT_CONTROL", "MAC_OVERRIDE", "MAC_ADMIN", "NET_ADMIN", "SYSLOG",
						"CHOWN", "NET_RAW", "DAC_OVERRIDE", "FOWNER", "DAC_READ_SEARCH",
						"FSETID", "KILL", "SETGID", "SETUID", "LINUX_IMMUTABLE", "NET_BIND_SERVICE",
						"NET_BROADCAST", "IPC_LOCK", "IPC_OWNER", "SYS_CHROOT", "SYS_PTRACE",
						"SYS_BOOT", "LEASE", "SETFCAP", "WAKE_ALARM", "BLOCK_SUSPEND",
					}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{
						"SYS_TIME", "SYS_MODULE", "SYS_RAWIO", "SYS_PACCT", "SYS_ADMIN", "SYS_NICE",
						"SYS_RESOURCE", "SYS_TIME", "SYS_TTY_CONFIG", "MKNOD", "AUDIT_WRITE",
						"AUDIT_CONTROL", "MAC_OVERRIDE", "MAC_ADMIN", "NET_ADMIN", "SYSLOG",
						"CHOWN", "NET_RAW", "DAC_OVERRIDE", "FOWNER", "DAC_READ_SEARCH",
						"FSETID", "KILL", "SETGID", "SETUID", "LINUX_IMMUTABLE", "NET_BIND_SERVICE",
						"NET_BROADCAST", "IPC_LOCK", "IPC_OWNER", "SYS_CHROOT", "SYS_PTRACE",
						"SYS_BOOT", "LEASE", "SETFCAP", "WAKE_ALARM", "BLOCK_SUSPEND",
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						// try adding back capabilities other than NET_BIND_SERVICE, should be forbidden
						"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT",
					}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						// try adding back capabilities other than NET_BIND_SERVICE, should be forbidden
						"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT",
					}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 22), check: "capabilities_restricted"},
		fixtureData_1_22,
	)
}
