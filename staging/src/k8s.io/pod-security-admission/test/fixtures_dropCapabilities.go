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
		expectErrorSubstring: "drop all",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"CAP_NET_BIND_SERVICE"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"CAP_NET_BIND_SERVICE"}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureCapabilities(p)
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{
						"CAP_SYS_TIME", "CAP_SYS_MODULE", "CAP_SYS_RAWIO", "CAP_SYS_PACCT", "CAP_SYS_ADMIN", "CAP_SYS_NICE",
						"CAP_SYS_RESOURCE", "CAP_SYS_TIME", "CAP_SYS_TTY_CONFIG", "CAP_MKNOD", "CAP_AUDIT_WRITE",
						"CAP_AUDIT_CONTROL", "CAP_MAC_OVERRIDE", "CAP_MAC_ADMIN", "CAP_NET_ADMIN", "CAP_SYSLOG",
						"CAP_CHOWN", "CAP_NET_RAW", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_DAC_READ_SEARCH",
						"CAP_FSETID", "CAP_KILL", "CAP_SETGID", "CAP_SETUID", "CAP_LINUX_IMMUTABLE", "CAP_NET_BIND_SERVICE",
						"CAP_NET_BROADCAST", "CAP_IPC_LOCK", "CAP_IPC_OWNER", "CAP_SYS_CHROOT", "CAP_SYS_PTRACE",
						"CAP_SYS_BOOT", "CAP_LEASE", "CAP_SETFCAP", "CAP_WAKE_ALARM", "CAP_BLOCK_SUSPEND",
					}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{
						"CAP_SYS_TIME", "CAP_SYS_MODULE", "CAP_SYS_RAWIO", "CAP_SYS_PACCT", "CAP_SYS_ADMIN", "CAP_SYS_NICE",
						"CAP_SYS_RESOURCE", "CAP_SYS_TIME", "CAP_SYS_TTY_CONFIG", "CAP_MKNOD", "CAP_AUDIT_WRITE",
						"CAP_AUDIT_CONTROL", "CAP_MAC_OVERRIDE", "CAP_MAC_ADMIN", "CAP_NET_ADMIN", "CAP_SYSLOG",
						"CAP_CHOWN", "CAP_NET_RAW", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_DAC_READ_SEARCH",
						"CAP_FSETID", "CAP_KILL", "CAP_SETGID", "CAP_SETUID", "CAP_LINUX_IMMUTABLE", "CAP_NET_BIND_SERVICE",
						"CAP_NET_BROADCAST", "CAP_IPC_LOCK", "CAP_IPC_OWNER", "CAP_SYS_CHROOT", "CAP_SYS_PTRACE",
						"CAP_SYS_BOOT", "CAP_LEASE", "CAP_SETFCAP", "CAP_WAKE_ALARM", "CAP_BLOCK_SUSPEND",
					}
				}),
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
					p.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						"CAP_SYS_TIME", "CAP_SYS_MODULE", "CAP_SYS_RAWIO", "CAP_SYS_PACCT", "CAP_SYS_ADMIN", "CAP_SYS_NICE",
						"CAP_SYS_RESOURCE", "CAP_SYS_TIME", "CAP_SYS_TTY_CONFIG", "CAP_MKNOD", "CAP_AUDIT_WRITE",
						"CAP_AUDIT_CONTROL", "CAP_MAC_OVERRIDE", "CAP_MAC_ADMIN", "CAP_NET_ADMIN", "CAP_SYSLOG",
						"CAP_CHOWN", "CAP_NET_RAW", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_DAC_READ_SEARCH",
						"CAP_FSETID", "CAP_KILL", "CAP_SETGID", "CAP_SETUID", "CAP_LINUX_IMMUTABLE", "CAP_NET_BROADCAST",
						"CAP_IPC_LOCK", "CAP_IPC_OWNER", "CAP_SYS_CHROOT", "CAP_SYS_PTRACE", "CAP_SYS_BOOT", "CAP_LEASE",
						"CAP_SETFCAP", "CAP_WAKE_ALARM", "CAP_BLOCK_SUSPEND",
					}
					p.Spec.InitContainers[0].SecurityContext.Capabilities.Add = []corev1.Capability{
						"CAP_SYS_TIME", "CAP_SYS_MODULE", "CAP_SYS_RAWIO", "CAP_SYS_PACCT", "CAP_SYS_ADMIN", "CAP_SYS_NICE",
						"CAP_SYS_RESOURCE", "CAP_SYS_TIME", "CAP_SYS_TTY_CONFIG", "CAP_MKNOD", "CAP_AUDIT_WRITE",
						"CAP_AUDIT_CONTROL", "CAP_MAC_OVERRIDE", "CAP_MAC_ADMIN", "CAP_NET_ADMIN", "CAP_SYSLOG",
						"CAP_CHOWN", "CAP_NET_RAW", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_DAC_READ_SEARCH",
						"CAP_FSETID", "CAP_KILL", "CAP_SETGID", "CAP_SETUID", "CAP_LINUX_IMMUTABLE", "CAP_NET_BROADCAST",
						"CAP_IPC_LOCK", "CAP_IPC_OWNER", "CAP_SYS_CHROOT", "CAP_SYS_PTRACE", "CAP_SYS_BOOT", "CAP_LEASE",
						"CAP_SETFCAP", "CAP_WAKE_ALARM", "CAP_BLOCK_SUSPEND",
					}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 22), check: "dropCapabilities"},
		fixtureData_1_22,
	)
}
