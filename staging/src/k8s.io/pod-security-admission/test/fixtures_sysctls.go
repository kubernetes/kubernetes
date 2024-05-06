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

podFields: []string{
	`spec.securityContext.sysctls.name`,
},
*/

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "forbidden sysctl",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// security context with no sysctls
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.Sysctls = nil }),
				// sysctls with name="kernel.shm_rmid_forced" ,"net.ipv4.ip_local_port_range"
				// "net.ipv4.tcp_syncookies", "net.ipv4.ping_group_range",
				// "net.ipv4.ip_unprivileged_port_start"
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{
						{Name: "kernel.shm_rmid_forced", Value: "0"},
						{Name: "net.ipv4.ip_local_port_range", Value: "1024 65535"},
						{Name: "net.ipv4.tcp_syncookies", Value: "0"},
						{Name: "net.ipv4.ping_group_range", Value: "1 0"},
						{Name: "net.ipv4.ip_unprivileged_port_start", Value: "1024"},
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// sysctls with out of allowed name
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{{Name: "othersysctl", Value: "other"}}
				}),
			}
		},
	}
	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "sysctls"},
		fixtureData_1_0,
	)

	fixtureData_1_27 := fixtureGenerator{
		expectErrorSubstring: "forbidden sysctl",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// security context with no sysctls
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.Sysctls = nil }),
				// sysctls with name="kernel.shm_rmid_forced" ,"net.ipv4.ip_local_port_range"
				// "net.ipv4.tcp_syncookies", "net.ipv4.ping_group_range",
				// "net.ipv4.ip_unprivileged_port_start", "net.ipv4.ip_local_reserved_ports"
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{
						{Name: "kernel.shm_rmid_forced", Value: "0"},
						{Name: "net.ipv4.ip_local_port_range", Value: "1024 65535"},
						{Name: "net.ipv4.tcp_syncookies", Value: "0"},
						{Name: "net.ipv4.ping_group_range", Value: "1 0"},
						{Name: "net.ipv4.ip_unprivileged_port_start", Value: "1024"},
						{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"},
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// sysctls with out of allowed name
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{{Name: "othersysctl", Value: "other"}}
				}),
			}
		},
	}
	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 27), check: "sysctls"},
		fixtureData_1_27,
	)

	fixtureDataV1Dot29 := fixtureGenerator{
		expectErrorSubstring: "forbidden sysctl",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// security context with no sysctls
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.Sysctls = nil }),
				// sysctls with name="kernel.shm_rmid_forced" ,"net.ipv4.ip_local_port_range"
				// "net.ipv4.tcp_syncookies", "net.ipv4.ping_group_range",
				// "net.ipv4.ip_unprivileged_port_start", "net.ipv4.ip_local_reserved_ports",
				// "net.ipv4.tcp_keepalive_time"
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{
						{Name: "kernel.shm_rmid_forced", Value: "0"},
						{Name: "net.ipv4.ip_local_port_range", Value: "1024 65535"},
						{Name: "net.ipv4.tcp_syncookies", Value: "0"},
						{Name: "net.ipv4.ping_group_range", Value: "1 0"},
						{Name: "net.ipv4.ip_unprivileged_port_start", Value: "1024"},
						{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"},
						{Name: "net.ipv4.tcp_keepalive_time", Value: "7200"},
						{Name: "net.ipv4.tcp_fin_timeout", Value: "60"},
						{Name: "net.ipv4.tcp_keepalive_intvl", Value: "75"},
						{Name: "net.ipv4.tcp_keepalive_probes", Value: "9"},
					}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			if p.Spec.SecurityContext == nil {
				p.Spec.SecurityContext = &corev1.PodSecurityContext{}
			}
			return []*corev1.Pod{
				// sysctls with out of allowed name
				tweak(p, func(p *corev1.Pod) {
					p.Spec.SecurityContext.Sysctls = []corev1.Sysctl{{Name: "othersysctl", Value: "other"}}
				}),
			}
		},
	}
	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 29), check: "sysctls"},
		fixtureDataV1Dot29,
	)
}
