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
	`spec.securityContext.seLinuxOptions.type`,
	`spec.securityContext.seLinuxOptions.user`,
	`spec.securityContext.seLinuxOptions.role`,
},
containerFields: []string{
	`securityContext.seLinuxOptions.type`,
	`securityContext.seLinuxOptions.user`,
	`securityContext.seLinuxOptions.role`,
},
*/

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "seLinuxOptions",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSELinuxOptions(p)
			return []*corev1.Pod{
				// security context with no seLinuxOptions
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions = nil }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions = nil }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions = nil }),
				// seLinuxOptions with type=""
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Type = "" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Type = "" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Type = "" }),
				// seLinuxOptions with type="container_t"
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Type = "container_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Type = "container_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Type = "container_t" }),
				// seLinuxOptions with type="container_init_t"
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Type = "container_init_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Type = "container_init_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Type = "container_init_t" }),
				// seLinuxOptions with type="container_kvm_t"
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Type = "container_kvm_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Type = "container_kvm_t" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Type = "container_kvm_t" }),
				// seLinuxOptions with level=""
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Level = "" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Level = "" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Level = "" }),
				// seLinuxOptions with arbitrary level=""
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Level = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Level = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Level = "somevalue" }),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSELinuxOptions(p)
			return []*corev1.Pod{
				// seLinuxOptions with out of bounds type
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Type = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Type = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Type = "somevalue" }),
				// seLinuxOptions with out of bounds user
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.User = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.User = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.User = "somevalue" }),
				// seLinuxOptions with out of bounds role
				tweak(p, func(p *corev1.Pod) { p.Spec.SecurityContext.SELinuxOptions.Role = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.Containers[0].SecurityContext.SELinuxOptions.Role = "somevalue" }),
				tweak(p, func(p *corev1.Pod) { p.Spec.InitContainers[0].SecurityContext.SELinuxOptions.Role = "somevalue" }),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "selinux"},
		fixtureData_1_0,
	)
}
