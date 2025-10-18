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

podFields: []string{
	`securityContext.windowsOptions.hostProcess`,
},
containerFields: []string{
	`securityContext.windowsOptions.hostProcess`,
},

*/

func init() {

	fixtureData_1_0 := fixtureGenerator{
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			// minimal valid pod already captures all valid combinations
			return nil
		},
		expectErrorSubstring: "hostProcess",
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			if p.Spec.SecurityContext.WindowsOptions == nil {
				p.Spec.SecurityContext.WindowsOptions = &corev1.WindowsSecurityContextOptions{}
			}
			if p.Spec.Containers[0].SecurityContext.WindowsOptions == nil {
				p.Spec.Containers[0].SecurityContext.WindowsOptions = &corev1.WindowsSecurityContextOptions{}
			}
			if p.Spec.InitContainers[0].SecurityContext.WindowsOptions == nil {
				p.Spec.InitContainers[0].SecurityContext.WindowsOptions = &corev1.WindowsSecurityContextOptions{}
			}
			return []*corev1.Pod{
				// true for pod
				tweak(p, func(p *corev1.Pod) {
					// HostNetwork is required to be true for HostProcess pods.
					// Set to true here so we pass API validation and get to admission checks.
					p.Spec.HostNetwork = true
					p.Spec.SecurityContext.WindowsOptions.HostProcess = ptr.To(true)
				}),
				// true for containers
				tweak(p, func(p *corev1.Pod) {
					// HostNetwork is required to be true for HostProcess pods.
					// Set to true here so we pass API validation and get to admission checks.
					p.Spec.HostNetwork = true
					p.Spec.Containers[0].SecurityContext.WindowsOptions.HostProcess = ptr.To(true)
					p.Spec.InitContainers[0].SecurityContext.WindowsOptions.HostProcess = ptr.To(true)
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "windowsHostProcess"},
		fixtureData_1_0,
	)
	// TODO: register another set of fixtures with passing test cases that explicitly set hostProcess=false at pod and container level once hostProcess is GA
}
