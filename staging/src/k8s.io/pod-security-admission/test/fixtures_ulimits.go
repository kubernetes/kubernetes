/*
Copyright The Kubernetes Authors.

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
	"k8s.io/component-base/featuregate"
	"k8s.io/pod-security-admission/api"
)

func init() {
	fixtureData1_0 := fixtureGenerator{
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			// minimal valid pod already captures all valid combinations
			return nil
		},
		failRequiresFeatures: []featuregate.Feature{"ContainerUlimits"},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// ulimits set in a container
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Containers[0].SecurityContext.Ulimits = []corev1.Ulimit{
						{Name: "nofile", Soft: 1024, Hard: 2048},
					}
				}),
				// ulimits set in an init container
				tweak(p, func(p *corev1.Pod) {
					p.Spec.InitContainers[0].SecurityContext.Ulimits = []corev1.Ulimit{
						{Name: "core", Soft: 1, Hard: 1},
					}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "ulimits"},
		fixtureData1_0,
	)
}
