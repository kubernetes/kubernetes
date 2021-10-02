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
	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/pod-security-admission/api"
)

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "procMount",
		generatePass: func(p *v1.Pod) []*v1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container and init container to a valid value
				tweak(p, func(copy *v1.Pod) {
					validProcMountType := v1.DefaultProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &validProcMountType
				}),
			}
		},
		failRequiresFeatures: []featuregate.Feature{"ProcMountType"},
		generateFail: func(p *v1.Pod) []*v1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container to a forbidden value
				tweak(p, func(copy *v1.Pod) {
					unmaskedProcMountType := v1.UnmaskedProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &unmaskedProcMountType
				}),
				// set proc mount of init container to a forbidden value
				tweak(p, func(copy *v1.Pod) {
					unmaskedProcMountType := v1.UnmaskedProcMount
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &unmaskedProcMountType
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "procMount"},
		fixtureData_1_0,
	)
}
