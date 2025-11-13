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
	"k8s.io/component-base/featuregate"
	"k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

func init() {
	fixtureData_1_0 := fixtureGenerator{
		expectErrorSubstring: "procMount",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container and init container to a valid value
				tweak(p, func(copy *corev1.Pod) {
					validProcMountType := corev1.DefaultProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
			}
		},
		failRequiresFeatures: []featuregate.Feature{"ProcMountType"},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
				// set proc mount of init container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "procMount"},
		fixtureData_1_0,
	)

	fixtureData1_35baseline := fixtureGenerator{
		expectErrorSubstring: "procMount",
		generatePass: func(p *corev1.Pod, level api.Level) []*corev1.Pod {
			p = ensureSecurityContext(p)
			retval := []*corev1.Pod{
				// set proc mount of container and init container to a valid value
				tweak(p, func(copy *corev1.Pod) {
					validProcMountType := corev1.DefaultProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
			}

			if level != api.LevelRestricted {
				// don't expect the unmasked namespace user pod to pass in restricted mode
				retval = append(retval, tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}))
			}

			return retval
		},
		failRequiresFeatures: []featuregate.Feature{"ProcMountType"},
		failRequiresError:    true, // the only combination that can fail the 1.35 baseline check also fails validation
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(true)
				}),
				// set proc mount of init container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(true)
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 35), check: "procMount"},
		fixtureData1_35baseline,
	)

	fixtureData1_35restricted := fixtureGenerator{
		expectErrorSubstring: "procMount",
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container and init container to a valid value
				tweak(p, func(copy *corev1.Pod) {
					validProcMountType := corev1.DefaultProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &validProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
			}
		},
		failRequiresFeatures: []featuregate.Feature{"ProcMountType"},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				// set proc mount of container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.Containers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
				// set proc mount of init container to a forbidden value
				tweak(p, func(copy *corev1.Pod) {
					unmaskedProcMountType := corev1.UnmaskedProcMount
					copy.Spec.InitContainers[0].SecurityContext.ProcMount = &unmaskedProcMountType
					copy.Spec.HostUsers = ptr.To(false)
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 35), check: "procMount_restricted"},
		fixtureData1_35restricted,
	)
}
