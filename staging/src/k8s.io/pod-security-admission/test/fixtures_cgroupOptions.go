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
		expectErrorSubstring: "cgroupOptions",
		failRequiresFeatures: []featuregate.Feature{"CgroupOptions"},
		generatePass: func(p *corev1.Pod, _ api.Level) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(copy *corev1.Pod) {
					copy.Spec.Containers[0].SecurityContext.CgroupOptions = nil
					copy.Spec.InitContainers[0].SecurityContext.CgroupOptions = nil
				}),
				tweak(p, func(copy *corev1.Pod) {
					readOnly := corev1.CgroupMountModeReadOnly
					copy.Spec.Containers[0].SecurityContext.CgroupOptions = &corev1.CgroupOptions{MountMode: &readOnly}
					copy.Spec.InitContainers[0].SecurityContext.CgroupOptions = &corev1.CgroupOptions{MountMode: &readOnly}
				}),
			}
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			p = ensureSecurityContext(p)
			return []*corev1.Pod{
				tweak(p, func(copy *corev1.Pod) {
					readWrite := corev1.CgroupMountModeReadWrite
					copy.Spec.Containers[0].SecurityContext.CgroupOptions = &corev1.CgroupOptions{MountMode: &readWrite}
				}),
				tweak(p, func(copy *corev1.Pod) {
					readWrite := corev1.CgroupMountModeReadWrite
					copy.Spec.InitContainers[0].SecurityContext.CgroupOptions = &corev1.CgroupOptions{MountMode: &readWrite}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelRestricted, version: api.MajorMinorVersion(1, 0), check: "cgroupOptions"},
		fixtureData1_0,
	)
}
