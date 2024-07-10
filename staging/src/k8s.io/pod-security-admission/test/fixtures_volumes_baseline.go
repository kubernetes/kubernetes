/*
Copyright 2024 The Kubernetes Authors.

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

func init() {

	fixtureData1_31 := fixtureGenerator{
		expectErrorSubstring: "restricted volume types",
		generatePass: func(p *corev1.Pod) []*corev1.Pod {
			// minimal valid pod already captures all valid combinations
			return nil
		},
		generateFail: func(p *corev1.Pod) []*corev1.Pod {
			return []*corev1.Pod{
				tweak(p, func(p *corev1.Pod) {
					p.Spec.Volumes = []corev1.Volume{{Name: "volume-git-repo", VolumeSource: corev1.VolumeSource{GitRepo: &corev1.GitRepoVolumeSource{Repository: "github.com/kubernetes/kubernetes"}}}}
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 31), check: "baselineVolumes"},
		fixtureData1_31,
	)
}
