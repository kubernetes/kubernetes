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

package containerulimits

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
)

type fakeFeatureGate struct {
	features map[string]bool
}

func (m *fakeFeatureGate) Enabled(key string) bool {
	return m.features[key]
}

func TestDiscover(t *testing.T) {
	tests := []struct {
		name string
		cfg  *ndf.NodeConfiguration
		want bool
	}{
		{
			name: "runtime supports container ulimits",
			cfg: &ndf.NodeConfiguration{
				FeatureGates:    &fakeFeatureGate{features: map[string]bool{ContainerUlimitsFeatureGate: true}},
				RuntimeFeatures: ndf.RuntimeFeatures{ContainerUlimits: true},
			},
			want: true,
		},
		{
			name: "runtime does not support container ulimits",
			cfg: &ndf.NodeConfiguration{
				FeatureGates:    &fakeFeatureGate{features: map[string]bool{ContainerUlimitsFeatureGate: true}},
				RuntimeFeatures: ndf.RuntimeFeatures{ContainerUlimits: false},
			},
			want: false,
		},
		{
			name: "feature gate disabled",
			cfg: &ndf.NodeConfiguration{
				FeatureGates:    &fakeFeatureGate{features: map[string]bool{ContainerUlimitsFeatureGate: false}},
				RuntimeFeatures: ndf.RuntimeFeatures{ContainerUlimits: true},
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, Feature.Discover(tc.cfg))
		})
	}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name string
		pod  *v1.Pod
		want bool
	}{
		{
			name: "container sets ulimits",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "c1", SecurityContext: &v1.SecurityContext{Ulimits: []v1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}}}},
				},
			}},
			want: true,
		},
		{
			name: "init container sets ulimits",
			pod: &v1.Pod{Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{Name: "i1", SecurityContext: &v1.SecurityContext{Ulimits: []v1.Ulimit{{Name: "memlock", Soft: 1024, Hard: 2048}}}},
				},
			}},
			want: true,
		},
		{
			name: "ephemeral container sets ulimits",
			pod: &v1.Pod{Spec: v1.PodSpec{
				EphemeralContainers: []v1.EphemeralContainer{
					{
						EphemeralContainerCommon: v1.EphemeralContainerCommon{
							Name:            "e1",
							SecurityContext: &v1.SecurityContext{Ulimits: []v1.Ulimit{{Name: "core", Soft: 1, Hard: 1}}},
						},
					},
				},
			}},
			want: true,
		},
		{
			name: "pod does not use ulimits",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "c1"}},
			}},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, Feature.InferForScheduling(&ndf.PodInfo{Spec: &tc.pod.Spec}))
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	oldPod := &v1.Pod{Spec: v1.PodSpec{
		Containers: []v1.Container{{Name: "c1"}},
	}}
	newPodWithUlimits := &v1.Pod{Spec: v1.PodSpec{
		Containers: []v1.Container{
			{Name: "c1", SecurityContext: &v1.SecurityContext{Ulimits: []v1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}}}},
		},
	}}
	newPodWithoutUlimits := &v1.Pod{Spec: v1.PodSpec{
		Containers: []v1.Container{{Name: "c1"}},
	}}

	assert.True(t, Feature.InferForUpdate(&ndf.PodInfo{Spec: &oldPod.Spec}, &ndf.PodInfo{Spec: &newPodWithUlimits.Spec}))
	assert.False(t, Feature.InferForUpdate(&ndf.PodInfo{Spec: &newPodWithUlimits.Spec}, &ndf.PodInfo{Spec: &newPodWithUlimits.Spec}))
	assert.False(t, Feature.InferForUpdate(&ndf.PodInfo{Spec: &oldPod.Spec}, &ndf.PodInfo{Spec: &newPodWithoutUlimits.Spec}))
}
