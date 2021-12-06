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

package pod

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func BenchmarkNoWarnings(b *testing.B) {
	ctx := context.TODO()
	resources := api.ResourceList{
		api.ResourceCPU:              resource.MustParse("100m"),
		api.ResourceMemory:           resource.MustParse("4M"),
		api.ResourceEphemeralStorage: resource.MustParse("4G"),
	}
	env := []api.EnvVar{
		{Name: "a"},
		{Name: "b"},
	}
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{`foo`: `bar`},
		},
		Spec: api.PodSpec{
			NodeSelector: map[string]string{"foo": "bar", "baz": "quux"},
			Affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{MatchExpressions: []api.NodeSelectorRequirement{{Key: `foo`}}},
						},
					},
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Preference: api.NodeSelectorTerm{MatchExpressions: []api.NodeSelectorRequirement{{Key: `foo`}}}},
					},
				},
			},
			TopologySpreadConstraints: []api.TopologySpreadConstraint{
				{TopologyKey: `foo`},
			},
			HostAliases: []api.HostAlias{
				{IP: "1.1.1.1"},
				{IP: "2.2.2.2"},
			},
			ImagePullSecrets: []api.LocalObjectReference{
				{Name: "secret1"},
				{Name: "secret2"},
			},
			InitContainers: []api.Container{
				{Name: "init1", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
				{Name: "init2", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
			},
			Containers: []api.Container{
				{Name: "container1", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
				{Name: "container2", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
			},
			Overhead: resources,
			Volumes: []api.Volume{
				{Name: "a"},
				{Name: "b"},
			},
		},
	}
	oldPod := &api.Pod{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := GetWarningsForPod(ctx, pod, oldPod)
		if len(w) > 0 {
			b.Fatalf("expected 0 warnings, got %q", w)
		}
	}
}

func BenchmarkWarnings(b *testing.B) {
	ctx := context.TODO()
	resources := api.ResourceList{
		api.ResourceCPU:              resource.MustParse("100m"),
		api.ResourceMemory:           resource.MustParse("4m"),
		api.ResourceEphemeralStorage: resource.MustParse("4m"),
	}
	env := []api.EnvVar{
		{Name: "a"},
		{Name: "a"},
	}
	pod := &api.Pod{
		Spec: api.PodSpec{
			HostAliases: []api.HostAlias{
				{IP: "1.1.1.1"},
				{IP: "1.1.1.1"},
			},
			ImagePullSecrets: []api.LocalObjectReference{
				{Name: "secret1"},
				{Name: "secret1"},
				{Name: ""},
			},
			InitContainers: []api.Container{
				{Name: "init1", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
				{Name: "init2", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
			},
			Containers: []api.Container{
				{Name: "container1", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
				{Name: "container2", Env: env, Resources: api.ResourceRequirements{Requests: resources, Limits: resources}},
			},
			Overhead: resources,
			Volumes: []api.Volume{
				{Name: "a"},
				{Name: "a"},
			},
		},
	}
	oldPod := &api.Pod{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GetWarningsForPod(ctx, pod, oldPod)
	}
}

func TestWarnings(t *testing.T) {
	resources := api.ResourceList{
		api.ResourceCPU:              resource.MustParse("100m"),
		api.ResourceMemory:           resource.MustParse("4m"),
		api.ResourceEphemeralStorage: resource.MustParse("4m"),
	}
	testcases := []struct {
		name     string
		template *api.PodTemplateSpec
		expected []string
	}{
		{
			name:     "null",
			template: nil,
			expected: nil,
		},
		{
			name: "photon",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "p", VolumeSource: api.VolumeSource{PhotonPersistentDisk: &api.PhotonPersistentDiskVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].photonPersistentDisk: deprecated in v1.11, non-functional in v1.16+`},
		},
		{
			name: "gitRepo",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "s", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].gitRepo: deprecated in v1.11`},
		},
		{
			name: "scaleIO",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "s", VolumeSource: api.VolumeSource{ScaleIO: &api.ScaleIOVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].scaleIO: deprecated in v1.16, non-functional in v1.22+`},
		},
		{
			name: "flocker",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "s", VolumeSource: api.VolumeSource{Flocker: &api.FlockerVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].flocker: deprecated in v1.22, support removal is planned in v1.26`},
		},
		{
			name: "storageOS",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "s", VolumeSource: api.VolumeSource{StorageOS: &api.StorageOSVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].storageOS: deprecated in v1.22, support removal is planned in v1.26`},
		},
		{
			name: "quobyte",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "s", VolumeSource: api.VolumeSource{Quobyte: &api.QuobyteVolumeSource{}}},
					},
				},
			},
			expected: []string{`spec.volumes[0].quobyte: deprecated in v1.22, support removal is planned in v1.26`},
		},
		{
			name: "duplicate hostAlias",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					HostAliases: []api.HostAlias{
						{IP: "1.1.1.1"},
						{IP: "1.1.1.1"},
						{IP: "1.1.1.1"},
					},
				},
			},
			expected: []string{
				`spec.hostAliases[1].ip: duplicate ip "1.1.1.1"`,
				`spec.hostAliases[2].ip: duplicate ip "1.1.1.1"`,
			},
		},
		{
			name: "duplicate imagePullSecret",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					ImagePullSecrets: []api.LocalObjectReference{
						{Name: "a"},
						{Name: "a"},
						{Name: "a"},
					},
				},
			},
			expected: []string{
				`spec.imagePullSecrets[1].name: duplicate name "a"`,
				`spec.imagePullSecrets[2].name: duplicate name "a"`,
			},
		},
		{
			name: "empty imagePullSecret",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					ImagePullSecrets: []api.LocalObjectReference{
						{Name: ""},
					},
				},
			},
			expected: []string{
				`spec.imagePullSecrets[0].name: invalid empty name ""`,
			},
		},
		{
			name: "duplicate volume",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{Name: "a"},
						{Name: "a"},
						{Name: "a"},
					},
				},
			},
			expected: []string{
				`spec.volumes[1].name: duplicate name "a"`,
				`spec.volumes[2].name: duplicate name "a"`,
			},
		},
		{
			name: "duplicate env",
			template: &api.PodTemplateSpec{Spec: api.PodSpec{
				InitContainers: []api.Container{{Env: []api.EnvVar{
					{Name: "a"},
					{Name: "a"},
					{Name: "a"},
				}}},
				Containers: []api.Container{{Env: []api.EnvVar{
					{Name: "b"},
					{Name: "b"},
					{Name: "b"},
				}}},
			}},
			expected: []string{
				`spec.initContainers[0].env[1].name: duplicate name "a"`,
				`spec.initContainers[0].env[2].name: duplicate name "a"`,
				`spec.containers[0].env[1].name: duplicate name "b"`,
				`spec.containers[0].env[2].name: duplicate name "b"`,
			},
		},
		{
			name: "fractional resources",
			template: &api.PodTemplateSpec{Spec: api.PodSpec{
				InitContainers: []api.Container{{
					Resources: api.ResourceRequirements{Requests: resources, Limits: resources},
				}},
				Containers: []api.Container{{
					Resources: api.ResourceRequirements{Requests: resources, Limits: resources},
				}},
				Overhead: resources,
			}},
			expected: []string{
				`spec.initContainers[0].resources.requests[ephemeral-storage]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.initContainers[0].resources.requests[memory]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.initContainers[0].resources.limits[ephemeral-storage]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.initContainers[0].resources.limits[memory]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.containers[0].resources.requests[ephemeral-storage]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.containers[0].resources.requests[memory]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.containers[0].resources.limits[ephemeral-storage]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.containers[0].resources.limits[memory]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.overhead[ephemeral-storage]: fractional byte value "4m" is invalid, must be an integer`,
				`spec.overhead[memory]: fractional byte value "4m" is invalid, must be an integer`,
			},
		},
		{
			name: "node labels in nodeSelector",
			template: &api.PodTemplateSpec{Spec: api.PodSpec{
				NodeSelector: map[string]string{
					`beta.kubernetes.io/arch`:                  `true`,
					`beta.kubernetes.io/os`:                    `true`,
					`failure-domain.beta.kubernetes.io/region`: `true`,
					`failure-domain.beta.kubernetes.io/zone`:   `true`,
					`beta.kubernetes.io/instance-type`:         `true`,
				},
			}},
			expected: []string{
				`spec.nodeSelector[beta.kubernetes.io/arch]: deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`spec.nodeSelector[beta.kubernetes.io/instance-type]: deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
				`spec.nodeSelector[beta.kubernetes.io/os]: deprecated since v1.14; use "kubernetes.io/os" instead`,
				`spec.nodeSelector[failure-domain.beta.kubernetes.io/region]: deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
				`spec.nodeSelector[failure-domain.beta.kubernetes.io/zone]: deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
			},
		},
		{
			name: "node labels in affinity requiredDuringSchedulingIgnoredDuringExecution",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Affinity: &api.Affinity{
						NodeAffinity: &api.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
								NodeSelectorTerms: []api.NodeSelectorTerm{
									{
										MatchExpressions: []api.NodeSelectorRequirement{
											{Key: `foo`},
											{Key: `beta.kubernetes.io/arch`},
											{Key: `beta.kubernetes.io/os`},
											{Key: `failure-domain.beta.kubernetes.io/region`},
											{Key: `failure-domain.beta.kubernetes.io/zone`},
											{Key: `beta.kubernetes.io/instance-type`},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []string{
				`spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].key: beta.kubernetes.io/arch is deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[2].key: beta.kubernetes.io/os is deprecated since v1.14; use "kubernetes.io/os" instead`,
				`spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[3].key: failure-domain.beta.kubernetes.io/region is deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
				`spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[4].key: failure-domain.beta.kubernetes.io/zone is deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
				`spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[5].key: beta.kubernetes.io/instance-type is deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
			},
		},
		{
			name: "node labels in affinity preferredDuringSchedulingIgnoredDuringExecution",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Affinity: &api.Affinity{
						NodeAffinity: &api.NodeAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
								{
									Preference: api.NodeSelectorTerm{
										MatchExpressions: []api.NodeSelectorRequirement{
											{Key: `foo`},
											{Key: `beta.kubernetes.io/arch`},
											{Key: `beta.kubernetes.io/os`},
											{Key: `failure-domain.beta.kubernetes.io/region`},
											{Key: `failure-domain.beta.kubernetes.io/zone`},
											{Key: `beta.kubernetes.io/instance-type`},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []string{
				`spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].preference.matchExpressions[1].key: beta.kubernetes.io/arch is deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].preference.matchExpressions[2].key: beta.kubernetes.io/os is deprecated since v1.14; use "kubernetes.io/os" instead`,
				`spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].preference.matchExpressions[3].key: failure-domain.beta.kubernetes.io/region is deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
				`spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].preference.matchExpressions[4].key: failure-domain.beta.kubernetes.io/zone is deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
				`spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].preference.matchExpressions[5].key: beta.kubernetes.io/instance-type is deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
			},
		},
		{
			name: "node labels in topologySpreadConstraints",
			template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					TopologySpreadConstraints: []api.TopologySpreadConstraint{
						{TopologyKey: `foo`},
						{TopologyKey: `beta.kubernetes.io/arch`},
						{TopologyKey: `beta.kubernetes.io/os`},
						{TopologyKey: `failure-domain.beta.kubernetes.io/region`},
						{TopologyKey: `failure-domain.beta.kubernetes.io/zone`},
						{TopologyKey: `beta.kubernetes.io/instance-type`},
					},
				},
			},
			expected: []string{
				`spec.topologySpreadConstraints[1].topologyKey: beta.kubernetes.io/arch is deprecated since v1.14; use "kubernetes.io/arch" instead`,
				`spec.topologySpreadConstraints[2].topologyKey: beta.kubernetes.io/os is deprecated since v1.14; use "kubernetes.io/os" instead`,
				`spec.topologySpreadConstraints[3].topologyKey: failure-domain.beta.kubernetes.io/region is deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
				`spec.topologySpreadConstraints[4].topologyKey: failure-domain.beta.kubernetes.io/zone is deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
				`spec.topologySpreadConstraints[5].topologyKey: beta.kubernetes.io/instance-type is deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
			},
		},
		{
			name: "annotations",
			template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					`foo`: `bar`,
					`scheduler.alpha.kubernetes.io/critical-pod`:         `true`,
					`seccomp.security.alpha.kubernetes.io/pod`:           `default`,
					`container.seccomp.security.alpha.kubernetes.io/foo`: `default`,
					`security.alpha.kubernetes.io/sysctls`:               `a,b,c`,
					`security.alpha.kubernetes.io/unsafe-sysctls`:        `d,e,f`,
				}},
				Spec: api.PodSpec{Containers: []api.Container{{Name: "foo"}}},
			},
			expected: []string{
				`metadata.annotations[scheduler.alpha.kubernetes.io/critical-pod]: non-functional in v1.16+; use the "priorityClassName" field instead`,
				`metadata.annotations[seccomp.security.alpha.kubernetes.io/pod]: deprecated since v1.19, non-functional in v1.25+; use the "seccompProfile" field instead`,
				`metadata.annotations[container.seccomp.security.alpha.kubernetes.io/foo]: deprecated since v1.19, non-functional in v1.25+; use the "seccompProfile" field instead`,
				`metadata.annotations[security.alpha.kubernetes.io/sysctls]: non-functional in v1.11+; use the "sysctls" field instead`,
				`metadata.annotations[security.alpha.kubernetes.io/unsafe-sysctls]: non-functional in v1.11+; use the "sysctls" field instead`,
			},
		},
		{
			name: "seccomp fields",
			template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					`seccomp.security.alpha.kubernetes.io/pod`:           `default`,
					`container.seccomp.security.alpha.kubernetes.io/foo`: `default`,
				}},
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						SeccompProfile: &api.SeccompProfile{Type: api.SeccompProfileTypeRuntimeDefault},
					},
					Containers: []api.Container{{
						Name: "foo",
						SecurityContext: &api.SecurityContext{
							SeccompProfile: &api.SeccompProfile{Type: api.SeccompProfileTypeRuntimeDefault},
						},
					}},
				},
			},
			expected: []string{},
		},
	}

	for _, tc := range testcases {
		t.Run("podspec_"+tc.name, func(t *testing.T) {
			actual := sets.NewString(GetWarningsForPodTemplate(context.TODO(), nil, tc.template, &api.PodTemplateSpec{})...)
			expected := sets.NewString(tc.expected...)
			for _, missing := range expected.Difference(actual).List() {
				t.Errorf("missing: %s", missing)
			}
			for _, extra := range actual.Difference(expected).List() {
				t.Errorf("extra: %s", extra)
			}
		})

		t.Run("pod_"+tc.name, func(t *testing.T) {
			var pod *api.Pod
			if tc.template != nil {
				pod = &api.Pod{
					ObjectMeta: tc.template.ObjectMeta,
					Spec:       tc.template.Spec,
				}
			}
			actual := sets.NewString(GetWarningsForPod(context.TODO(), pod, &api.Pod{})...)
			expected := sets.NewString(tc.expected...)
			for _, missing := range expected.Difference(actual).List() {
				t.Errorf("missing: %s", missing)
			}
			for _, extra := range actual.Difference(expected).List() {
				t.Errorf("extra: %s", extra)
			}
		})
	}
}
