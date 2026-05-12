/*
Copyright 2025 The Kubernetes Authors.

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

package framework

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
)

func TestHostPortsSigner(t *testing.T) {
	tests := []struct {
		name string
		pod  *v1.Pod
		want []int32
	}{
		{
			name: "no containers",
			pod:  &v1.Pod{},
			want: []int32{},
		},
		{
			name: "containers without host ports",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Ports: []v1.ContainerPort{{ContainerPort: 80}}},
					},
				},
			},
			want: []int32{},
		},
		{
			name: "single container with host port",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Ports: []v1.ContainerPort{{HostPort: 8080}}},
					},
				},
			},
			want: []int32{8080},
		},
		{
			name: "multiple containers, unsorted host ports, duplicates",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{Ports: []v1.ContainerPort{{HostPort: 9090}}},
					},
					Containers: []v1.Container{
						{Ports: []v1.ContainerPort{{HostPort: 80}}},
						{Ports: []v1.ContainerPort{{HostPort: 443}, {HostPort: 80}}}, // duplicate 80
					},
				},
			},
			want: []int32{80, 443, 9090},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := HostPortsSigner(tt.pod)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HostPortsSigner() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTolerationsSigner(t *testing.T) {
	tests := []struct {
		name string
		pod  *v1.Pod
		want []v1.Toleration
	}{
		{
			name: "no tolerations",
			pod:  &v1.Pod{},
			want: []v1.Toleration{},
		},
		{
			name: "single toleration",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Tolerations: []v1.Toleration{
						{Key: "key1", Value: "value1", Effect: v1.TaintEffectNoSchedule},
					},
				},
			},
			want: []v1.Toleration{
				{Key: "key1", Value: "value1", Effect: v1.TaintEffectNoSchedule},
			},
		},
		{
			name: "multiple tolerations, unsorted",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Tolerations: []v1.Toleration{
						{Key: "b", Value: "2"},
						{Key: "a", Value: "1"},
						{Key: "b", Value: "1"},
					},
				},
			},
			want: []v1.Toleration{
				{Key: "a", Value: "1"},
				{Key: "b", Value: "1"},
				{Key: "b", Value: "2"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TolerationsSigner(tt.pod)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TolerationsSigner() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolumesSigner(t *testing.T) {
	hostPath := v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/tmp"}}
	emptyDir := v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}
	configMap := v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "cm"}}}
	secret := v1.VolumeSource{Secret: &v1.SecretVolumeSource{SecretName: "secret"}}

	marshal := func(vs v1.VolumeSource) string {
		b, _ := json.Marshal(vs)
		return string(b)
	}

	tests := []struct {
		name string
		pod  *v1.Pod
		want []string
	}{
		{
			name: "no volumes",
			pod:  &v1.Pod{},
			want: []string{},
		},
		{
			name: "only ignored volumes (ConfigMap, Secret)",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{Name: "v1", VolumeSource: configMap},
						{Name: "v2", VolumeSource: secret},
					},
				},
			},
			want: []string{},
		},
		{
			name: "mixed volumes, should be filtered and sorted",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{Name: "v1", VolumeSource: hostPath},  // e.g. {"hostPath":{"path":"/tmp"}}
						{Name: "v2", VolumeSource: configMap}, // ignored
						{Name: "v3", VolumeSource: emptyDir},  // e.g. {"emptyDir":{}}
					},
				},
			},
			// Expected sort order depends on the exact JSON string.
			// {"emptyDir":{}} comes before {"hostPath":...} alphabetically.
			want: []string{marshal(emptyDir), marshal(hostPath)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := VolumesSigner(tt.pod)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("VolumesSigner() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodeAffinitySigner(t *testing.T) {
	table := []struct {
		name        string
		input       *v1.Pod
		expected    any
		expectedErr error
	}{
		{
			name:        "nil affinity",
			input:       &v1.Pod{},
			expected:    nil,
			expectedErr: nil,
		},
		{
			name: "empty affinity",
			input: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{},
						},
					},
				},
			},
			expected:    nodeAffinitySignerResult{Required: []string{}, Preferred: []string{}},
			expectedErr: nil,
		},
		{
			name: "affinity unsorted",
			input: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{Key: "kk3", Operator: v1.NodeSelectorOpIn, Values: []string{"v3", "kv4"}},
											{Key: "kk2", Operator: v1.NodeSelectorOpIn, Values: []string{"kv1", "v2"}},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "kk1", Operator: v1.NodeSelectorOpIn, Values: []string{"kv3", "v4"}},
										},
									},
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{Key: "k2", Operator: v1.NodeSelectorOpIn, Values: []string{"v1", "v2"}},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "k1", Operator: v1.NodeSelectorOpIn, Values: []string{"v3", "v4"}},
										},
									},
								},
							},
							PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
								{
									Weight: 3,
									Preference: v1.NodeSelectorTerm{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{Key: "ppk2", Operator: v1.NodeSelectorOpIn, Values: []string{"ppv1", "v2"}},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "ppk1", Operator: v1.NodeSelectorOpIn, Values: []string{"ppv3", "v4"}},
										},
									},
								},
								{
									Weight: 1,
									Preference: v1.NodeSelectorTerm{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{Key: "pk2", Operator: v1.NodeSelectorOpIn, Values: []string{"pv1", "v2"}},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "pk1", Operator: v1.NodeSelectorOpIn, Values: []string{"pv3", "v4"}},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: nodeAffinitySignerResult{
				Required: []string{
					`{"MatchExpressions":["{\"key\":\"k2\",\"operator\":\"In\",\"values\":[\"v1\",\"v2\"]}"],"MatchFields":["{\"key\":\"k1\",\"operator\":\"In\",\"values\":[\"v3\",\"v4\"]}"]}`,
					`{"MatchExpressions":["{\"key\":\"kk2\",\"operator\":\"In\",\"values\":[\"kv1\",\"v2\"]}","{\"key\":\"kk3\",\"operator\":\"In\",\"values\":[\"kv4\",\"v3\"]}"],"MatchFields":["{\"key\":\"kk1\",\"operator\":\"In\",\"values\":[\"kv3\",\"v4\"]}"]}`,
				},
				Preferred: []string{
					`{"Weight":1,"Preference":{"MatchExpressions":["{\"key\":\"pk2\",\"operator\":\"In\",\"values\":[\"pv1\",\"v2\"]}"],"MatchFields":["{\"key\":\"pk1\",\"operator\":\"In\",\"values\":[\"pv3\",\"v4\"]}"]}}`,
					`{"Weight":3,"Preference":{"MatchExpressions":["{\"key\":\"ppk2\",\"operator\":\"In\",\"values\":[\"ppv1\",\"v2\"]}"],"MatchFields":["{\"key\":\"ppk1\",\"operator\":\"In\",\"values\":[\"ppv3\",\"v4\"]}"]}}`,
				},
			},
			expectedErr: nil,
		},
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			res, err := NodeAffinitySigner(tt.input)
			if !errors.Is(err, tt.expectedErr) {
				t.Fatalf("unexpected error %v, expected %v", err, tt.expectedErr)
			}
			if diff := cmp.Diff(res, tt.expected); diff != "" {
				t.Fatalf("unexpected result %s", diff)
			}
		})
	}
}
