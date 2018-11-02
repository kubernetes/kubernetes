/*
Copyright 2017 The Kubernetes Authors.

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

package priorities

import (
	"reflect"
	"testing"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPriorityMetadata(t *testing.T) {
	nonZeroReqs := &schedulercache.Resource{}
	nonZeroReqs.MilliCPU = priorityutil.DefaultMilliCPURequest
	nonZeroReqs.Memory = priorityutil.DefaultMemoryRequest

	specifiedReqs := &schedulercache.Resource{}
	specifiedReqs.MilliCPU = 200
	specifiedReqs.Memory = 2000

	tolerations := []v1.Toleration{{
		Key:      "foo",
		Operator: v1.TolerationOpEqual,
		Value:    "bar",
		Effect:   v1.TaintEffectPreferNoSchedule,
	}}
	podAffinity := &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					Weight: 5,
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
	}
	podWithTolerationsAndAffinity := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
				},
			},
			Affinity:    podAffinity,
			Tolerations: tolerations,
		},
	}
	podWithTolerationsAndRequests := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("200m"),
							v1.ResourceMemory: resource.MustParse("2000"),
						},
					},
				},
			},
			Tolerations: tolerations,
		},
	}
	podWithAffinityAndRequests := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("200m"),
							v1.ResourceMemory: resource.MustParse("2000"),
						},
					},
				},
			},
			Affinity: podAffinity,
		},
	}
	tests := []struct {
		pod      *v1.Pod
		name     string
		expected interface{}
	}{
		{
			pod:      nil,
			expected: nil,
			name:     "pod is nil , priorityMetadata is nil",
		},
		{
			pod: podWithTolerationsAndAffinity,
			expected: &priorityMetadata{
				nonZeroRequest: nonZeroReqs,
				podTolerations: tolerations,
				affinity:       podAffinity,
			},
			name: "Produce a priorityMetadata with default requests",
		},
		{
			pod: podWithTolerationsAndRequests,
			expected: &priorityMetadata{
				nonZeroRequest: specifiedReqs,
				podTolerations: tolerations,
				affinity:       nil,
			},
			name: "Produce a priorityMetadata with specified requests",
		},
		{
			pod: podWithAffinityAndRequests,
			expected: &priorityMetadata{
				nonZeroRequest: specifiedReqs,
				podTolerations: nil,
				affinity:       podAffinity,
			},
			name: "Produce a priorityMetadata with specified requests",
		},
	}
	metaDataProducer := NewPriorityMetadataFactory(
		schedulertesting.FakeServiceLister([]*v1.Service{}),
		schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
		schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
		schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ptData := metaDataProducer(test.pod, nil)
			if !reflect.DeepEqual(test.expected, ptData) {
				t.Errorf("expected %#v, got %#v", test.expected, ptData)
			}
		})
	}
}
