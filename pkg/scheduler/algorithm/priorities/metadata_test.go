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
	"fmt"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/pkg/scheduler/listers/fake"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPriorityMetadata(t *testing.T) {
	nonZeroReqs := &schedulernodeinfo.Resource{}
	nonZeroReqs.MilliCPU = priorityutil.DefaultMilliCPURequest
	nonZeroReqs.Memory = priorityutil.DefaultMemoryRequest

	specifiedReqs := &schedulernodeinfo.Resource{}
	specifiedReqs.MilliCPU = 200
	specifiedReqs.Memory = 2000

	nonPodLimits := &schedulernodeinfo.Resource{}

	specifiedPodLimits := &schedulernodeinfo.Resource{}
	specifiedPodLimits.MilliCPU = 200
	specifiedPodLimits.Memory = 2000

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
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("200m"),
							v1.ResourceMemory: resource.MustParse("2000"),
						},
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
				podLimits:      nonPodLimits,
				podTolerations: tolerations,
				affinity:       podAffinity,
				podSelector:    labels.NewSelector(),
			},
			name: "Produce a priorityMetadata with default requests",
		},
		{
			pod: podWithTolerationsAndRequests,
			expected: &priorityMetadata{
				podLimits:      nonPodLimits,
				podTolerations: tolerations,
				affinity:       nil,
				podSelector:    labels.NewSelector(),
			},
			name: "Produce a priorityMetadata with tolerations and requests",
		},
		{
			pod: podWithAffinityAndRequests,
			expected: &priorityMetadata{
				podLimits:      specifiedPodLimits,
				podTolerations: nil,
				affinity:       podAffinity,
				podSelector:    labels.NewSelector(),
			},
			name: "Produce a priorityMetadata with affinity and requests",
		},
	}
	client := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	metaDataProducer := NewMetadataFactory(
		informerFactory.Core().V1().Services().Lister(),
		informerFactory.Core().V1().ReplicationControllers().Lister(),
		informerFactory.Apps().V1().ReplicaSets().Lister(),
		informerFactory.Apps().V1().StatefulSets().Lister(),
		1,
	)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ptData := metaDataProducer(test.pod, nil, nil)
			if !reflect.DeepEqual(test.expected, ptData) {
				t.Errorf("expected %#v, got %#v", test.expected, ptData)
			}
		})
	}
}

func BenchmarkTestGetSelector(b *testing.B) {
	pod := st.MakePod().Name("pod-0").Label("service", "service-0").Obj()
	pod.OwnerReferences = controllerRef("apps/v1", "ReplicaSet", "pod-0", "abc123")
	var (
		services []*v1.Service
		rcs      []*v1.ReplicationController
		rss      []*appsv1.ReplicaSet
		stss     []*appsv1.StatefulSet
	)

	for i := 0; i < 1000; i++ {
		service := v1.Service{
			Spec: v1.ServiceSpec{
				Selector: map[string]string{"service": fmt.Sprintf("service-%d", i)},
			},
		}
		services = append(services, &service)

		rc := v1.ReplicationController{
			Spec: v1.ReplicationControllerSpec{
				Selector: map[string]string{"rc": fmt.Sprintf("rc-%d", i)},
			},
		}
		rcs = append(rcs, &rc)

		rs := appsv1.ReplicaSet{
			Spec: appsv1.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"rs": fmt.Sprintf("rs-%d", i)},
				},
			},
		}
		rss = append(rss, &rs)

		sts := appsv1.StatefulSet{
			Spec: appsv1.StatefulSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"sts": fmt.Sprintf("sts-%d", i)},
				},
			},
		}
		stss = append(stss, &sts)
	}
	ss := SelectorSpread{
		serviceLister:     fake.ServiceLister(services),
		controllerLister:  fake.ControllerLister(rcs),
		replicaSetLister:  fake.ReplicaSetLister(rss),
		statefulSetLister: fake.StatefulSetLister(stss),
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = getSelector(pod, ss.serviceLister, ss.controllerLister, ss.replicaSetLister, ss.statefulSetLister)
	}
}
