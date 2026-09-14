/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndftesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

var nodeInfoCmpOpts = []cmp.Option{
	cmp.AllowUnexported(NodeInfo{}, PodInfo{}, fwk.PodResource{}),
}

func TestNewResource(t *testing.T) {
	tests := []struct {
		name         string
		resourceList v1.ResourceList
		expected     *Resource
	}{
		{
			name:         "empty resource",
			resourceList: map[v1.ResourceName]resource.Quantity{},
			expected:     &Resource{},
		},
		{
			name: "complex resource",
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:                      *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:                   *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourcePods:                     *resource.NewQuantity(80, resource.BinarySI),
				v1.ResourceEphemeralStorage:         *resource.NewQuantity(5000, resource.BinarySI),
				"scalar.test/" + "scalar1":          *resource.NewQuantity(1, resource.DecimalSI),
				v1.ResourceHugePagesPrefix + "test": *resource.NewQuantity(2, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			r := NewResource(test.resourceList)
			if diff := cmp.Diff(test.expected, r); diff != "" {
				t.Errorf("Unexpected resource (-want, +got):\n%s", diff)
			}
		})
	}
}

var (
	midPriority  = int32(100)
	highPriority = int32(1000)
)

func TestPodGroupMemberPodsOrderingFunc(t *testing.T) {
	timestamp := time.Now()
	timestampNewer := timestamp.Add(time.Second)

	// Desired order: pod3 > pod5 > pod1 > pod4 > pod2.
	pInfo1 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").UID("uid1").Priority(midPriority).Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo2 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Name("pod2").UID("uid2").Priority(midPriority).Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestampNewer,
		},
	}
	pInfo3 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Name("pod3").UID("uid3").Priority(highPriority).Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo4 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Name("pod4").UID("uid4").Priority(midPriority).Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo5 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Name("pod5").UID("uid5").Priority(midPriority).Obj()},
		QueueingParams: QueueingParams{
			Attempts:  2,
			Timestamp: timestamp,
		},
	}

	tests := []struct {
		name     string
		a        *QueuedPodInfo
		b        *QueuedPodInfo
		expected int
	}{
		{
			name:     "higher priority comes first",
			a:        pInfo3,
			b:        pInfo1,
			expected: -1,
		},
		{
			name:     "lower priority comes second",
			a:        pInfo1,
			b:        pInfo3,
			expected: 1,
		},
		{
			name:     "higher attempts comes first",
			a:        pInfo5,
			b:        pInfo1,
			expected: -1,
		},
		{
			name:     "lower attempts comes second",
			a:        pInfo1,
			b:        pInfo5,
			expected: 1,
		},
		{
			name:     "older timestamp comes first",
			a:        pInfo1,
			b:        pInfo2,
			expected: -1,
		},
		{
			name:     "newer timestamp comes second",
			a:        pInfo2,
			b:        pInfo1,
			expected: 1,
		},
		{
			name:     "same priority, same attempts, same timestamp, lower name comes first",
			a:        pInfo1,
			b:        pInfo4,
			expected: -1,
		},
		{
			name:     "same priority, same attempts, same timestamp, higher name comes second",
			a:        pInfo4,
			b:        pInfo1,
			expected: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := PodGroupMemberPodsOrderingFunc(test.a, test.b)
			if got != test.expected {
				t.Errorf("Unexpected result, want %v, got %v", test.expected, got)
			}
		})
	}
}

func TestQueuedPodGroupInfoOrdering(t *testing.T) {
	timestamp := time.Now()
	timestampNewer := timestamp.Add(time.Minute)

	opts := []cmp.Option{
		cmp.AllowUnexported(QueuedPodInfo{}, PodInfo{}, fwk.PodResource{}),
	}

	// Desired order: pod3 > pod5 > pod1 > pod4 > pod2.
	pInfo1 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod1").UID("uid1").Priority(midPriority).PodGroupName("pg1").Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo2 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod2").UID("uid2").Priority(midPriority).PodGroupName("pg1").Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestampNewer,
		},
	}
	pInfo3 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod3").UID("uid3").Priority(highPriority).PodGroupName("pg1").Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo4 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod4").UID("uid4").Priority(midPriority).PodGroupName("pg1").Obj()},
		QueueingParams: QueueingParams{
			Attempts:  1,
			Timestamp: timestamp,
		},
	}
	pInfo5 := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod5").UID("uid5").Priority(midPriority).PodGroupName("pg1").Obj()},
		QueueingParams: QueueingParams{
			Attempts:  2,
			Timestamp: timestamp,
		},
	}

	tests := []struct {
		name          string
		podsToAdd     []*QueuedPodInfo
		podToRemove   *QueuedPodInfo
		expectedOrder []*QueuedPodInfo
	}{
		{
			name:          "Add high priority pod to empty group",
			podsToAdd:     []*QueuedPodInfo{pInfo3},
			expectedOrder: []*QueuedPodInfo{pInfo3},
		},
		{
			name:          "Add lower priority pod, goes to end",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo1},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo1},
		},
		{
			name:          "Add pod with higher priority to front",
			podsToAdd:     []*QueuedPodInfo{pInfo1, pInfo2, pInfo3},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo1, pInfo2},
		},
		{
			name:          "Add pod with same priority but lower attempts, goes to end",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo1},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo1},
		},
		{
			name:          "Add pod with same priority but higher attempts, goes before",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo1, pInfo5},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo1},
		},
		{
			name:          "Add pod with same priority but later timestamp, goes to end",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo2},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo2},
		},
		{
			name:          "Add pod with same priority but earlier timestamp, goes before",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo2, pInfo1},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo1, pInfo2},
		},
		{
			name:          "Add pod with same priority and timestamp, ordered by name",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo2, pInfo4},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo4, pInfo2},
		},
		{
			name:          "Add pods out of order, gets sorted",
			podsToAdd:     []*QueuedPodInfo{pInfo1, pInfo2, pInfo3, pInfo4, pInfo5},
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo4, pInfo2},
		},
		{
			name:          "Remove pod from middle",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo2},
			podToRemove:   pInfo1,
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo2},
		},
		{
			name:          "Remove first pod",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo2},
			podToRemove:   pInfo3,
			expectedOrder: []*QueuedPodInfo{pInfo5, pInfo1, pInfo2},
		},
		{
			name:          "Remove last pod",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo5, pInfo1, pInfo2},
			podToRemove:   pInfo2,
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo5, pInfo1},
		},
		{
			name:          "Remove non-existent pod",
			podsToAdd:     []*QueuedPodInfo{pInfo3, pInfo1},
			podToRemove:   pInfo2,
			expectedOrder: []*QueuedPodInfo{pInfo3, pInfo1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pg := st.MakePodGroup().Namespace("default").Name("pg1").Obj()
			pgqi := &QueuedPodGroupInfo{
				PodGroupInfo: &PodGroupInfo{
					GenericPodGroup: fwk.NewGenericPodGroup(pg),
				},
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			}
			for _, p := range tt.podsToAdd {
				pgqi.AddPod(p)
			}
			if tt.podToRemove != nil {
				pgqi.RemovePod(tt.podToRemove.Pod)
			}

			key := fwk.PodGroupKey("default", "pg1")
			var actualOrder []*QueuedPodInfo
			if pgqi.QueuedPodInfos != nil {
				actualOrder = pgqi.QueuedPodInfos[key]
			}

			if diff := cmp.Diff(tt.expectedOrder, actualOrder, opts...); diff != "" {
				t.Errorf("Unexpected order in QueuedPodInfos (-want, +got):\n%s", diff)
			}

			expectedUnscheduled := make([]*v1.Pod, len(tt.expectedOrder))
			for i, qpi := range tt.expectedOrder {
				expectedUnscheduled[i] = qpi.Pod
			}
			if diff := cmp.Diff(expectedUnscheduled, pgqi.UnscheduledPods); diff != "" {
				t.Errorf("Unexpected order in UnscheduledPods (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestQueuedEntityInfo_HasPodsWithPendingPlugins(t *testing.T) {
	podWithoutPending := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod1").PodGroupName("pg1").Obj()},
	}
	podWithPending1 := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod2").PodGroupName("pg1").Obj()},
		PendingPlugins: sets.New("pluginA"),
	}
	podWithPending2 := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod3").PodGroupName("pg1").Obj()},
		PendingPlugins: sets.New("pluginB"),
	}
	nonExistentPod := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("nonexistent").PodGroupName("pg1").Obj()},
	}
	podGroup := st.MakePodGroup().Namespace("default").Name("pg1").Obj()
	nonExistentPG := st.MakePodGroup().Namespace("default").Name("nonexistent-pg").Obj()

	cpgRoot := st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj()
	cpgChild := st.MakeCompositePodGroup().Namespace("default").Name("cpg-child").ParentCompositePodGroup("cpg-root").Obj()
	cpgNested := st.MakeCompositePodGroup().Namespace("default").Name("cpg-nested").ParentCompositePodGroup("cpg-child").Obj()
	nonExistentCPG := st.MakeCompositePodGroup().Namespace("default").Name("nonexistent-cpg").Obj()

	pgChild1 := st.MakePodGroup().Namespace("default").Name("pg-child1").ParentCompositePodGroup("cpg-root").Obj()
	pgChild2 := st.MakePodGroup().Namespace("default").Name("pg-child2").ParentCompositePodGroup("cpg-root").Obj()
	pgLeaf1 := st.MakePodGroup().Namespace("default").Name("pg-leaf1").ParentCompositePodGroup("cpg-nested").Obj()
	pgLeaf2 := st.MakePodGroup().Namespace("default").Name("pg-leaf2").ParentCompositePodGroup("cpg-nested").Obj()

	podChild1WithPending := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-c1").PodGroupName("pg-child1").Obj()},
		PendingPlugins: sets.New("pluginA"),
	}
	podChild1WithoutPending := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-c1-np").PodGroupName("pg-child1").Obj()},
	}
	podChild2WithPending := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-c2").PodGroupName("pg-child2").Obj()},
		PendingPlugins: sets.New("pluginB"),
	}
	podChild2WithoutPending := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-c2-np").PodGroupName("pg-child2").Obj()},
	}
	podLeaf1WithPending := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-l1").PodGroupName("pg-leaf1").Obj()},
		PendingPlugins: sets.New("pluginA"),
	}
	podLeaf2WithPending := &QueuedPodInfo{
		PodInfo:        &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-l2").PodGroupName("pg-leaf2").Obj()},
		PendingPlugins: sets.New("pluginB"),
	}
	podLeaf2WithoutPending := &QueuedPodInfo{
		PodInfo: &PodInfo{Pod: st.MakePod().Namespace("default").Name("pod-l2-np").PodGroupName("pg-leaf2").Obj()},
	}

	tests := []struct {
		name     string
		entity   QueuedEntityInfo
		expected bool
	}{
		{
			name:     "single pod without pending plugins",
			entity:   podWithoutPending,
			expected: false,
		},
		{
			name:     "single pod with pending plugins",
			entity:   podWithPending1,
			expected: true,
		},
		{
			name: "empty pod group",
			entity: &QueuedPodGroupInfo{
				PodGroupInfo: &PodGroupInfo{
					GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
				},
			},
			expected: false,
		},
		{
			name: "pod group with pod without pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithoutPending)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "pod group with pod with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithPending1)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group with multiple pods with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithoutPending)
				pgqi.AddPod(podWithPending1)
				pgqi.AddPod(podWithPending2)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group after removing non-existent pod",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithPending1)
				pgqi.RemovePod(nonExistentPod.Pod)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group after removing pod without pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithoutPending)
				pgqi.AddPod(podWithPending1)
				pgqi.RemovePod(podWithoutPending.Pod)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group after removing one of multiple pods with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithPending1)
				pgqi.AddPod(podWithPending2)
				pgqi.RemovePod(podWithPending1.Pod)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group after removing all pods with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithoutPending)
				pgqi.AddPod(podWithPending1)
				pgqi.RemovePod(podWithPending1.Pod)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "pod group after re-adding pod with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericPodGroup(podGroup),
					},
				}
				pgqi.AddPod(podWithPending1)
				pgqi.RemovePod(podWithPending1.Pod)
				pgqi.AddPod(podWithPending1)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "pod group with pending plugins after removing pod group",
			entity: func() *QueuedPodGroupInfo {
				gpg := fwk.NewGenericPodGroup(podGroup)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: gpg,
					},
				}
				pgqi.AddPod(podWithoutPending)
				pgqi.AddPod(podWithPending1)
				pgqi.RemoveGenericPodGroup(gpg)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "pod group with pending plugins after removing non-existent pod group",
			entity: func() *QueuedPodGroupInfo {
				gpg := fwk.NewGenericPodGroup(podGroup)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: gpg,
					},
				}
				pgqi.AddPod(podWithPending1)
				pgqi.RemoveGenericPodGroup(fwk.NewGenericPodGroup(nonExistentPG))
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing child pod group with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gpgChild1 := fwk.NewGenericPodGroup(pgChild1)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gpgChild1,
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild2),
							},
						},
					},
				}
				pgqi.AddPod(podChild1WithPending)
				pgqi.AddPod(podChild2WithoutPending)
				pgqi.RemoveGenericPodGroup(gpgChild1)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "composite pod group after removing child pod group with mixed pending and non-pending pods",
			entity: func() *QueuedPodGroupInfo {
				gpgChild1 := fwk.NewGenericPodGroup(pgChild1)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gpgChild1,
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild2),
							},
						},
					},
				}
				pgqi.AddPod(podChild1WithPending)
				pgqi.AddPod(podChild1WithoutPending)
				pgqi.AddPod(podChild2WithoutPending)
				pgqi.RemoveGenericPodGroup(gpgChild1)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "composite pod group after removing one of multiple child pod groups with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gpgChild1 := fwk.NewGenericPodGroup(pgChild1)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gpgChild1,
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild2),
							},
						},
					},
				}
				pgqi.AddPod(podChild1WithPending)
				pgqi.AddPod(podChild2WithPending)
				pgqi.RemoveGenericPodGroup(gpgChild1)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing child pod group without pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gpgChild2 := fwk.NewGenericPodGroup(pgChild2)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild1),
							},
							{
								GenericPodGroup: gpgChild2,
							},
						},
					},
				}
				pgqi.AddPod(podChild1WithPending)
				pgqi.AddPod(podChild2WithoutPending)
				pgqi.RemoveGenericPodGroup(gpgChild2)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing child composite pod group with nested subtree pods with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gcpgChild := fwk.NewGenericCompositePodGroup(cpgChild)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gcpgChild,
								Children: []*PodGroupInfo{
									{
										GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgNested),
										Children: []*PodGroupInfo{
											{
												GenericPodGroup: fwk.NewGenericPodGroup(pgLeaf1),
											},
											{
												GenericPodGroup: fwk.NewGenericPodGroup(pgLeaf2),
											},
										},
									},
								},
							},
						},
					},
				}
				pgqi.AddPod(podLeaf1WithPending)
				pgqi.AddPod(podLeaf2WithPending)
				pgqi.RemoveGenericPodGroup(gcpgChild)
				return pgqi
			}(),
			expected: false,
		},
		{
			name: "composite pod group after removing child composite pod group while sibling branch has pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gcpgChild := fwk.NewGenericCompositePodGroup(cpgChild)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gcpgChild,
								Children: []*PodGroupInfo{
									{
										GenericPodGroup: fwk.NewGenericPodGroup(pgLeaf1),
									},
								},
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild1),
							},
						},
					},
				}
				pgqi.AddPod(podLeaf1WithPending)
				pgqi.AddPod(podChild1WithPending)
				pgqi.RemoveGenericPodGroup(gcpgChild)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing child composite pod group without pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gcpgChild := fwk.NewGenericCompositePodGroup(cpgChild)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gcpgChild,
								Children: []*PodGroupInfo{
									{
										GenericPodGroup: fwk.NewGenericPodGroup(pgLeaf2),
									},
								},
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild1),
							},
						},
					},
				}
				pgqi.AddPod(podLeaf2WithoutPending)
				pgqi.AddPod(podChild1WithPending)
				pgqi.RemoveGenericPodGroup(gcpgChild)
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing non-existent composite pod group",
			entity: func() *QueuedPodGroupInfo {
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgChild),
								Children: []*PodGroupInfo{
									{
										GenericPodGroup: fwk.NewGenericPodGroup(pgLeaf1),
									},
								},
							},
						},
					},
				}
				pgqi.AddPod(podLeaf1WithPending)
				pgqi.RemoveGenericPodGroup(fwk.NewGenericCompositePodGroup(nonExistentCPG))
				return pgqi
			}(),
			expected: true,
		},
		{
			name: "composite pod group after removing all child pod groups with pending plugins",
			entity: func() *QueuedPodGroupInfo {
				gpgChild1 := fwk.NewGenericPodGroup(pgChild1)
				gpgChild2 := fwk.NewGenericPodGroup(pgChild2)
				pgqi := &QueuedPodGroupInfo{
					PodGroupInfo: &PodGroupInfo{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: gpgChild1,
							},
							{
								GenericPodGroup: gpgChild2,
							},
						},
					},
				}
				pgqi.AddPod(podChild1WithPending)
				pgqi.AddPod(podChild2WithPending)
				pgqi.RemoveGenericPodGroup(gpgChild1)
				pgqi.RemoveGenericPodGroup(gpgChild2)
				return pgqi
			}(),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: true,
				features.CompositePodGroup:               true,
			})
			if got := tt.entity.HasPodsWithPendingPlugins(); got != tt.expected {
				t.Errorf("Unexpected HasPodsWithPendingPlugins: %v, want: %v", got, tt.expected)
			}
		})
	}
}

func TestResourceClone(t *testing.T) {
	tests := []struct {
		resource *Resource
		expected *Resource
	}{
		{
			resource: &Resource{},
			expected: &Resource{},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			r := test.resource.Clone()
			// Modify the field to check if the result is a clone of the origin one.
			test.resource.MilliCPU += 1000
			if diff := cmp.Diff(test.expected, r); diff != "" {
				t.Errorf("Unexpected resource (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestResourceAddScalar(t *testing.T) {
	tests := []struct {
		resource       *Resource
		scalarName     v1.ResourceName
		scalarQuantity int64
		expected       *Resource
	}{
		{
			resource:       &Resource{},
			scalarName:     "scalar1",
			scalarQuantity: 100,
			expected: &Resource{
				ScalarResources: map[v1.ResourceName]int64{"scalar1": 100},
			},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"hugepages-test": 2},
			},
			scalarName:     "scalar2",
			scalarQuantity: 200,
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"hugepages-test": 2, "scalar2": 200},
			},
		},
	}

	for _, test := range tests {
		t.Run(string(test.scalarName), func(t *testing.T) {
			test.resource.AddScalar(test.scalarName, test.scalarQuantity)
			if diff := cmp.Diff(test.expected, test.resource); diff != "" {
				t.Errorf("Unexpected resource (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestSetMaxResource(t *testing.T) {
	tests := []struct {
		resource     *Resource
		resourceList v1.ResourceList
		expected     *Resource
	}{
		{
			resource: &Resource{},
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:              *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:           *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
			},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           4000,
				EphemeralStorage: 5000,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:                      *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:                   *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourceEphemeralStorage:         *resource.NewQuantity(7000, resource.BinarySI),
				"scalar.test/scalar1":               *resource.NewQuantity(4, resource.DecimalSI),
				v1.ResourceHugePagesPrefix + "test": *resource.NewQuantity(5, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           4000,
				EphemeralStorage: 7000,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 4, "hugepages-test": 5},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			test.resource.SetMaxResource(test.resourceList)
			if diff := cmp.Diff(test.expected, test.resource); diff != "" {
				t.Errorf("Unexpected resource (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestNewNodeInfo(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		st.MakePod().UID("test-1").Namespace("node_info_cache_test").Name("test-1").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100m",
				v1.ResourceMemory: "500",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 80,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),

		st.MakePod().UID("test-2").Namespace("node_info_cache_test").Name("test-2").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "200m",
				v1.ResourceMemory: "1Ki",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 8080,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),
	}

	expected := &NodeInfo{
		Requested: &Resource{
			MilliCPU:         300,
			Memory:           1524,
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		NonZeroRequested: &Resource{
			MilliCPU:         300,
			Memory:           1524,
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		Allocatable: &Resource{},
		Generation:  2,
		UsedPorts: fwk.HostPortInfo{
			"127.0.0.1": map[fwk.ProtocolPort]struct{}{
				{Protocol: "TCP", Port: 80}:   {},
				{Protocol: "TCP", Port: 8080}: {},
			},
		},
		ImageStates:  map[string]*fwk.ImageStateSummary{},
		PVCRefCounts: map[string]int{},
		Pods: []fwk.PodInfo{
			&PodInfo{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-1",
						UID:       types.UID("test-1"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("500"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 80,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
					},
				},
				cachedResource: &fwk.PodResource{
					Resource: &Resource{
						MilliCPU: 100,
						Memory:   500,
					},
					Non0CPU: 100,
					Non0Mem: 500,
				},
			},
			&PodInfo{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-2",
						UID:       types.UID("test-2"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("200m"),
										v1.ResourceMemory: resource.MustParse("1Ki"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
					},
				},
				cachedResource: &fwk.PodResource{
					Resource: &Resource{
						MilliCPU: 200,
						Memory:   1024,
					},
					Non0CPU: 200,
					Non0Mem: 1024,
				},
			},
		},
	}

	gen := generation
	ni := NewNodeInfo(pods...)
	if ni.Generation <= gen {
		t.Errorf("Generation is not incremented. previous: %v, current: %v", gen, ni.Generation)
	}
	expected.Generation = ni.Generation
	if diff := cmp.Diff(expected, ni, nodeInfoCmpOpts...); diff != "" {
		t.Errorf("Unexpected NodeInfo (-want, +got):\n%s", diff)
	}
}

func TestNodeInfoClone(t *testing.T) {
	nodeName := "test-node"
	declaredFeatureSet := ndf.NewFeatureMapper([]string{"A", "B", "C"}).MustMapSorted([]string{"A", "C"})

	tests := []struct {
		nodeInfo *NodeInfo
		expected *NodeInfo
	}{
		{
			nodeInfo: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       2,
				UsedPorts: fwk.HostPortInfo{
					"127.0.0.1": map[fwk.ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*fwk.ImageStateSummary{},
				PVCRefCounts: map[string]int{},
				Pods: []fwk.PodInfo{
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 100,
								Memory:   500,
							},
							Non0CPU: 100,
							Non0Mem: 500,
						},
					},
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 200,
								Memory:   1024,
							},
							Non0CPU: 200,
							Non0Mem: 1024,
						},
					},
				},
			},
			expected: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       2,
				UsedPorts: fwk.HostPortInfo{
					"127.0.0.1": map[fwk.ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*fwk.ImageStateSummary{},
				PVCRefCounts: map[string]int{},
				Pods: []fwk.PodInfo{
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 100,
								Memory:   500,
							},
							Non0CPU: 100,
							Non0Mem: 500,
						},
					},
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 200,
								Memory:   1024,
							},
							Non0CPU: 200,
							Non0Mem: 1024,
						},
					},
				},
			},
		},
		{
			nodeInfo: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       3,
				UsedPorts:        fwk.HostPortInfo{},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{},
				DeclaredFeatures: declaredFeatureSet.Clone(),
			},
			expected: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       3,
				UsedPorts:        fwk.HostPortInfo{},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{},
				DeclaredFeatures: declaredFeatureSet,
			},
		},
		{
			nodeInfo: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       3,
				UsedPorts:        fwk.HostPortInfo{},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{},
				DeclaredFeatures: declaredFeatureSet.Clone(),
			},
			expected: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       3,
				UsedPorts:        fwk.HostPortInfo{},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{},
				DeclaredFeatures: declaredFeatureSet,
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			ni := test.nodeInfo.Snapshot()
			// Modify the field to check if the result is a clone of the origin one.
			test.nodeInfo.Generation += 10
			test.nodeInfo.UsedPorts.Remove("127.0.0.1", "TCP", 80)
			if diff := cmp.Diff(test.expected, ni, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Unexpected NodeInfo (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestNodeInfoAddPod(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-1",
				UID:       types.UID("test-1"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("500"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 80,
								Protocol: "TCP",
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("500m"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-2",
				UID:       types.UID("test-2"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU: resource.MustParse("200m"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 8080,
								Protocol: "TCP",
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("500"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-3",
				UID:       types.UID("test-3"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU: resource.MustParse("200m"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 8080,
								Protocol: "TCP",
							},
						},
					},
				},
				InitContainers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("500m"),
								v1.ResourceMemory: resource.MustParse("200Mi"),
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("500"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-2",
							},
						},
					},
				},
			},
		},
	}
	expected := &NodeInfo{
		node: &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node",
			},
		},
		Requested: &Resource{
			MilliCPU:         2300,
			Memory:           209716700, //1500 + 200MB in initContainers
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		NonZeroRequested: &Resource{
			MilliCPU:         2300,
			Memory:           419431900, //200MB(initContainers) + 200MB(default memory value) + 1500 specified in requests/overhead
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		Allocatable: &Resource{},
		Generation:  2,
		UsedPorts: fwk.HostPortInfo{
			"127.0.0.1": map[fwk.ProtocolPort]struct{}{
				{Protocol: "TCP", Port: 80}:   {},
				{Protocol: "TCP", Port: 8080}: {},
			},
		},
		ImageStates:      map[string]*fwk.ImageStateSummary{},
		PVCRefCounts:     map[string]int{"node_info_cache_test/pvc-1": 2, "node_info_cache_test/pvc-2": 1},
		DeclaredFeatures: ndf.DefaultFramework.NewFeatureSet(), // Empty FeatureSet.
		Pods: []fwk.PodInfo{
			&PodInfo{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-1",
						UID:       types.UID("test-1"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("500"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 80,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("500m"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-1",
									},
								},
							},
						},
					},
				},
				cachedResource: &fwk.PodResource{
					Resource: &Resource{
						MilliCPU: 600,
						Memory:   500,
					},
					Non0CPU: 600,
					Non0Mem: 500,
				},
			},
			&PodInfo{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-2",
						UID:       types.UID("test-2"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("200m"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("500"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-1",
									},
								},
							},
						},
					},
				},
				cachedResource: &fwk.PodResource{
					Resource: &Resource{
						MilliCPU: 700,
						Memory:   500,
					},
					Non0CPU: 700,
					Non0Mem: schedutil.DefaultMemoryRequest + 500,
				},
			},
			&PodInfo{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-3",
						UID:       types.UID("test-3"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("200m"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						InitContainers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("500m"),
										v1.ResourceMemory: resource.MustParse("200Mi"),
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("500"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-2",
									},
								},
							},
						},
					},
				},
				cachedResource: &fwk.PodResource{
					Resource: &Resource{
						MilliCPU: 1000,
						Memory:   schedutil.DefaultMemoryRequest + 500,
					},
					Non0CPU: 1000,
					Non0Mem: schedutil.DefaultMemoryRequest + 500,
				},
			},
		},
	}

	ni := fakeNodeInfo()
	gen := ni.Generation
	for _, pod := range pods {
		ni.AddPod(pod)
		if ni.Generation <= gen {
			t.Errorf("Generation is not incremented. Prev: %v, current: %v", gen, ni.Generation)
		}
		gen = ni.Generation
	}

	expected.Generation = ni.Generation
	if diff := cmp.Diff(expected, ni, nodeInfoCmpOpts...); diff != "" {
		t.Errorf("Unexpected NodeInfo (-want, +got):\n%s", diff)
	}
}

func TestNodeInfoRemovePod(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		st.MakePod().UID("test-1").Namespace("node_info_cache_test").Name("test-1").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100m",
				v1.ResourceMemory: "500",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 80,
				Protocol: "TCP",
			}}).Obj()}).
			Volumes([]v1.Volume{{VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "pvc-1"}}}}).
			Obj(),

		st.MakePod().UID("test-2").Namespace("node_info_cache_test").Name("test-2").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "200m",
				v1.ResourceMemory: "1Ki",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 8080,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),
	}

	// add pod Overhead
	for _, pod := range pods {
		pod.Spec.Overhead = v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("500m"),
			v1.ResourceMemory: resource.MustParse("500"),
		}
	}

	tests := []struct {
		pod              *v1.Pod
		errExpected      bool
		expectedNodeInfo *NodeInfo
	}{
		{
			pod:         st.MakePod().UID("non-exist").Namespace("node_info_cache_test").Node(nodeName).Obj(),
			errExpected: true,
			expectedNodeInfo: &NodeInfo{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-node",
					},
				},
				Requested: &Resource{
					MilliCPU:         1300,
					Memory:           2524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				NonZeroRequested: &Resource{
					MilliCPU:         1300,
					Memory:           2524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				Allocatable: &Resource{},
				Generation:  2,
				UsedPorts: fwk.HostPortInfo{
					"127.0.0.1": map[fwk.ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{"node_info_cache_test/pvc-1": 1},
				DeclaredFeatures: ndf.DefaultFramework.NewFeatureSet(), // Empty FeatureSet.
				Pods: []fwk.PodInfo{
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
								Volumes: []v1.Volume{
									{
										VolumeSource: v1.VolumeSource{
											PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
												ClaimName: "pvc-1",
											},
										},
									},
								},
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 600,
								Memory:   1000,
							},
							Non0CPU: 600,
							Non0Mem: 1000,
						},
					},
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 700,
								Memory:   1524,
							},
							Non0CPU: 700,
							Non0Mem: 1524,
						},
					},
				},
			},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "node_info_cache_test",
					Name:      "test-1",
					UID:       types.UID("test-1"),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
							Ports: []v1.ContainerPort{
								{
									HostIP:   "127.0.0.1",
									HostPort: 80,
									Protocol: "TCP",
								},
							},
						},
					},
					NodeName: nodeName,
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("500"),
					},
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "pvc-1",
								},
							},
						},
					},
				},
			},
			errExpected: false,
			expectedNodeInfo: &NodeInfo{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-node",
					},
				},
				Requested: &Resource{
					MilliCPU:         700,
					Memory:           1524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				NonZeroRequested: &Resource{
					MilliCPU:         700,
					Memory:           1524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				Allocatable: &Resource{},
				Generation:  3,
				UsedPorts: fwk.HostPortInfo{
					"127.0.0.1": map[fwk.ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:      map[string]*fwk.ImageStateSummary{},
				PVCRefCounts:     map[string]int{},
				DeclaredFeatures: ndf.DefaultFramework.NewFeatureSet(), // Empty FeatureSet.
				Pods: []fwk.PodInfo{
					&PodInfo{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
						},
						cachedResource: &fwk.PodResource{
							Resource: &Resource{
								MilliCPU: 700,
								Memory:   1524,
							},
							Non0CPU: 700,
							Non0Mem: 1524,
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			ni := fakeNodeInfo(pods...)

			gen := ni.Generation
			err := ni.RemovePod(logger, test.pod)
			if err != nil {
				if test.errExpected {
					expectedErrorMsg := fmt.Errorf("no corresponding pod %s in pods of node %s", test.pod.Name, ni.Node().Name)
					if expectedErrorMsg == err {
						t.Errorf("expected error: %v, got: %v", expectedErrorMsg, err)
					}
				} else {
					t.Errorf("expected no error, got: %v", err)
				}
			} else {
				if ni.Generation <= gen {
					t.Errorf("Generation is not incremented. Prev: %v, current: %v", gen, ni.Generation)
				}
			}

			test.expectedNodeInfo.Generation = ni.Generation
			if diff := cmp.Diff(test.expectedNodeInfo, ni, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Unexpected NodeInfo (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestSetNodeDeclaredFeatures(t *testing.T) {
	ndfFramework, _ := ndftesting.NewMockFramework(t, "FeatureA", "FeatureB")
	ndftesting.SetFrameworkDuringTest(t, ndfFramework)
	tests := []struct {
		name               string
		featureGateEnabled bool
		nodeStatus         v1.NodeStatus
		expectedFeatures   []string
	}{
		{
			name:               "Feature gate disabled",
			featureGateEnabled: false,
			nodeStatus: v1.NodeStatus{
				DeclaredFeatures: []string{"FeatureA", "FeatureB"},
			},
			expectedFeatures: nil,
		},
		{
			name:               "Feature gate enabled, node has features",
			featureGateEnabled: true,
			nodeStatus: v1.NodeStatus{
				DeclaredFeatures: []string{"FeatureA", "FeatureB"},
			},
			expectedFeatures: []string{"FeatureA", "FeatureB"},
		},
		{
			name:               "Feature gate enabled, node has no features",
			featureGateEnabled: true,
			nodeStatus: v1.NodeStatus{
				DeclaredFeatures: []string{},
			},
			expectedFeatures: nil,
		},
		{
			name:               "Feature gate enabled, node has an unknown feature",
			featureGateEnabled: true,
			nodeStatus: v1.NodeStatus{
				DeclaredFeatures: []string{"FeatureA", "OtherFeature"},
			},
			expectedFeatures: []string{"FeatureA"},
		},
		{
			name:               "Feature gate enabled, node status has nil features",
			featureGateEnabled: true,
			nodeStatus: v1.NodeStatus{
				DeclaredFeatures: nil,
			},
			expectedFeatures: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.featureGateEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.36"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, tt.featureGateEnabled)
			}
			ni := NewNodeInfo()
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				Status:     tt.nodeStatus,
			}
			ni.SetNode(node)
			gotFeatures := ni.GetNodeDeclaredFeatures()
			if !tt.featureGateEnabled {
				if !gotFeatures.IsEmpty() {
					got, err := ndfFramework.Unmap(gotFeatures)
					if err != nil {
						t.Fatalf("Failed to unmap features: %v", err)
					}
					t.Errorf("Expected GetNodeDeclaredFeatures() to return nil; got %v", got)
				}
				return
			}
			expected := ndfFramework.MustMapSorted(tt.expectedFeatures)
			if !gotFeatures.Equal(expected) {
				got, err := ndfFramework.Unmap(gotFeatures)
				if err != nil {
					t.Fatalf("Failed to unmap features: %v", err)
				}
				t.Errorf("SetNode() or GetNodeDeclaredFeatures() unexpected result, got: %v, want: %v", got, tt.expectedFeatures)
			}
		})
	}
}

func fakeNodeInfo(pods ...*v1.Pod) *NodeInfo {
	ni := NewNodeInfo(pods...)
	ni.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	})
	return ni
}

func TestFitError_Error(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		numAllNodes   int
		diagnosis     Diagnosis
		wantReasonMsg string
	}{
		{
			name:        "nodes failed Prefilter plugin",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "Node(s) failed PreFilter plugin FalsePreFilter",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					// They're inserted by the framework.
					// We don't include them in the reason message because they'd be just duplicates.
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: Node(s) failed PreFilter plugin FalsePreFilter.",
		},
		{
			name:        "nodes failed Prefilter plugin and the preemption also failed",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "Node(s) failed PreFilter plugin FalsePreFilter",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					// They're inserted by the framework.
					// We don't include them in the reason message because they'd be just duplicates.
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
				// PostFilterMsg will be included.
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: Node(s) failed PreFilter plugin FalsePreFilter. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "nodes failed one Filter plugin with an empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: 3 Node(s) failed Filter plugin FalseFilter-1.",
		},
		{
			name:        "nodes failed one Filter plugin with a non-empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: 3 Node(s) failed Filter plugin FalseFilter-1. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "nodes failed two Filter plugins with an empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-2"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: 1 Node(s) failed Filter plugin FalseFilter-2, 2 Node(s) failed Filter plugin FalseFilter-1.",
		},
		{
			name:        "nodes failed two Filter plugins with a non-empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": fwk.NewStatus(fwk.Unschedulable, "Node(s) failed Filter plugin FalseFilter-2"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: 1 Node(s) failed Filter plugin FalseFilter-2, 2 Node(s) failed Filter plugin FalseFilter-1. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "failed to Permit on node",
			numAllNodes: 1,
			diagnosis: Diagnosis{
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					// There should be only one node here.
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node failed Permit plugin Permit-1"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/1 nodes are available: 1 Node failed Permit plugin Permit-1.",
		},
		{
			name:        "failed to Reserve on node",
			numAllNodes: 1,
			diagnosis: Diagnosis{
				NodeToStatus: NewNodeToStatus(map[string]*fwk.Status{
					// There should be only one node here.
					"node1": fwk.NewStatus(fwk.Unschedulable, "Node failed Reserve plugin Reserve-1"),
				}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/1 nodes are available: 1 Node failed Reserve plugin Reserve-1.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &FitError{
				Pod:         tt.pod,
				NumAllNodes: tt.numAllNodes,
				Diagnosis:   tt.diagnosis,
			}
			if gotReasonMsg := f.Error(); gotReasonMsg != tt.wantReasonMsg {
				t.Errorf("Error() = Got: %v Want: %v", gotReasonMsg, tt.wantReasonMsg)
			}
		})
	}
}

var (
	cpu100m       = resource.MustParse("100m")
	mem200M       = resource.MustParse("200Mi")
	cpu500m       = resource.MustParse("500m")
	mem500M       = resource.MustParse("500Mi")
	cpu700m       = resource.MustParse("700m")
	mem800M       = resource.MustParse("800Mi")
	cpu1000m      = resource.MustParse("1000m")
	cpu1200m      = resource.MustParse("1200m")
	mem1200M      = resource.MustParse("1200Mi")
	restartAlways = v1.ContainerRestartPolicyAlways
)

func TestPodInfoCalculateResources(t *testing.T) {
	testCases := []struct {
		name                                 string
		containers                           []v1.Container
		podResources                         *v1.ResourceRequirements
		podLevelResourcesEnabled             bool
		nodeAllocatableResourcesDRAEnabled   bool
		nodeAllocatableResourceClaimStatuses []v1.NodeAllocatableResourceClaimStatus
		expectedResource                     fwk.PodResource
		initContainers                       []v1.Container
		overhead                             *v1.ResourceList
	}{
		{
			name:       "requestless container",
			containers: []v1.Container{{}},
			expectedResource: fwk.PodResource{
				Resource: &Resource{},
				Non0CPU:  schedutil.DefaultMilliCPURequest,
				Non0Mem:  schedutil.DefaultMemoryRequest,
			},
		},
		{
			name: "1X container with requests",
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name: "2X container with requests",
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu700m,
							v1.ResourceMemory: mem800M,
						},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu700m.MilliValue(),
					Memory:   mem500M.Value() + mem800M.Value(),
				},
				Non0CPU: cpu500m.MilliValue() + cpu700m.MilliValue(),
				Non0Mem: mem500M.Value() + mem800M.Value(),
			},
		},
		{
			name:                     "1X container and 1X init container with pod-level requests",
			podLevelResourcesEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    cpu1200m,
					v1.ResourceMemory: mem1200M,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu1200m.MilliValue(),
					Memory:   mem1200M.Value(),
				},
				Non0CPU: cpu1200m.MilliValue(),
				Non0Mem: mem1200M.Value(),
			},
		},
		{
			name:                     "1X container and 1X sidecar container with pod-level requests",
			podLevelResourcesEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
					RestartPolicy: &restartAlways,
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    cpu1200m,
					v1.ResourceMemory: mem1200M,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu1200m.MilliValue(),
					Memory:   mem1200M.Value(),
				},
				Non0CPU: cpu1200m.MilliValue(),
				Non0Mem: mem1200M.Value(),
			},
		},
		{
			name:                     "1X container with pod-level memory requests",
			podLevelResourcesEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: mem1200M,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					Memory: mem1200M.Value(),
				},
				Non0CPU: schedutil.DefaultMilliCPURequest,
				Non0Mem: mem1200M.Value(),
			},
		},
		{
			name:                     "1X container with pod-level cpu requests",
			podLevelResourcesEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: cpu500m,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: schedutil.DefaultMemoryRequest,
			},
		},
		{
			name:                     "1X container unsupported resources and pod-level supported resources",
			podLevelResourcesEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceEphemeralStorage: mem500M,
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceEphemeralStorage: mem800M,
						},
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: cpu500m,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU:         cpu500m.MilliValue(),
					EphemeralStorage: mem800M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: schedutil.DefaultMemoryRequest,
			},
		},
		{
			name:                               "DRA gate disabled, with node allocatable resource claim",
			nodeAllocatableResourcesDRAEnabled: false,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						// We do not set Pod.Spec.ResourceClaims in this test.
						// We assume the name maps to Pod.Spec.ResourceClaims[].ResourceClaimName.
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:                               "container with DRA node allocatable resource claim",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu100m.MilliValue(),
					Memory:   mem500M.Value() + mem200M.Value(),
				},
				Non0CPU: cpu500m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "Multiple DRA node allocatable resource claims",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
							{
								Name: "node-allocatable-claim-2",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim-1",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
				{
					ResourceClaimName: "node-allocatable-claim-2",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(),
					Memory:   mem500M.Value() + mem200M.Value(),
				},
				Non0CPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "Single DRA claim with multiple resources",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim-1",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu1000m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu1000m.MilliValue(),
					Memory:   mem500M.Value() + mem200M.Value(),
				},
				Non0CPU: cpu500m.MilliValue() + cpu1000m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "DRA node allocatable Resources with Init Container",
			nodeAllocatableResourcesDRAEnabled: true,
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu1200m, // Higher than app container
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem1200M, // Higher than init container
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim-1",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu1200m.MilliValue() + cpu100m.MilliValue(), // max(cpu500m, cpu1200m) + draCpu
					Memory:   mem1200M.Value() + mem200M.Value(),           // max(mem500M, mem1200M) + draMem
				},
				Non0CPU: cpu1200m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem1200M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "DRA node allocatable Resources with Pod Level Resources",
			nodeAllocatableResourcesDRAEnabled: true,
			podLevelResourcesEnabled:           true,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
						},
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    cpu700m,
					v1.ResourceMemory: mem800M,
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim-1",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(), // pod level requests determines the overall footprint
					Memory:   mem800M.Value(),      // pod level requests determines the overall footprint
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem800M.Value(),
			},
		},
		{
			name:                               "DRA node allocatable Resources with Pod Overhead",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim-1",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim-1",
					Mapping: []v1.NodeAllocatableMappedResources{
						{Name: v1.ResourceCPU, Quantity: new(cpu100m)},
						{Name: v1.ResourceMemory, Quantity: new(mem200M)},
					},
				},
			},
			// Pod Overhead
			overhead: &v1.ResourceList{
				v1.ResourceCPU:    cpu100m,
				v1.ResourceMemory: mem200M,
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(), // container + dra + overhead
					Memory:   mem500M.Value() + mem200M.Value() + mem200M.Value(),                // container + dra + overhead
				},
				Non0CPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "DRA gate enabled, with node allocatable resource claim specifying Overhead",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim",
					Containers:        []string{"c1"},
					Overhead: []v1.NodeAllocatableOverheadResources{
						{
							Name:         v1.ResourceCPU,
							PerPod:       &cpu100m,
							PerContainer: &cpu100m,
						},
						{
							Name:         v1.ResourceMemory,
							PerPod:       &mem200M,
							PerContainer: &mem200M,
						},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(), // container (500m) + DRA flat pod overhead (100m) + DRA container overhead (100m) = 700m
					Memory:   mem500M.Value() + mem200M.Value() + mem200M.Value(),                // container (500M) + DRA flat pod overhead (200M) + DRA container overhead (200M) = 900M
				},
				Non0CPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value() + mem200M.Value(),
			},
		},
		{
			name:                               "DRA gate enabled, with node allocatable resource claim specifying both Mapping and Overhead",
			nodeAllocatableResourcesDRAEnabled: true,
			containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu500m,
							v1.ResourceMemory: mem500M,
						},
						Claims: []v1.ResourceClaim{
							{
								Name: "node-allocatable-claim",
							},
						},
					},
				},
			},
			nodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "node-allocatable-claim",
					Containers:        []string{"c1"},
					Mapping: []v1.NodeAllocatableMappedResources{
						{
							Name:     v1.ResourceCPU,
							Quantity: new(cpu100m),
						},
						{
							Name:     v1.ResourceMemory,
							Quantity: new(mem200M),
						},
					},
					Overhead: []v1.NodeAllocatableOverheadResources{
						{
							Name:         v1.ResourceCPU,
							PerPod:       &cpu100m,
							PerContainer: &cpu100m,
						},
						{
							Name:         v1.ResourceMemory,
							PerPod:       &mem200M,
							PerContainer: &mem200M,
						},
					},
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(), // spec (500) + mapping (100) + perPod (100) + perContainer (100) = 800m
					Memory:   mem500M.Value() + mem200M.Value() + mem200M.Value() + mem200M.Value(),                     // spec (500) + mapping (200) + perPod (200) + perContainer (200) = 1100M
				},
				Non0CPU: cpu500m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue() + cpu100m.MilliValue(),
				Non0Mem: mem500M.Value() + mem200M.Value() + mem200M.Value() + mem200M.Value(),
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.PodLevelResources:           tc.podLevelResourcesEnabled,
				features.DRANodeAllocatableResources: tc.nodeAllocatableResourcesDRAEnabled,
			})
			podSpec := v1.PodSpec{
				Resources:      tc.podResources,
				Containers:     tc.containers,
				InitContainers: tc.initContainers,
			}
			if tc.overhead != nil {
				podSpec.Overhead = *tc.overhead
			}
			podInfo := PodInfo{
				Pod: &v1.Pod{
					Spec: podSpec,
					Status: v1.PodStatus{
						NodeAllocatableResourceClaimStatuses: tc.nodeAllocatableResourceClaimStatuses,
					},
				},
			}
			res := podInfo.CalculateResource()
			if diff := cmp.Diff(tc.expectedResource, res, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Unexpected resource (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestCalculatePodResourcesWithResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)

	testpod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "pod_resize_test",
			Name:      "testpod",
			UID:       types.UID("testpod"),
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}

	restartAlways := v1.ContainerRestartPolicyAlways

	preparePodInfo := func(pod v1.Pod,
		podRequests, podStatusResources,
		requests, statusResources,
		initRequests, initStatusResources,
		sidecarRequests, sidecarStatusResources *v1.ResourceList,
		resizeStatus []*v1.PodCondition) PodInfo {

		if podRequests != nil {
			pod.Spec.Resources = &v1.ResourceRequirements{
				Requests: *podRequests,
			}
		}

		if podStatusResources != nil {
			pod.Status.Resources = &v1.ResourceRequirements{
				Requests: *podStatusResources,
			}
		}

		if requests != nil {
			pod.Spec.Containers = append(pod.Spec.Containers,
				v1.Container{
					Name:      "c1",
					Resources: v1.ResourceRequirements{Requests: *requests},
				})
		}
		if statusResources != nil {
			pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses,
				v1.ContainerStatus{
					Name: "c1",
					Resources: &v1.ResourceRequirements{
						Requests: *statusResources,
					},
				})
		}

		if initRequests != nil {
			pod.Spec.InitContainers = append(pod.Spec.InitContainers,
				v1.Container{
					Name:      "i1",
					Resources: v1.ResourceRequirements{Requests: *initRequests},
				},
			)
		}
		if initStatusResources != nil {
			pod.Status.InitContainerStatuses = append(pod.Status.InitContainerStatuses,
				v1.ContainerStatus{
					Name: "i1",
					Resources: &v1.ResourceRequirements{
						Requests: *initStatusResources,
					},
				})
		}

		if sidecarRequests != nil {
			pod.Spec.InitContainers = append(pod.Spec.InitContainers,
				v1.Container{
					Name:          "s1",
					Resources:     v1.ResourceRequirements{Requests: *sidecarRequests},
					RestartPolicy: &restartAlways,
				},
			)
		}
		if sidecarStatusResources != nil {
			pod.Status.InitContainerStatuses = append(pod.Status.InitContainerStatuses,
				v1.ContainerStatus{
					Name: "s1",
					Resources: &v1.ResourceRequirements{
						Requests: *sidecarStatusResources,
					},
				})
		}

		for _, c := range resizeStatus {
			pod.Status.Conditions = append(pod.Status.Conditions, *c)
		}

		return PodInfo{Pod: &pod}
	}

	tests := []struct {
		name                    string
		podLevelRequests        *v1.ResourceList
		requests                v1.ResourceList
		statusResources         v1.ResourceList
		podLevelStatusResources *v1.ResourceList
		initRequests            *v1.ResourceList
		initStatusResources     *v1.ResourceList
		resizeStatus            []*v1.PodCondition
		sidecarRequests         *v1.ResourceList
		sidecarStatusResources  *v1.ResourceList
		expectedResource        fwk.PodResource
	}{
		{
			name:            "Pod with no pending resize",
			requests:        v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:                    "Pod with pod-level resources no pending resize",
			podLevelRequests:        &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem500M},
			requests:                v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			podLevelStatusResources: &v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources:         v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem500M},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:            "Pod with resize in progress",
			requests:        v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: v1.ConditionTrue,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:                    "Pod with pod-level resources and resize in progress",
			podLevelRequests:        &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem500M},
			requests:                v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			podLevelStatusResources: &v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources:         v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem500M},
			resizeStatus: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: v1.ConditionTrue,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:            "Pod with deferred resize",
			requests:        v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			statusResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus: []*v1.PodCondition{
				{
					Type:   v1.PodResizePending,
					Status: v1.ConditionTrue,
					Reason: v1.PodReasonDeferred,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(),
					Memory:   mem800M.Value(),
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem800M.Value(),
			},
		},
		{
			name:                    "Pod with pod-level resources and with deferred resize",
			podLevelRequests:        &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			requests:                v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			podLevelStatusResources: &v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources:         v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus: []*v1.PodCondition{
				{
					Type:   v1.PodResizePending,
					Status: v1.ConditionTrue,
					Reason: v1.PodReasonDeferred,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(),
					Memory:   mem800M.Value(),
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem800M.Value(),
			},
		},
		{
			name:            "Pod with infeasible resize",
			requests:        v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			statusResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus: []*v1.PodCondition{
				{
					Type:   v1.PodResizePending,
					Status: v1.ConditionTrue,
					Reason: v1.PodReasonInfeasible,
				},
			},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue(),
					Memory:   mem500M.Value(),
				},
				Non0CPU: cpu500m.MilliValue(),
				Non0Mem: mem500M.Value(),
			},
		},
		{
			name:                "Pod with init container and no pending resize",
			requests:            v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources:     v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			initRequests:        &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			initStatusResources: &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu700m.MilliValue(),
					Memory:   mem800M.Value(),
				},
				Non0CPU: cpu700m.MilliValue(),
				Non0Mem: mem800M.Value(),
			},
		},
		{
			name:                   "Pod with sider container and no pending resize",
			requests:               v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			statusResources:        v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			initRequests:           &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			initStatusResources:    &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			sidecarRequests:        &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			sidecarStatusResources: &v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			expectedResource: fwk.PodResource{
				Resource: &Resource{
					MilliCPU: cpu500m.MilliValue() + cpu700m.MilliValue(),
					Memory:   mem500M.Value() + mem800M.Value(),
				},
				Non0CPU: cpu500m.MilliValue() + cpu700m.MilliValue(),
				Non0Mem: mem500M.Value() + mem800M.Value(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podInfo := preparePodInfo(*testpod.DeepCopy(),
				tt.podLevelRequests, tt.podLevelStatusResources,
				&tt.requests, &tt.statusResources,
				tt.initRequests, tt.initStatusResources,
				tt.sidecarRequests, tt.sidecarStatusResources,
				tt.resizeStatus)

			res := podInfo.CalculateResource()
			if diff := cmp.Diff(tt.expectedResource, res, nodeInfoCmpOpts...); diff != "" {
				t.Errorf("Unexpected podResource (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestClusterEventMatching(t *testing.T) {
	testCases := []struct {
		name        string
		event       fwk.ClusterEvent
		comingEvent fwk.ClusterEvent
		wantResult  bool
	}{
		{
			name:        "wildcard event matches with all kinds of coming events",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.All},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' matching with coming events carries matching actionType",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel | fwk.UpdateNodeTaint},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' also matches event with 'UnscheduledPod' resource and matching actionType",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel | fwk.UpdateNodeTaint},
			comingEvent: fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = '*' matching with coming events carries same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.UpdateNodeLabel},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = '*' matching with coming events carries different actionType",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.UpdateNodeLabel},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeAllocatable},
			wantResult:  false,
		},
		{
			name:        "event matching with coming events carries '*' resources",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			comingEvent: fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.UpdateNodeLabel},
			wantResult:  false,
		},
		{
			name:        "event with resource = '*' matching with coming events carrying a too broad actionType",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.UpdateNodeLabel},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Update},
			wantResult:  false,
		},
		{
			name:        "event with resource = '*' matching with coming events carrying a more specific actionType",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.Update},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' also matches event with 'UnscheduledPod' resource and matching actionType",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel | fwk.UpdateNodeTaint},
			comingEvent: fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' also matches event with AssignedPod resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeLabel},
			comingEvent: fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' also matches event with 'TargetPod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdateNodeTaint},
			comingEvent: fwk.ClusterEvent{Resource: fwk.TargetPod, ActionType: fwk.UpdateNodeTaint},
			wantResult:  true,
		},
		{
			name:        "event with resource 'AssignedPod' does not match event with 'UnscheduledPod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Add},
			comingEvent: fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.Add},
			wantResult:  false,
		},
		{
			name:        "event with resource 'AssignedPod' does not match event with 'TargetPod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Update},
			comingEvent: fwk.ClusterEvent{Resource: fwk.TargetPod, ActionType: fwk.Update},
			wantResult:  false,
		},
		{
			name:        "event with resource 'TargetPod' does not match event with 'AssignedPod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.TargetPod, ActionType: fwk.Update},
			comingEvent: fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Update},
			wantResult:  false,
		},
		{
			name:        "event with resource 'AssignedPod' does not match with broad 'Pod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Add},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add},
			wantResult:  false,
		},
		{
			name:        "event with resource 'AssignedPod' does not match with 'WildCard' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Add},
			comingEvent: fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.Add},
			wantResult:  false,
		},
		{
			name:        "event with resource 'UnscheduledPod' does not match with broad 'Pod' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.Add},
			comingEvent: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add},
			wantResult:  false,
		},
		{
			name:        "event with resource 'UnscheduledPod' does not match with 'WildCard' resource and same actionType",
			event:       fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.Add},
			comingEvent: fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.Add},
			wantResult:  false,
		},
		{
			name:        "WildCard matches with a pod sub-type resource 'TargetPod' and any actionType",
			event:       fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.All},
			comingEvent: fwk.ClusterEvent{Resource: fwk.TargetPod, ActionType: fwk.Update},
			wantResult:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchClusterEvents(tc.event, tc.comingEvent)
			if got != tc.wantResult {
				t.Fatalf("unexpected result")
			}
		})
	}
}

func TestUnrollPodEvent(t *testing.T) {
	tests := []struct {
		name      string
		event     fwk.ClusterEvent
		wantTypes []fwk.EventResource
	}{
		{
			name:      "Pod/Add unrolls into pod sub-types with add action type",
			event:     fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add},
			wantTypes: []fwk.EventResource{fwk.AssignedPod, fwk.UnscheduledPod, fwk.TargetPod},
		},
		{
			name:      "Pod/Delete unrolls into pod sub-types with delete action type",
			event:     fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete},
			wantTypes: []fwk.EventResource{fwk.AssignedPod, fwk.UnscheduledPod, fwk.TargetPod},
		},
		{
			name:      "Pod/Update unrolls into pod sub-types with update action type",
			event:     fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Update},
			wantTypes: []fwk.EventResource{fwk.AssignedPod, fwk.UnscheduledPod, fwk.TargetPod},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UnrollPodEvent(tt.event)
			for i, res := range tt.wantTypes {
				if got[i].Resource != res {
					t.Errorf("event[%d]: expected resource %q, got %q", i, res, got[i].Resource)
				}
				if got[i].ActionType != tt.event.ActionType {
					t.Errorf("event[%d]: expected ActionType %v, got %v", i, tt.event.ActionType, got[i].ActionType)
				}
			}
		})
	}
}

func TestNodeInfoKMetadata(t *testing.T) {
	tCtx := ktesting.Init(t, initoption.BufferLogs(true))
	logger := tCtx.Logger()
	logger.Info("Some NodeInfo slice", "nodes", klog.KObjSlice([]*NodeInfo{nil, {}, {node: &v1.Node{}}, {node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "worker"}}}}))

	output := logger.GetSink().(ktesting.Underlier).GetBuffer().String()

	// The initial nil entry gets turned into empty ObjectRef by klog,
	// which becomes an empty string during output formatting.
	if !strings.Contains(output, `Some NodeInfo slice nodes=["","<no node>","","worker"]`) {
		tCtx.Fatalf("unexpected output:\n%s", output)
	}
}

func TestUpdateUsedPorts_PodAdd(t *testing.T) {
	testCases := []struct {
		ports fwk.HostPortInfo
		pod   *v1.Pod
		want  fwk.HostPortInfo
	}{
		{
			ports: fwk.HostPortInfo{},
			pod:   nil,
			want:  fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: nil,
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
					fwk.ProtocolPort{Protocol: "TCP", Port: 8002}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{},
			pod: st.MakePod().
				InitContainerPort(false /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(false /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
					fwk.ProtocolPort{Protocol: "TCP", Port: 8002}: struct{}{},
				},
			},
		},
	}
	for _, tc := range testCases {
		ni := NodeInfo{UsedPorts: tc.ports}
		ni.updateUsedPorts(tc.pod, true)
		if diff := cmp.Diff(tc.want, ni.UsedPorts); diff != "" {
			t.Errorf("updateUsedPorts() unexpected diff (-want, +got):\n%s", diff)
		}
	}
}

func TestUpdateUsedPorts_PodRemove(t *testing.T) {
	testCases := []struct {
		ports fwk.HostPortInfo
		pod   *v1.Pod
		want  fwk.HostPortInfo
	}{
		{
			ports: fwk.HostPortInfo{},
			pod:   nil,
			want:  fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: nil,
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
					fwk.ProtocolPort{Protocol: "TCP", Port: 8002}: struct{}{},
				},
			},
			pod: st.MakePod().
				ContainerPort([]v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(false /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
		{
			ports: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
					fwk.ProtocolPort{Protocol: "TCP", Port: 8002}: struct{}{},
				},
			},
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8002,
						HostPort:      8002,
						Protocol:      v1.ProtocolTCP,
					}}).
				Obj(),
			want: fwk.HostPortInfo{
				"0.0.0.0": {
					fwk.ProtocolPort{Protocol: "TCP", Port: 8001}: struct{}{},
				},
			},
		},
	}
	for _, tc := range testCases {
		ni := NodeInfo{UsedPorts: tc.ports}
		ni.updateUsedPorts(tc.pod, false)
		if diff := cmp.Diff(tc.want, ni.UsedPorts); diff != "" {
			t.Errorf("updateUsedPorts() unexpected diff (-want, +got):\n%s", diff)
		}
	}
}

func TestQueuedPodInfo_UpdateInvalidatesSignature(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Label("version", "1").Obj()
	pod2 := pod1.DeepCopy()
	pod2.Labels["version"] = "2"

	podInfo, _ := NewPodInfo(pod1)
	queuedPodInfo := &QueuedPodInfo{
		PodInfo:      podInfo,
		PodSignature: fwk.PodSignature("sig-1"),
	}

	_, err := queuedPodInfo.Update(pod2)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	if queuedPodInfo.PodSignature != nil {
		t.Errorf("Expected signature to be nil after Update, got '%s'", string(queuedPodInfo.PodSignature))
	}
}

func TestPodInfo_Update(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").UID("uid1").Obj()
	pod2 := pod1.DeepCopy()
	pod2.Labels = map[string]string{"foo": "bar"}
	pod3 := st.MakePod().Name("pod2").UID("uid2").Obj()

	tests := []struct {
		name        string
		podInfo     *PodInfo
		pod         *v1.Pod
		expectedErr string
		verify      func(t *testing.T, pi *PodInfo)
	}{
		{
			name:    "successful update (same UID)",
			podInfo: func() *PodInfo { pi, _ := NewPodInfo(pod1); return pi }(),
			pod:     pod2,
			verify: func(t *testing.T, pi *PodInfo) {
				if pi.Pod.Labels["foo"] != "bar" {
					t.Errorf("Expected updated labels, got %v", pi.Pod.Labels)
				}
			},
		},
		{
			name: "successful update (same UID) - clears cachedResource",
			podInfo: func() *PodInfo {
				pi, _ := NewPodInfo(pod1)
				pi.cachedResource = &fwk.PodResource{}
				return pi
			}(),
			pod: pod2,
			verify: func(t *testing.T, pi *PodInfo) {
				if pi.cachedResource != nil {
					t.Errorf("Expected cachedResource to be nil after Update, got %v", pi.cachedResource)
				}
			},
		},
		{
			name:        "failed update (different UID)",
			podInfo:     func() *PodInfo { pi, _ := NewPodInfo(pod1); return pi }(),
			pod:         pod3,
			expectedErr: "pod UID mismatch",
		},
		{
			name:        "failed update (nil pod)",
			podInfo:     func() *PodInfo { pi, _ := NewPodInfo(pod1); return pi }(),
			pod:         nil,
			expectedErr: "cannot update with nil pod",
		},
		{
			name:        "failed update (nil PodInfo.Pod)",
			podInfo:     &PodInfo{},
			pod:         pod1,
			expectedErr: "cannot update PodInfo - its Pod is nil",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.podInfo.Update(tc.pod)
			if tc.expectedErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("Expected error containing '%s', got %v", tc.expectedErr, err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got %v", err)
				}
				if tc.verify != nil {
					tc.verify(t, tc.podInfo)
				}
			}
		})
	}
}

func TestNewPodInfo(t *testing.T) {
	for _, interPodAffinityHostnameFastPathEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("InterPodAffinityHostnameFastPath=%v", interPodAffinityHostnameFastPathEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InterPodAffinityHostnameFastPath, interPodAffinityHostnameFastPathEnabled)
			affinity := &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"foo": "bar"},
							},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			}

			tests := []struct {
				name        string
				pod         *v1.Pod
				expectedErr string
				verify      func(t *testing.T, pi *PodInfo)
			}{
				{
					name:        "nil pod",
					pod:         nil,
					expectedErr: "pod cannot be nil",
				},
				{
					name: "pod without affinity",
					pod:  st.MakePod().Name("pod1").Obj(),
					verify: func(t *testing.T, pi *PodInfo) {
						if pi.Pod.Name != "pod1" {
							t.Errorf("Expected pod name 'pod1', got %v", pi.Pod.Name)
						}
					},
				},
				{
					name: "pod with affinity",
					pod: func() *v1.Pod {
						p := st.MakePod().Name("pod2").Obj()
						p.Spec.Affinity = affinity
						return p
					}(),
					verify: func(t *testing.T, pi *PodInfo) {
						if len(pi.RequiredAffinityTerms) != 1 {
							t.Errorf("Expected 1 required affinity term, got %v", len(pi.RequiredAffinityTerms))
						}
					},
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					pi, err := NewPodInfo(tc.pod)
					if tc.expectedErr != "" {
						if err == nil || !strings.Contains(err.Error(), tc.expectedErr) {
							t.Errorf("Expected error containing '%s', got %v", tc.expectedErr, err)
						}
					} else {
						if err != nil {
							t.Errorf("Expected no error, got %v", err)
						}
						if tc.verify != nil {
							tc.verify(t, pi)
						}
					}
				})
			}
		})
	}
}

func TestNodeInfo_PodsWithRequiredNonHostScopedAntiAffinity(t *testing.T) {
	podNonHost1 := st.MakePod().Name("pod-nh-1").UID("uid-nh-1").PodAntiAffinityExists("label", "zone", st.PodAntiAffinityWithRequiredReq).Obj()
	podNonHost2 := st.MakePod().Name("pod-nh-2").UID("uid-nh-2").PodAntiAffinityExists("label", "zone", st.PodAntiAffinityWithRequiredReq).Obj()
	podHost1 := st.MakePod().Name("pod-h-1").UID("uid-h-1").PodAntiAffinityExists("label", v1.LabelHostname, st.PodAntiAffinityWithRequiredReq).Obj()
	podNoAA := st.MakePod().Name("pod-no-aa").UID("uid-no-aa").Obj()

	tests := []struct {
		name                                string
		podsToAdd                           []*v1.Pod // Pods to add consecutively
		expectedPodNamesWithFastPathEnabled sets.Set[string]
	}{
		{
			name:                                "2 pods with non-host-scoped AA",
			podsToAdd:                           []*v1.Pod{podNonHost1, podNonHost2},
			expectedPodNamesWithFastPathEnabled: sets.New("pod-nh-1", "pod-nh-2"),
		},
		{
			name:                                "2 pods with non-host-scoped AA + 1 pod with host-scoped AA",
			podsToAdd:                           []*v1.Pod{podNonHost1, podNonHost2, podHost1},
			expectedPodNamesWithFastPathEnabled: sets.New("pod-nh-1", "pod-nh-2"),
		},
		{
			name:                                "1 pod with host-scoped AA",
			podsToAdd:                           []*v1.Pod{podHost1},
			expectedPodNamesWithFastPathEnabled: sets.New[string](),
		},
		{
			name:                                "1 pod with no AA at all",
			podsToAdd:                           []*v1.Pod{podNoAA},
			expectedPodNamesWithFastPathEnabled: sets.New[string](),
		},
	}

	for _, tt := range tests {
		for _, interPodAffinityHostnameFastPathEnabled := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s (InterPodAffinityHostnameFastPath=%v)", tt.name, interPodAffinityHostnameFastPathEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InterPodAffinityHostnameFastPath, interPodAffinityHostnameFastPathEnabled)

				ni := NewNodeInfo()
				for _, pod := range tt.podsToAdd {
					ni.AddPod(pod)
				}

				expected := sets.New[string]() // always empty if FastPath is disabled
				if interPodAffinityHostnameFastPathEnabled {
					expected = tt.expectedPodNamesWithFastPathEnabled
				}

				// Verify AddPod correctly populates the list
				actual := sets.New[string]()
				for _, p := range ni.PodsWithRequiredNonHostScopedAntiAffinity {
					actual.Insert(p.GetPod().Name)
				}
				if !expected.Equal(actual) {
					t.Errorf("AddPod expected %v, got %v", expected, actual)
				}

				// Verify SnapshotConcrete explicitly copies the slice
				clone := ni.SnapshotConcrete()
				cloneActual := sets.New[string]()
				for _, p := range clone.PodsWithRequiredNonHostScopedAntiAffinity {
					cloneActual.Insert(p.GetPod().Name)
				}
				if !expected.Equal(cloneActual) {
					t.Errorf("SnapshotConcrete expected %v, got %v", expected, cloneActual)
				}

				// Modify the cloned NodeInfo's slice to ensure SnapshotConcrete deep-copied the slice
				if len(clone.PodsWithRequiredNonHostScopedAntiAffinity) > 0 {
					originalFirst := clone.PodsWithRequiredNonHostScopedAntiAffinity[0]
					dummyPod, _ := NewPodInfo(st.MakePod().Name("dummy").UID("dummy").Obj())
					clone.PodsWithRequiredNonHostScopedAntiAffinity[0] = dummyPod
					if ni.PodsWithRequiredNonHostScopedAntiAffinity[0] == dummyPod {
						t.Errorf("SnapshotConcrete did not safely copy PodsWithRequiredNonHostScopedAntiAffinity slice, mutation affected original")
					}
					// Restore it just in case
					clone.PodsWithRequiredNonHostScopedAntiAffinity[0] = originalFirst
				}

				// Verify RemovePod
				logger, _ := ktesting.NewTestContext(t)
				for _, pod := range tt.podsToAdd {
					err := ni.RemovePod(logger, pod)
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
				}

				if len(ni.PodsWithRequiredNonHostScopedAntiAffinity) != 0 {
					var remainingPods []string
					for _, p := range ni.PodsWithRequiredNonHostScopedAntiAffinity {
						remainingPods = append(remainingPods, p.GetPod().Name)
					}
					t.Errorf("expected empty list after RemovePod, got: %v", remainingPods)
				}
			})
		}
	}
}

func TestUnrollWildCardResource_WithGenericWorkload(t *testing.T) {
	tests := []struct {
		name                  string
		enableGenericWorkload bool
		wantHasPodGroup       bool
	}{
		{
			name:                  "Events should have PodGroup when GenericWorkload is enabled",
			enableGenericWorkload: true,
			wantHasPodGroup:       true,
		},
		{
			name:                  "Events should not have PodGroup when GenericWorkload is disabled",
			enableGenericWorkload: false,
			wantHasPodGroup:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, tt.enableGenericWorkload)
			events := UnrollWildCardResource()
			hasPodGroup := false
			for _, e := range events {
				if e.Event.Resource == fwk.PodGroup {
					hasPodGroup = true
					break
				}
			}
			if hasPodGroup != tt.wantHasPodGroup {
				t.Errorf("Unexpected events returned from UnrollWildCardResource(): hasPodGroup = %v, want %v", hasPodGroup, tt.wantHasPodGroup)
			}
		})
	}
}

func TestPodGroupInfoGetChildrenSorting(t *testing.T) {
	now := time.Now()
	pgInfo := func(name, namespace string, creationTime time.Time) *PodGroupInfo {
		return &PodGroupInfo{
			GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:              name,
					Namespace:         namespace,
					CreationTimestamp: metav1.NewTime(creationTime),
				},
			}),
		}
	}

	pgInfo1 := pgInfo("pg1", "default", now)
	pgInfo2 := pgInfo("pg2", "default", now.Add(time.Minute))
	pgInfo3 := pgInfo("pg3", "default", now.Add(-time.Minute))
	// pgInfo4 has same timestamp as pgInfo1 but is listed as the last one.
	// We verify sorting stability by checking that pg1 is still before pg4.
	pgInfo4 := pgInfo("pg4", "default", now)

	pgi := &PodGroupInfo{
		GenericPodGroup: fwk.NewGenericCompositePodGroup(&schedulingv1alpha3.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "parent-cpg",
				Namespace: "default",
			},
		}),
		Children: []*PodGroupInfo{pgInfo1, pgInfo2, pgInfo3, pgInfo4},
	}

	expectedOrder := []*PodGroupInfo{pgInfo3, pgInfo1, pgInfo4, pgInfo2}
	gotOrder := pgi.GetChildGroups()

	if diff := cmp.Diff(expectedOrder, gotOrder); diff != "" {
		t.Errorf("GetChildGroups() returned diff (-want +got):\n%s", diff)
	}
}

func TestQueuedPodGroupInfo_AddCompositePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	cpgChild := st.MakeCompositePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	cpgNested := st.MakeCompositePodGroup().Name("cpg-nested").Namespace("ns1").ParentCompositePodGroup("cpg-child").Obj()
	cpgNotFoundParent := st.MakeCompositePodGroup().Name("cpg-orphan").Namespace("ns1").ParentCompositePodGroup("non-existent").Obj()

	tests := []struct {
		name    string
		qpgi    *QueuedPodGroupInfo
		subtree *PodGroupInfo
		verify  func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Add child CPG to root",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(cpgChild),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].GetName() != "cpg-child" {
					t.Errorf("Child CPG not added to root correctly")
				}
			},
		},
		{
			name: "Add CPG with non-existent parent",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(cpgNotFoundParent),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("CPG with non-existent parent should not be added")
				}
			},
		},
		{
			name: "Add standalone CPG (no parent set)",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(st.MakeCompositePodGroup().Name("standalone-cpg").Namespace("ns1").Obj()),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Standalone CPG should not be added to another root")
				}
			},
		},
		{
			name: "Add deeply nested CPG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(cpgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(cpgNested),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children[0].Children) != 1 || qpgi.PodGroupInfo.Children[0].Children[0].GetName() != "cpg-nested" {
					t.Errorf("Deeply nested CPG not added correctly")
				}
			},
		},
		{
			name: "Add CPG subtree with nested CPGs",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(cpgChild,
				newCompositePodGroupInfoForTest(cpgNested,
					newPodGroupInfoForTest(st.MakePodGroup().Name("pg-nested-leaf").Namespace("ns1").ParentCompositePodGroup("cpg-nested").Obj()),
				),
			),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].GetName() != "cpg-child" {
					t.Errorf("Child CPG not added correctly")
				}
				if len(qpgi.PodGroupInfo.Children[0].Children) != 1 || qpgi.PodGroupInfo.Children[0].Children[0].GetName() != "cpg-nested" {
					t.Errorf("Nested CPG not added correctly as part of subtree")
				}
				if len(qpgi.PodGroupInfo.Children[0].Children[0].Children) != 1 || qpgi.PodGroupInfo.Children[0].Children[0].Children[0].GetName() != "pg-nested-leaf" {
					t.Errorf("Leaf PodGroup not added correctly as part of subtree")
				}
			},
		},
		{
			name: "Add CPG while having a sibling PG with the same name",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			subtree: newCompositePodGroupInfoForTest(st.MakeCompositePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 2 {
					t.Fatalf("Expected 2 children under root CPG, got %d", len(qpgi.PodGroupInfo.Children))
				}
				if qpgi.PodGroupInfo.Children[0].GetType() != fwk.PodGroupKeyType || qpgi.PodGroupInfo.Children[0].GetName() != "shared-name" {
					t.Errorf("First child should be PG shared-name")
				}
				if qpgi.PodGroupInfo.Children[1].GetType() != fwk.CompositePodGroupKeyType || qpgi.PodGroupInfo.Children[1].GetName() != "shared-name" {
					t.Errorf("Second child should be CPG shared-name")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.qpgi.AddSubtree(tt.subtree)
			tt.verify(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_UpdateCompositePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	cpgChild := st.MakeCompositePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	cpgChildUpdated := st.MakeCompositePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	cpgChildUpdated.Annotations = map[string]string{"updated": "true"}

	tests := []struct {
		name      string
		qpgi      *QueuedPodGroupInfo
		updateCPG *schedulingv1alpha3.CompositePodGroup
		verify    func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Update existing child CPG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(cpgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updateCPG: cpgChildUpdated,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].CompositePodGroup.Annotations["updated"] != "true" {
					t.Errorf("Child CPG not updated correctly")
				}
			},
		},
		{
			name: "Update non-existent CPG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updateCPG: st.MakeCompositePodGroup().Name("non-existent").Namespace("ns1").Obj(),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				// No panic or errors expected, should just be a no-op
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Non-existent CPG update should not alter hierarchy")
				}
			},
		},
		{
			name: "Update CPG while having a sibling PG with the same name",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(st.MakePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").MinCount(2).Obj()),
					newCompositePodGroupInfoForTest(cpgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updateCPG: cpgChildUpdated,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 2 {
					t.Fatalf("Expected 2 children, got %d", len(qpgi.PodGroupInfo.Children))
				}
				if qpgi.PodGroupInfo.Children[0].GetType() != fwk.PodGroupKeyType || qpgi.PodGroupInfo.Children[0].PodGroup.Spec.SchedulingPolicy.Gang.MinCount != 2 {
					t.Errorf("PG sibling was corrupted during CPG update")
				}
				if qpgi.PodGroupInfo.Children[1].GetType() != fwk.CompositePodGroupKeyType || qpgi.PodGroupInfo.Children[1].CompositePodGroup.Annotations["updated"] != "true" {
					t.Errorf("Child CPG not updated correctly")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.qpgi.UpdateGenericPodGroup(fwk.NewGenericCompositePodGroup(tt.updateCPG))
			tt.verify(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_RemoveCompositePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	cpgChild := st.MakeCompositePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	cpgNested := st.MakeCompositePodGroup().Name("cpg-nested").Namespace("ns1").ParentCompositePodGroup("cpg-child").Obj()

	pgLeaf := st.MakePodGroup().Name("pg-leaf").Namespace("ns1").ParentCompositePodGroup("cpg-nested").Obj()
	podKey := fwk.PodGroupKey("ns1", "pg-leaf")

	tests := []struct {
		name      string
		removeCPG *schedulingv1alpha3.CompositePodGroup
		qpgi      *QueuedPodGroupInfo
		verify    func(*testing.T, *QueuedPodGroupInfo, []*QueuedPodInfo)
	}{
		{
			name:      "Remove child CPG and its subtree pods",
			removeCPG: cpgChild,
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(cpgChild,
						newCompositePodGroupInfoForTest(cpgNested,
							newPodGroupInfoForTest(pgLeaf),
						),
					),
				),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					podKey: {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-leaf").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Child CPG not removed from hierarchy")
				}
				if len(removed) != 1 || removed[0].Pod.Name != "pod1" {
					t.Errorf("Subtree pods not correctly removed and returned")
				}
				if len(qpgi.QueuedPodInfos[podKey]) != 0 {
					t.Errorf("Pod not removed from QueuedPodInfos map")
				}
			},
		},
		{
			name:      "Remove non-existent CPG",
			removeCPG: st.MakeCompositePodGroup().Name("non-existent").Namespace("ns1").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(cpgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 {
					t.Errorf("Hierarchy should not be modified for non-existent CPG")
				}
				if len(removed) != 0 {
					t.Errorf("No pods should be removed")
				}
			},
		},
		{
			name:      "Remove nested CPG with multiple podgroups",
			removeCPG: cpgNested,
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(cpgChild,
						newCompositePodGroupInfoForTest(cpgNested,
							newPodGroupInfoForTest(pgLeaf),
							newPodGroupInfoForTest(st.MakePodGroup().Name("pg-leaf2").Namespace("ns1").ParentCompositePodGroup("cpg-nested").Obj()),
						),
					),
				),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					fwk.PodGroupKey("ns1", "pg-leaf"):  {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-leaf").Obj()}}},
					fwk.PodGroupKey("ns1", "pg-leaf2"): {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg-leaf2").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || len(qpgi.PodGroupInfo.Children[0].Children) != 0 {
					t.Errorf("Nested CPG not removed correctly, hierarchy is wrong: %v", qpgi.PodGroupInfo.Children)
				}
				if len(removed) != 2 {
					t.Errorf("Expected 2 pods to be removed, got %d", len(removed))
				}
				if len(qpgi.QueuedPodInfos[fwk.PodGroupKey("ns1", "pg-leaf")]) != 0 || len(qpgi.QueuedPodInfos[fwk.PodGroupKey("ns1", "pg-leaf2")]) != 0 {
					t.Errorf("Pods not removed from QueuedPodInfos map")
				}
			},
		},
		{
			name:      "Remove CPG while having a sibling PG with the same name",
			removeCPG: cpgChild,
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(st.MakePodGroup().Name("cpg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
					newCompositePodGroupInfoForTest(cpgChild,
						newCompositePodGroupInfoForTest(cpgNested,
							newPodGroupInfoForTest(pgLeaf),
						),
					),
				),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					podKey:                              {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-leaf").Obj()}}},
					fwk.PodGroupKey("ns1", "cpg-child"): {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod-sibling").Namespace("ns1").PodGroupName("cpg-child").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].GetType() != fwk.PodGroupKeyType {
					t.Errorf("Expected only sibling PG to remain, got %v", qpgi.PodGroupInfo.Children)
				}
				if len(removed) != 1 || removed[0].Pod.Name != "pod1" {
					t.Errorf("Subtree pods not correctly removed and returned: %v", removed)
				}
				if len(qpgi.QueuedPodInfos[podKey]) != 0 {
					t.Errorf("Nested pod not removed from QueuedPodInfos map")
				}
				if len(qpgi.QueuedPodInfos[fwk.PodGroupKey("ns1", "cpg-child")]) != 1 {
					t.Errorf("Sibling PG pods should not have been removed")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			removed := tt.qpgi.RemoveGenericPodGroup(fwk.NewGenericCompositePodGroup(tt.removeCPG))
			tt.verify(t, tt.qpgi, removed)
		})
	}
}

func TestQueuedPodGroupInfo_AddPodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	pgChild := st.MakePodGroup().Name("pg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").Obj()
	pgNotFoundParent := st.MakePodGroup().Name("pg-orphan").Namespace("ns1").ParentCompositePodGroup("non-existent").Obj()

	tests := []struct {
		name    string
		qpgi    *QueuedPodGroupInfo
		pgToAdd *schedulingv1beta1.PodGroup
		verify  func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Add child PG to root CPG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			pgToAdd: pgChild,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].GetName() != "pg-child" {
					t.Errorf("Child PG not added to root correctly")
				}
				if qpgi.PodGroupInfo.Children[0].GetType() != fwk.PodGroupKeyType {
					t.Errorf("Child PG has wrong key type")
				}
			},
		},
		{
			name: "Add standalone PG (should be ignored by hierarchy builder as it's the root itself)",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			pgToAdd: pgStandalone,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Standalone PG should not be added to a root CPG's children")
				}
			},
		},
		{
			name: "Add PG with non-existent parent",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			pgToAdd: pgNotFoundParent,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("PG with non-existent parent should not be added")
				}
			},
		},
		{
			name: "Add child PG to parent CPG while having a sibling PG with the same name as the parent",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(st.MakePodGroup().Name("cpg-intermediate").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
					newCompositePodGroupInfoForTest(st.MakeCompositePodGroup().Name("cpg-intermediate").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			pgToAdd: st.MakePodGroup().Name("pg-child-nested").Namespace("ns1").ParentCompositePodGroup("cpg-intermediate").Obj(),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				pgSiblingNode := qpgi.PodGroupInfo.Children[0]
				cpgIntermediateNode := qpgi.PodGroupInfo.Children[1]
				if len(pgSiblingNode.Children) != 0 {
					t.Errorf("Child PG was incorrectly added to PG node")
				}
				if len(cpgIntermediateNode.Children) != 1 || cpgIntermediateNode.Children[0].GetName() != "pg-child-nested" {
					t.Errorf("Child PG was not added to intermediate CPG node")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.qpgi.AddSubtree(newPodGroupInfoForTest(tt.pgToAdd))
			tt.verify(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_UpdatePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	pgChild := st.MakePodGroup().Name("pg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").MinCount(2).Obj()
	pgChildUpdated := st.MakePodGroup().Name("pg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").MinCount(5).Obj()

	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").MinCount(1).Obj()
	pgStandaloneUpdated := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").MinCount(3).Obj()

	tests := []struct {
		name     string
		qpgi     *QueuedPodGroupInfo
		updatePG *schedulingv1beta1.PodGroup
		verify   func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Update child PG in CPG hierarchy",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(pgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updatePG: pgChildUpdated,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if qpgi.PodGroupInfo.Children[0].PodGroup.Spec.SchedulingPolicy.Gang.MinCount != 5 {
					t.Errorf("Child PG not updated correctly")
				}
			},
		},
		{
			name: "Update standalone PG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newPodGroupInfoForTest(pgStandalone),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updatePG: pgStandaloneUpdated,
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if qpgi.PodGroupInfo.PodGroup.Spec.SchedulingPolicy.Gang.MinCount != 3 {
					t.Errorf("Standalone PG root not updated correctly")
				}
			},
		},
		{
			name: "Update non-existent PG",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newCompositePodGroupInfoForTest(cpgRoot),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updatePG: st.MakePodGroup().Name("non-existent").Namespace("ns1").Obj(),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Non-existent PG update should not alter hierarchy")
				}
			},
		},
		{
			name: "Update PG while having a sibling CPG with the same name",
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(st.MakeCompositePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
					newPodGroupInfoForTest(st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").MinCount(2).Obj()),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			updatePG: st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").MinCount(10).Obj(),
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if qpgi.PodGroupInfo.Children[0].GetType() != fwk.CompositePodGroupKeyType {
					t.Errorf("CPG node was overwritten during PG update")
				}
				if qpgi.PodGroupInfo.Children[1].GetType() != fwk.PodGroupKeyType || qpgi.PodGroupInfo.Children[1].PodGroup.Spec.SchedulingPolicy.Gang.MinCount != 10 {
					t.Errorf("PG node was not updated correctly")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.qpgi.UpdateGenericPodGroup(fwk.NewGenericPodGroup(tt.updatePG))
			tt.verify(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_RemovePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	pgChild := st.MakePodGroup().Name("pg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").Obj()

	podKeyChild := fwk.PodGroupKey("ns1", "pg-child")
	podKeyStandalone := fwk.PodGroupKey("ns1", "pg-standalone")

	tests := []struct {
		name     string
		removePG *schedulingv1beta1.PodGroup
		qpgi     *QueuedPodGroupInfo
		verify   func(*testing.T, *QueuedPodGroupInfo, []*QueuedPodInfo)
	}{
		{
			name:     "Remove child PG and its pods",
			removePG: pgChild,
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(pgChild),
				),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					podKeyChild: {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-child").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(qpgi.PodGroupInfo.Children) != 0 {
					t.Errorf("Child PG not removed from hierarchy")
				}
				if len(removed) != 1 || removed[0].Pod.Name != "pod1" {
					t.Errorf("Pods not correctly removed and returned")
				}
				if len(qpgi.QueuedPodInfos[podKeyChild]) != 0 {
					t.Errorf("Pod not removed from QueuedPodInfos map")
				}
			},
		},
		{
			name:     "Remove standalone PG and its pods (root removal)",
			removePG: pgStandalone,
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newPodGroupInfoForTest(pgStandalone),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					podKeyStandalone: {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg-standalone").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				// For standalone PG, removing the root essentially clears the QueuedPodInfos because
				// deleteSubtreePods will match the root node.
				if len(removed) != 1 || removed[0].Pod.Name != "pod2" {
					t.Errorf("Standalone pods not correctly removed and returned")
				}
				if len(qpgi.QueuedPodInfos[podKeyStandalone]) != 0 {
					t.Errorf("Pod not removed from QueuedPodInfos map")
				}
			},
		},
		{
			name:     "Remove PG while having a sibling CPG with the same name",
			removePG: st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newCompositePodGroupInfoForTest(
						st.MakeCompositePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj(),
						newPodGroupInfoForTest(st.MakePodGroup().Name("pg-nested").Namespace("ns1").ParentCompositePodGroup("shared-name").Obj()),
					),
					newPodGroupInfoForTest(st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()),
				),
				QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
					fwk.PodGroupKey("ns1", "pg-nested"):   {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod-nested").Namespace("ns1").PodGroupName("pg-nested").Obj()}}},
					fwk.PodGroupKey("ns1", "shared-name"): {{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod-sibling").Namespace("ns1").PodGroupName("shared-name").Obj()}}},
				},
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo, removed []*QueuedPodInfo) {
				if len(removed) != 1 || removed[0].Pod.Name != "pod-sibling" {
					t.Fatalf("Expected pod-sibling removed, got %v", removed)
				}
				if len(qpgi.PodGroupInfo.Children) != 1 || qpgi.PodGroupInfo.Children[0].GetType() != fwk.CompositePodGroupKeyType {
					t.Fatalf("Expected only CPG child to remain, got %v", qpgi.PodGroupInfo.Children)
				}
				if len(qpgi.QueuedPodInfos[fwk.PodGroupKey("ns1", "pg-nested")]) != 1 {
					t.Errorf("CPG's nested pods should not have been removed")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			removed := tt.qpgi.RemoveGenericPodGroup(fwk.NewGenericPodGroup(tt.removePG))
			tt.verify(t, tt.qpgi, removed)
		})
	}
}

func TestQueuedPodGroupInfo_AddPod(t *testing.T) {
	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	pgChild := st.MakePodGroup().Name("pg-child").Namespace("ns1").ParentCompositePodGroup("cpg-root").Obj()
	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").Obj()

	podKeyChild := fwk.PodGroupKey("ns1", "pg-child")
	podKeyStandalone := fwk.PodGroupKey("ns1", "pg-standalone")

	tests := []struct {
		name   string
		pod    *v1.Pod
		qpgi   *QueuedPodGroupInfo
		verify func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Add pod to CPG hierarchy leaf PG",
			pod:  st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-child").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(cpgRoot,
					newPodGroupInfoForTest(pgChild),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.QueuedPodInfos[podKeyChild]) != 1 {
					t.Errorf("Pod not added to QueuedPodInfos map for child PG")
				}
			},
		},
		{
			name: "Add pod to standalone PG",
			pod:  st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg-standalone").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newPodGroupInfoForTest(pgStandalone),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.QueuedPodInfos[podKeyStandalone]) != 1 {
					t.Errorf("Pod not added to QueuedPodInfos map for standalone PG")
				}
			},
		},
		{
			name: "Add pod that does not match leaf PG name",
			pod:  st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("non-existent-pg").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo:   newPodGroupInfoForTest(pgStandalone),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				if len(qpgi.QueuedPodInfos) != 0 {
					t.Errorf("Pod added despite non-existent leaf PG")
				}
			},
		},
		{
			name: "Add pod to a leaf PG, where the PG and its parent CPG share the same name",
			pod:  st.MakePod().Name("pod4").Namespace("ns1").PodGroupName("shared-name").Obj(),
			qpgi: &QueuedPodGroupInfo{
				PodGroupInfo: newCompositePodGroupInfoForTest(
					st.MakeCompositePodGroup().Name("shared-name").Namespace("ns1").Obj(),
					newPodGroupInfoForTest(st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("shared-name").Obj()),
				),
				QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
			},
			verify: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				key := fwk.PodGroupKey("ns1", "shared-name")
				if len(qpgi.QueuedPodInfos[key]) != 1 {
					t.Fatalf("Expected 1 queued pod info for leaf PG key, got %d", len(qpgi.QueuedPodInfos[key]))
				}
				leafInfo := qpgi.PodGroupInfo.Children[0]
				if len(leafInfo.UnscheduledPods) != 1 {
					t.Fatalf("Expected 1 unscheduled pod in leaf PG, got %d", len(leafInfo.UnscheduledPods))
				}
				if len(qpgi.PodGroupInfo.UnscheduledPods) != 0 {
					t.Fatalf("Expected 0 unscheduled pods directly on root CPG, got %d", len(qpgi.PodGroupInfo.UnscheduledPods))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pInfo := &QueuedPodInfo{PodInfo: &PodInfo{Pod: tt.pod}}
			tt.qpgi.AddPod(pInfo)
			tt.verify(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_UpdateAndRemovePod(t *testing.T) {
	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").Obj()
	podKeyStandalone := fwk.PodGroupKey("ns1", "pg-standalone")

	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg-standalone").Obj()
	pod1Updated := pod1.DeepCopy()
	pod1Updated.Annotations = map[string]string{"updated": "true"}

	cpgSameName := st.MakeCompositePodGroup().Name("shared-name").Namespace("ns1").Obj()
	pgSameName := st.MakePodGroup().Name("shared-name").Namespace("ns1").ParentCompositePodGroup("shared-name").Obj()
	podShared := st.MakePod().Name("pod-shared").Namespace("ns1").PodGroupName("shared-name").Obj()
	podSharedUpdated := podShared.DeepCopy()
	podSharedUpdated.Annotations = map[string]string{"updated": "true"}
	podKeyShared := fwk.PodGroupKey("ns1", "shared-name")

	newStandaloneQPGInfo := func() *QueuedPodGroupInfo {
		return &QueuedPodGroupInfo{
			PodGroupInfo: newPodGroupInfoForTest(pgStandalone),
			QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
				podKeyStandalone: {{PodInfo: &PodInfo{Pod: pod1}}},
			},
		}
	}

	newSharedNameQPGInfo := func() *QueuedPodGroupInfo {
		leafInfo := newPodGroupInfoForTest(pgSameName)
		leafInfo.UnscheduledPods = []*v1.Pod{podShared}
		return &QueuedPodGroupInfo{
			PodGroupInfo: newCompositePodGroupInfoForTest(cpgSameName, leafInfo),
			QueuedPodInfos: map[fwk.EntityKey][]*QueuedPodInfo{
				podKeyShared: {{PodInfo: &PodInfo{Pod: podShared}}},
			},
		}
	}

	tests := []struct {
		name    string
		qpgi    *QueuedPodGroupInfo
		execute func(*testing.T, *QueuedPodGroupInfo)
	}{
		{
			name: "Update Pod",
			qpgi: newStandaloneQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				pInfo, err := qpgi.Update(pod1Updated)
				if err != nil {
					t.Errorf("Update failed: %v", err)
				}
				if pInfo.Pod.Annotations["updated"] != "true" {
					t.Errorf("Pod was not correctly updated")
				}
			},
		},
		{
			name: "Update Pod not found",
			qpgi: newStandaloneQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				podNotFound := st.MakePod().Name("pod-not-found").Namespace("ns1").PodGroupName("pg-standalone").Obj()
				_, err := qpgi.Update(podNotFound)
				if err == nil {
					t.Errorf("Expected error when updating non-existent pod")
				}
			},
		},
		{
			name: "Remove Pod",
			qpgi: newStandaloneQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				removed := qpgi.RemovePod(pod1)
				if removed == nil || removed.Pod.Name != "pod1" {
					t.Errorf("Pod not correctly removed")
				}
				if len(qpgi.QueuedPodInfos[podKeyStandalone]) != 0 {
					t.Errorf("Pod still present in QueuedPodInfos")
				}
			},
		},
		{
			name: "Remove Pod not found",
			qpgi: newStandaloneQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				podNotFound := st.MakePod().Name("pod-not-found").Namespace("ns1").PodGroupName("pg-standalone").Obj()
				removed := qpgi.RemovePod(podNotFound)
				if removed != nil {
					t.Errorf("Expected nil when removing non-existent pod")
				}
			},
		},
		{
			name: "Update Pod in a leaf PG, where the PG and its parent CPG share the same name",
			qpgi: newSharedNameQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				pInfo, err := qpgi.Update(podSharedUpdated)
				if err != nil {
					t.Fatalf("Update failed: %v", err)
				}
				if pInfo.Pod.Annotations["updated"] != "true" {
					t.Errorf("Pod was not correctly updated")
				}
				leafInfo := qpgi.PodGroupInfo.Children[0]
				if len(leafInfo.UnscheduledPods) != 1 || leafInfo.UnscheduledPods[0].Annotations["updated"] != "true" {
					t.Errorf("Leaf PG UnscheduledPods was not updated")
				}
			},
		},
		{
			name: "Remove Pod from a leaf PG, where the PG and its parent CPG share the same name",
			qpgi: newSharedNameQPGInfo(),
			execute: func(t *testing.T, qpgi *QueuedPodGroupInfo) {
				removed := qpgi.RemovePod(podShared)
				if removed == nil || removed.Pod.Name != "pod-shared" {
					t.Fatalf("Pod not correctly removed")
				}
				if len(qpgi.QueuedPodInfos[podKeyShared]) != 0 {
					t.Errorf("Pod still present in QueuedPodInfos")
				}
				leafInfo := qpgi.PodGroupInfo.Children[0]
				if len(leafInfo.UnscheduledPods) != 0 {
					t.Errorf("Leaf PG UnscheduledPods was not cleared")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.execute(t, tt.qpgi)
		})
	}
}

func TestQueuedPodGroupInfo_ForEachPodInfo(t *testing.T) {
	qpgi := &QueuedPodGroupInfo{
		QueuedPodInfos: make(map[fwk.EntityKey][]*QueuedPodInfo),
	}

	podKey1 := fwk.PodGroupKey("ns1", "pg1")
	podKey2 := fwk.PodGroupKey("ns1", "pg2")

	qpgi.QueuedPodInfos[podKey1] = []*QueuedPodInfo{
		{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod1").Obj()}},
		{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod2").Obj()}},
	}
	qpgi.QueuedPodInfos[podKey2] = []*QueuedPodInfo{
		{PodInfo: &PodInfo{Pod: st.MakePod().Name("pod3").Obj()}},
	}

	count := 0
	for range qpgi.ForEachPodInfo() {
		count++
	}

	if count != 3 {
		t.Errorf("Expected 3 pods, got %d", count)
	}

	// Test early exit
	count = 0
	for range qpgi.ForEachPodInfo() {
		count++
		break
	}

	if count != 1 {
		t.Errorf("Expected 1 pod after early exit, got %d", count)
	}
}

func TestPodGroupInfo_GetUnscheduledPods(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").Obj()
	pod4 := st.MakePod().Name("pod4").Namespace("ns1").Obj()

	pgStandalone := st.MakePodGroup().Name("pg-standalone").Namespace("ns1").Obj()
	pgChild1 := st.MakePodGroup().Name("pg-child1").Namespace("ns1").Obj()
	pgChild2 := st.MakePodGroup().Name("pg-child2").Namespace("ns1").Obj()
	pgChild3 := st.MakePodGroup().Name("pg-child3").Namespace("ns1").Obj()
	cpgParent := st.MakeCompositePodGroup().Name("cpg-parent").Namespace("ns1").Obj()
	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").Namespace("ns1").Obj()
	cpgSub := st.MakeCompositePodGroup().Name("cpg-sub").Namespace("ns1").Obj()

	tests := []struct {
		name     string
		pgi      *PodGroupInfo
		expected []*v1.Pod
	}{
		{
			name: "Standalone PodGroupInfo with unscheduled pods",
			pgi: &PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(pgStandalone),
				UnscheduledPods: []*v1.Pod{pod1, pod2},
			},
			expected: []*v1.Pod{pod1, pod2},
		},
		{
			name: "PodGroupInfo with children",
			pgi: &PodGroupInfo{
				GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgParent),
				Children: []*PodGroupInfo{
					{
						GenericPodGroup: fwk.NewGenericPodGroup(pgChild1),
						UnscheduledPods: []*v1.Pod{pod1},
					},
					{
						GenericPodGroup: fwk.NewGenericPodGroup(pgChild2),
						UnscheduledPods: []*v1.Pod{pod2, pod3},
					},
				},
			},
			expected: []*v1.Pod{pod1, pod2, pod3},
		},
		{
			name: "Multi-level PodGroupInfo with children",
			pgi: &PodGroupInfo{
				GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgRoot),
				Children: []*PodGroupInfo{
					{
						GenericPodGroup: fwk.NewGenericCompositePodGroup(cpgSub),
						Children: []*PodGroupInfo{
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild1),
								UnscheduledPods: []*v1.Pod{pod1},
							},
							{
								GenericPodGroup: fwk.NewGenericPodGroup(pgChild2),
								UnscheduledPods: []*v1.Pod{pod2, pod3},
							},
						},
					},
					{
						GenericPodGroup: fwk.NewGenericPodGroup(pgChild3),
						UnscheduledPods: []*v1.Pod{pod4},
					},
				},
			},
			expected: []*v1.Pod{pod1, pod2, pod3, pod4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.pgi.GetUnscheduledPods()
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Errorf("GetUnscheduledPods() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func newPodGroupInfoForTest(pg *schedulingv1beta1.PodGroup, children ...*PodGroupInfo) *PodGroupInfo {
	return &PodGroupInfo{
		GenericPodGroup: fwk.NewGenericPodGroup(pg),
		Children:        children,
	}
}

func newCompositePodGroupInfoForTest(cpg *schedulingv1alpha3.CompositePodGroup, children ...*PodGroupInfo) *PodGroupInfo {
	return &PodGroupInfo{
		GenericPodGroup: fwk.NewGenericCompositePodGroup(cpg),
		Children:        children,
	}
}
