/*
Copyright 2016 The Kubernetes Authors.

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

package deployment

import (
	"fmt"
	"testing"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/controller"
)

func TestScaleDownOldReplicaSets(t *testing.T) {
	tests := []struct {
		oldRSSizes []int
		d          *apps.Deployment
	}{
		{
			oldRSSizes: []int{3},
			d:          newDeployment("foo", 3, nil, nil, nil, map[string]string{"foo": "bar"}),
		},
	}

	for i := range tests {
		t.Logf("running scenario %d", i)
		test := tests[i]

		var oldRSs []*apps.ReplicaSet
		var expected []runtime.Object

		for n, size := range test.oldRSSizes {
			rs := newReplicaSet(test.d, fmt.Sprintf("%s-%d", test.d.Name, n), size)
			oldRSs = append(oldRSs, rs)

			rsCopy := rs.DeepCopy()

			zero := int32(0)
			rsCopy.Spec.Replicas = &zero
			expected = append(expected, rsCopy)

			if *(oldRSs[n].Spec.Replicas) == *(expected[n].(*apps.ReplicaSet).Spec.Replicas) {
				t.Errorf("broken test - original and expected RS have the same size")
			}
		}

		kc := fake.NewSimpleClientset(expected...)
		informers := informers.NewSharedInformerFactory(kc, controller.NoResyncPeriodFunc())
		c, err := NewDeploymentController(informers.Apps().V1().Deployments(), informers.Apps().V1().ReplicaSets(), informers.Core().V1().Pods(), kc)
		if err != nil {
			t.Fatalf("error creating Deployment controller: %v", err)
		}
		c.eventRecorder = &record.FakeRecorder{}

		c.scaleDownOldReplicaSetsForRecreate(oldRSs, test.d)
		for j := range oldRSs {
			rs := oldRSs[j]

			if *rs.Spec.Replicas != 0 {
				t.Errorf("rs %q has non-zero replicas", rs.Name)
			}
		}
	}
}

func TestOldPodsRunning(t *testing.T) {
	tests := []struct {
		name string

		newRS  *apps.ReplicaSet
		oldRSs []*apps.ReplicaSet
		podMap map[types.UID]*v1.PodList

		hasOldPodsRunning bool
	}{
		{
			name:              "no old RSs",
			hasOldPodsRunning: false,
		},
		{
			name:              "old RSs with running pods",
			oldRSs:            []*apps.ReplicaSet{rsWithUID("some-uid"), rsWithUID("other-uid")},
			podMap:            podMapWithUIDs([]string{"some-uid", "other-uid"}),
			hasOldPodsRunning: true,
		},
		{
			name:              "old RSs without pods but with non-zero status replicas",
			oldRSs:            []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 1, nil)},
			hasOldPodsRunning: true,
		},
		{
			name:              "old RSs without pods or non-zero status replicas",
			oldRSs:            []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			hasOldPodsRunning: false,
		},
		{
			name:   "old RSs with zero status replicas but pods in terminal state are present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID]*v1.PodList{
				"uid-1": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodFailed,
							},
						},
						{
							Status: v1.PodStatus{
								Phase: v1.PodSucceeded,
							},
						},
					},
				},
			},
			hasOldPodsRunning: false,
		},
		{
			name:   "old RSs with zero status replicas but pod in unknown phase present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID]*v1.PodList{
				"uid-1": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodUnknown,
							},
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas with pending pod present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID]*v1.PodList{
				"uid-1": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodPending,
							},
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas with running pod present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID]*v1.PodList{
				"uid-1": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodRunning,
							},
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas but pods in terminal state and pending are present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID]*v1.PodList{
				"uid-1": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodFailed,
							},
						},
						{
							Status: v1.PodStatus{
								Phase: v1.PodSucceeded,
							},
						},
					},
				},
				"uid-2": {
					Items: []v1.Pod{},
				},
				"uid-3": {
					Items: []v1.Pod{
						{
							Status: v1.PodStatus{
								Phase: v1.PodPending,
							},
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if expected, got := test.hasOldPodsRunning, oldPodsRunning(test.newRS, test.oldRSs, test.podMap); expected != got {
				t.Errorf("%s: expected %t, got %t", test.name, expected, got)
			}
		})
	}
}

func rsWithUID(uid string) *apps.ReplicaSet {
	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	rs := newReplicaSet(d, fmt.Sprintf("foo-%s", uid), 0)
	rs.UID = types.UID(uid)
	return rs
}

func podMapWithUIDs(uids []string) map[types.UID]*v1.PodList {
	podMap := make(map[types.UID]*v1.PodList)
	for _, uid := range uids {
		podMap[types.UID(uid)] = &v1.PodList{
			Items: []v1.Pod{
				{ /* supposedly a pod */ },
				{ /* supposedly another pod pod */ },
			},
		}
	}
	return podMap
}
