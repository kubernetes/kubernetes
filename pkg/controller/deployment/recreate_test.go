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

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
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
		d          *extensions.Deployment
	}{
		{
			oldRSSizes: []int{3},
			d:          newDeployment("foo", 3, nil, nil, nil, map[string]string{"foo": "bar"}),
		},
	}

	for i := range tests {
		t.Logf("running scenario %d", i)
		test := tests[i]

		var oldRSs []*extensions.ReplicaSet
		var expected []runtime.Object

		for n, size := range test.oldRSSizes {
			rs := newReplicaSet(test.d, fmt.Sprintf("%s-%d", test.d.Name, n), size)
			oldRSs = append(oldRSs, rs)

			rsCopy := rs.DeepCopy()

			zero := int32(0)
			rsCopy.Spec.Replicas = &zero
			expected = append(expected, rsCopy)

			if *(oldRSs[n].Spec.Replicas) == *(expected[n].(*extensions.ReplicaSet).Spec.Replicas) {
				t.Errorf("broken test - original and expected RS have the same size")
			}
		}

		kc := fake.NewSimpleClientset(expected...)
		informers := informers.NewSharedInformerFactory(kc, controller.NoResyncPeriodFunc())
		c := NewDeploymentController(informers.Extensions().V1beta1().Deployments(), informers.Extensions().V1beta1().ReplicaSets(), informers.Core().V1().Pods(), kc)
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

		newRS  *extensions.ReplicaSet
		oldRSs []*extensions.ReplicaSet
		podMap map[types.UID]*v1.PodList

		expected bool
	}{
		{
			name:     "no old RSs",
			expected: false,
		},
		{
			name:     "old RSs with running pods",
			oldRSs:   []*extensions.ReplicaSet{rsWithUID("some-uid"), rsWithUID("other-uid")},
			podMap:   podMapWithUIDs([]string{"some-uid", "other-uid"}),
			expected: true,
		},
		{
			name:     "old RSs without pods but with non-zero status replicas",
			oldRSs:   []*extensions.ReplicaSet{newRSWithStatus("rs-blabla", 0, 1, nil)},
			expected: true,
		},
		{
			name:     "old RSs without pods or non-zero status replicas",
			oldRSs:   []*extensions.ReplicaSet{newRSWithStatus("rs-blabla", 0, 0, nil)},
			expected: false,
		},
	}

	for _, test := range tests {
		if expected, got := test.expected, oldPodsRunning(test.newRS, test.oldRSs, test.podMap); expected != got {
			t.Errorf("%s: expected %t, got %t", test.name, expected, got)
		}
	}
}

func rsWithUID(uid string) *extensions.ReplicaSet {
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
