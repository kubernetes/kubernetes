/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

func newPod(now time.Time, ready bool, beforeSec int) api.Pod {
	conditionStatus := api.ConditionFalse
	if ready {
		conditionStatus = api.ConditionTrue
	}
	return api.Pod{
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{
					Type:               api.PodReady,
					LastTransitionTime: unversioned.NewTime(now.Add(-1 * time.Duration(beforeSec) * time.Second)),
					Status:             conditionStatus,
				},
			},
		},
	}
}

func TestGetReadyPodsCount(t *testing.T) {
	now := time.Now()
	tests := []struct {
		pods            []api.Pod
		minReadySeconds int
		expected        int
	}{
		{
			[]api.Pod{
				newPod(now, true, 0),
				newPod(now, true, 2),
				newPod(now, false, 1),
			},
			1,
			1,
		},
		{
			[]api.Pod{
				newPod(now, true, 2),
				newPod(now, true, 11),
				newPod(now, true, 5),
			},
			10,
			1,
		},
	}

	for _, test := range tests {
		if count := getReadyPodsCount(test.pods, test.minReadySeconds); count != test.expected {
			t.Errorf("Pods = %#v, minReadySeconds = %d, expected %d, got %d", test.pods, test.minReadySeconds, test.expected, count)
		}
	}
}
