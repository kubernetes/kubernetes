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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"fmt"
	"reflect"
	"testing"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
)

func TestCalculateStatus(t *testing.T) {
	labelMap := map[string]string{"name": "foo"}
	fullLabelMap := map[string]string{"name": "foo", "type": "production"}
	notFullyLabelledRS := newReplicaSet(1, labelMap)
	// Set replica num to 2 for status condition testing (diff < 0, diff > 0)
	fullyLabelledRS := newReplicaSet(2, fullLabelMap)
	longMinReadySecondsRS := newReplicaSet(1, fullLabelMap)
	longMinReadySecondsRS.Spec.MinReadySeconds = 3600

	rsStatusTests := []struct {
		name                     string
		replicaset               *apps.ReplicaSet
		filteredPods             []*v1.Pod
		expectedReplicaSetStatus apps.ReplicaSetStatus
	}{
		{
			"1 fully labelled pod",
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
			},
		},
		{
			"1 not fully labelled pod",
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 0,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
			},
		},
		{
			"2 fully labelled pods",
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 2,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
			},
		},
		{
			"2 not fully labelled pods",
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", notFullyLabelledRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 0,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
			},
		},
		{
			"1 fully labelled pod, 1 not fully labelled pod",
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
			},
		},
		{
			"1 non-ready pod",
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodPending, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        0,
				AvailableReplicas:    0,
			},
		},
		{
			"1 ready but non-available pod",
			longMinReadySecondsRS,
			[]*v1.Pod{
				newPod("pod1", longMinReadySecondsRS, v1.PodRunning, nil, true),
			},
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    0,
			},
		},
	}

	for _, test := range rsStatusTests {
		replicaSetStatus := calculateStatus(test.replicaset, test.filteredPods, nil)
		if !reflect.DeepEqual(replicaSetStatus, test.expectedReplicaSetStatus) {
			t.Errorf("%s: unexpected replicaset status: expected %v, got %v", test.name, test.expectedReplicaSetStatus, replicaSetStatus)
		}
	}
}

func TestCalculateStatusConditions(t *testing.T) {
	labelMap := map[string]string{"name": "foo"}
	rs := newReplicaSet(2, labelMap)
	replicaFailureRS := newReplicaSet(10, labelMap)
	replicaFailureRS.Status.Conditions = []apps.ReplicaSetCondition{
		{
			Type:   apps.ReplicaSetReplicaFailure,
			Status: v1.ConditionTrue,
		},
	}

	rsStatusConditionTests := []struct {
		name                         string
		replicaset                   *apps.ReplicaSet
		filteredPods                 []*v1.Pod
		manageReplicasErr            error
		expectedReplicaSetConditions []apps.ReplicaSetCondition
	}{

		{
			"manageReplicasErr != nil && failureCond == nil, diff < 0",
			rs,
			[]*v1.Pod{
				newPod("pod1", rs, v1.PodRunning, nil, true),
			},
			fmt.Errorf("fake manageReplicasErr"),
			[]apps.ReplicaSetCondition{
				{
					Type:    apps.ReplicaSetReplicaFailure,
					Status:  v1.ConditionTrue,
					Reason:  "FailedCreate",
					Message: "fake manageReplicasErr",
				},
			},
		},
		{
			"manageReplicasErr != nil && failureCond == nil, diff > 0",
			rs,
			[]*v1.Pod{
				newPod("pod1", rs, v1.PodRunning, nil, true),
				newPod("pod2", rs, v1.PodRunning, nil, true),
				newPod("pod3", rs, v1.PodRunning, nil, true),
			},
			fmt.Errorf("fake manageReplicasErr"),
			[]apps.ReplicaSetCondition{
				{
					Type:    apps.ReplicaSetReplicaFailure,
					Status:  v1.ConditionTrue,
					Reason:  "FailedDelete",
					Message: "fake manageReplicasErr",
				},
			},
		},
		{
			"manageReplicasErr == nil && failureCond != nil",
			replicaFailureRS,
			[]*v1.Pod{
				newPod("pod1", replicaFailureRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
		},
		{
			"manageReplicasErr != nil && failureCond != nil",
			replicaFailureRS,
			[]*v1.Pod{
				newPod("pod1", replicaFailureRS, v1.PodRunning, nil, true),
			},
			fmt.Errorf("fake manageReplicasErr"),
			[]apps.ReplicaSetCondition{
				{
					Type:   apps.ReplicaSetReplicaFailure,
					Status: v1.ConditionTrue,
				},
			},
		},
		{
			"manageReplicasErr == nil && failureCond == nil",
			rs,
			[]*v1.Pod{
				newPod("pod1", rs, v1.PodRunning, nil, true),
			},
			nil,
			nil,
		},
	}

	for _, test := range rsStatusConditionTests {
		replicaSetStatus := calculateStatus(test.replicaset, test.filteredPods, test.manageReplicasErr)
		// all test cases have at most 1 status condition
		if len(replicaSetStatus.Conditions) > 0 {
			test.expectedReplicaSetConditions[0].LastTransitionTime = replicaSetStatus.Conditions[0].LastTransitionTime
		}
		if !reflect.DeepEqual(replicaSetStatus.Conditions, test.expectedReplicaSetConditions) {
			t.Errorf("%s: unexpected replicaset status: expected %v, got %v", test.name, test.expectedReplicaSetConditions, replicaSetStatus.Conditions)
		}
	}
}
