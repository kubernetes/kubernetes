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
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	clocktesting "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestCalculateStatus(t *testing.T) {
	labelMap := map[string]string{"name": "foo"}
	fullLabelMap := map[string]string{"name": "foo", "type": "production"}
	notFullyLabelledRS := newReplicaSet(1, labelMap)
	// Set replica num to 2 for status condition testing (diff < 0, diff > 0)
	fullyLabelledRS := newReplicaSet(2, fullLabelMap)
	longMinReadySeconds := 3600
	longMinReadySecondsRS := newReplicaSet(1, fullLabelMap)
	longMinReadySecondsRS.Spec.MinReadySeconds = int32(longMinReadySeconds)

	asTerminating := func(pod *v1.Pod) *v1.Pod {
		pod.DeletionTimestamp = ptr.To(meta.Now())
		return pod
	}

	clock := clocktesting.NewFakePassiveClock(time.Now())

	rsStatusTests := []struct {
		name                                          string
		enableDeploymentReplicaSetTerminatingReplicas bool
		replicaset                                    *apps.ReplicaSet
		activePods                                    []*v1.Pod
		terminatingPods                               []*v1.Pod
		controllerFeatures                            *ReplicaSetControllerFeatures
		expectedReplicaSetStatus                      apps.ReplicaSetStatus
	}{
		{
			"1 fully labelled pod",
			false,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 not fully labelled pod",
			false,
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 0,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
		{
			"2 fully labelled pods",
			false,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 2,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
				TerminatingReplicas:  nil,
			},
		},
		{
			"2 fully labelled pods with DeploymentReplicaSetTerminatingReplicas",
			true,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 2,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
				TerminatingReplicas:  ptr.To[int32](0),
			},
		},
		{
			"2 not fully labelled pods",
			false,
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", notFullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 0,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 fully labelled pod, 1 not fully labelled pod",
			false,
			notFullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", notFullyLabelledRS, v1.PodRunning, nil, true),
				newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             2,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        2,
				AvailableReplicas:    2,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 non-ready pod",
			false,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodPending, nil, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        0,
				AvailableReplicas:    0,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 ready but non-available pod",
			false,
			longMinReadySecondsRS,
			[]*v1.Pod{
				newPod("pod1", longMinReadySecondsRS, v1.PodRunning, &meta.Time{Time: clock.Now()}, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    0,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 available pod with minReadySeconds",
			false,
			longMinReadySecondsRS,
			[]*v1.Pod{
				newPod("pod1", longMinReadySecondsRS, v1.PodRunning, &meta.Time{Time: clock.Now().Add(-time.Duration(longMinReadySeconds) * time.Second)}, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 available pod for a long time with minReadySeconds",
			false,
			longMinReadySecondsRS,
			[]*v1.Pod{
				newPod("pod1", longMinReadySecondsRS, v1.PodRunning, &meta.Time{Time: clock.Now().Add(2 * -time.Duration(longMinReadySeconds) * time.Second)}, true),
			},
			nil,
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 fully labelled pod and 1 terminating without DeploymentReplicaSetTerminatingReplicas",
			false,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			[]*v1.Pod{
				asTerminating(newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true)),
			},
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
		{
			"1 fully labelled pods and 2 terminating with DeploymentReplicaSetTerminatingReplicas",
			true,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			[]*v1.Pod{
				asTerminating(newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true)),
				asTerminating(newPod("pod3", fullyLabelledRS, v1.PodRunning, nil, true)),
			},
			nil,
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  ptr.To[int32](2),
			},
		},
		{
			"1 fully labelled pods and 2 terminating with DeploymentReplicaSetTerminatingReplicas (ReplicationController)",
			true,
			fullyLabelledRS,
			[]*v1.Pod{
				newPod("pod1", fullyLabelledRS, v1.PodRunning, nil, true),
			},
			[]*v1.Pod{
				asTerminating(newPod("pod2", fullyLabelledRS, v1.PodRunning, nil, true)),
				asTerminating(newPod("pod3", fullyLabelledRS, v1.PodRunning, nil, true)),
			},
			&ReplicaSetControllerFeatures{
				EnableStatusTerminatingReplicas: false,
			},
			apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 1,
				ReadyReplicas:        1,
				AvailableReplicas:    1,
				TerminatingReplicas:  nil,
			},
		},
	}

	for _, test := range rsStatusTests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeploymentReplicaSetTerminatingReplicas, test.enableDeploymentReplicaSetTerminatingReplicas)

			// test ReplicaSet controller default behavior unless specified otherwise in the test case
			controllerFeatures := ptr.Deref(test.controllerFeatures, DefaultReplicaSetControllerFeatures())
			replicaSetStatus := calculateStatus(test.replicaset, test.activePods, test.terminatingPods, nil, controllerFeatures, clock)
			if !reflect.DeepEqual(replicaSetStatus, test.expectedReplicaSetStatus) {
				t.Errorf("unexpected replicaset status: expected %v, got %v", test.expectedReplicaSetStatus, replicaSetStatus)
			}
		})
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
		activePods                   []*v1.Pod
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
		t.Run(test.name, func(t *testing.T) {
			clock := clocktesting.NewFakePassiveClock(time.Now())
			replicaSetStatus := calculateStatus(test.replicaset, test.activePods, nil, test.manageReplicasErr, DefaultReplicaSetControllerFeatures(), clock)
			// all test cases have at most 1 status condition
			if len(replicaSetStatus.Conditions) > 0 {
				test.expectedReplicaSetConditions[0].LastTransitionTime = replicaSetStatus.Conditions[0].LastTransitionTime
			}
			if !reflect.DeepEqual(replicaSetStatus.Conditions, test.expectedReplicaSetConditions) {
				t.Errorf("unexpected replicaset status: expected %v, got %v", test.expectedReplicaSetConditions, replicaSetStatus.Conditions)
			}
		})
	}
}
