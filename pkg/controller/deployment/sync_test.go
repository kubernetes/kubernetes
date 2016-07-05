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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	exp "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestScale(t *testing.T) {
	newTimestamp := unversioned.Date(2016, 5, 20, 2, 0, 0, 0, time.UTC)
	oldTimestamp := unversioned.Date(2016, 5, 20, 1, 0, 0, 0, time.UTC)
	olderTimestamp := unversioned.Date(2016, 5, 20, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name          string
		deployment    *exp.Deployment
		oldDeployment *exp.Deployment

		newRS  *exp.ReplicaSet
		oldRSs []*exp.ReplicaSet

		expectedNew *exp.ReplicaSet
		expectedOld []*exp.ReplicaSet

		desiredReplicasAnnotations map[string]int32
	}{
		{
			name:          "normal scaling event: 10 -> 12",
			deployment:    newDeployment(12, nil),
			oldDeployment: newDeployment(10, nil),

			newRS:  rs("foo-v1", 10, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{},

			expectedNew: rs("foo-v1", 12, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{},
		},
		{
			name:          "normal scaling event: 10 -> 5",
			deployment:    newDeployment(5, nil),
			oldDeployment: newDeployment(10, nil),

			newRS:  rs("foo-v1", 10, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{},

			expectedNew: rs("foo-v1", 5, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{},
		},
		{
			name:          "proportional scaling: 5 -> 10",
			deployment:    newDeployment(10, nil),
			oldDeployment: newDeployment(5, nil),

			newRS:  rs("foo-v2", 2, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v1", 3, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 4, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v1", 6, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 5 -> 3",
			deployment:    newDeployment(3, nil),
			oldDeployment: newDeployment(5, nil),

			newRS:  rs("foo-v2", 2, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v1", 3, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 1, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v1", 2, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 9 -> 4",
			deployment:    newDeployment(4, nil),
			oldDeployment: newDeployment(9, nil),

			newRS:  rs("foo-v2", 8, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v1", 1, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 4, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v1", 0, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 7 -> 10",
			deployment:    newDeployment(10, nil),
			oldDeployment: newDeployment(7, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 3, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 3, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 4, nil, oldTimestamp), rs("foo-v1", 3, nil, olderTimestamp)},
		},
		{
			name:          "proportional scaling: 13 -> 8",
			deployment:    newDeployment(8, nil),
			oldDeployment: newDeployment(13, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 8, nil, oldTimestamp), rs("foo-v1", 3, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 1, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 5, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},
		},
		// Scales up the new replica set.
		{
			name:          "leftover distribution: 3 -> 4",
			deployment:    newDeployment(4, nil),
			oldDeployment: newDeployment(3, nil),

			newRS:  rs("foo-v3", 1, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 2, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},
		},
		// Scales down the older replica set.
		{
			name:          "leftover distribution: 3 -> 2",
			deployment:    newDeployment(2, nil),
			oldDeployment: newDeployment(3, nil),

			newRS:  rs("foo-v3", 1, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 1, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
		},
		// Scales up the latest replica set first.
		{
			name:          "proportional scaling (no new rs): 4 -> 5",
			deployment:    newDeployment(5, nil),
			oldDeployment: newDeployment(4, nil),

			newRS:  nil,
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},

			expectedNew: nil,
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 3, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},
		},
		// Scales down to zero
		{
			name:          "proportional scaling: 6 -> 0",
			deployment:    newDeployment(0, nil),
			oldDeployment: newDeployment(6, nil),

			newRS:  rs("foo-v3", 3, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 0, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
		},
		// Scales up from zero
		{
			name:          "proportional scaling: 0 -> 6",
			deployment:    newDeployment(6, nil),
			oldDeployment: newDeployment(0, nil),

			newRS:  rs("foo-v3", 0, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 6, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
		},
		// Scenario: deployment.spec.replicas == 3 ( foo-v1.spec.replicas == foo-v2.spec.replicas == foo-v3.spec.replicas == 1 )
		// Deployment is scaled to 5. foo-v3.spec.replicas and foo-v2.spec.replicas should increment by 1 but foo-v2 fails to
		// update.
		{
			name:          "failed rs update",
			deployment:    newDeployment(5, nil),
			oldDeployment: newDeployment(5, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 2, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			desiredReplicasAnnotations: map[string]int32{"foo-v2": int32(3)},
		},
		{
			name:          "deployment with surge pods",
			deployment:    newDeploymentEnhanced(20, intstr.FromInt(2)),
			oldDeployment: newDeploymentEnhanced(10, intstr.FromInt(2)),

			newRS:  rs("foo-v2", 6, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v1", 6, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 11, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v1", 11, nil, oldTimestamp)},
		},
		{
			name:          "change both surge and size",
			deployment:    newDeploymentEnhanced(50, intstr.FromInt(6)),
			oldDeployment: newDeploymentEnhanced(10, intstr.FromInt(3)),

			newRS:  rs("foo-v2", 5, nil, newTimestamp),
			oldRSs: []*exp.ReplicaSet{rs("foo-v1", 8, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 22, nil, newTimestamp),
			expectedOld: []*exp.ReplicaSet{rs("foo-v1", 34, nil, oldTimestamp)},
		},
	}

	for _, test := range tests {
		_ = olderTimestamp
		t.Log(test.name)
		fake := fake.Clientset{}
		dc := &DeploymentController{
			client:        &fake,
			eventRecorder: &record.FakeRecorder{},
		}

		if test.newRS != nil {
			desiredReplicas := test.oldDeployment.Spec.Replicas
			if desired, ok := test.desiredReplicasAnnotations[test.newRS.Name]; ok {
				desiredReplicas = desired
			}
			deploymentutil.SetReplicasAnnotations(test.newRS, desiredReplicas, desiredReplicas+deploymentutil.MaxSurge(*test.oldDeployment))
		}
		for i := range test.oldRSs {
			rs := test.oldRSs[i]
			if rs == nil {
				continue
			}
			desiredReplicas := test.oldDeployment.Spec.Replicas
			if desired, ok := test.desiredReplicasAnnotations[rs.Name]; ok {
				desiredReplicas = desired
			}
			deploymentutil.SetReplicasAnnotations(rs, desiredReplicas, desiredReplicas+deploymentutil.MaxSurge(*test.oldDeployment))
		}

		if err := dc.scale(test.deployment, test.newRS, test.oldRSs); err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		}
		if test.expectedNew != nil && test.newRS != nil && test.expectedNew.Spec.Replicas != test.newRS.Spec.Replicas {
			t.Errorf("%s: expected new replicas: %d, got: %d", test.name, test.expectedNew.Spec.Replicas, test.newRS.Spec.Replicas)
			continue
		}
		if len(test.expectedOld) != len(test.oldRSs) {
			t.Errorf("%s: expected %d old replica sets, got %d", test.name, len(test.expectedOld), len(test.oldRSs))
			continue
		}
		for n := range test.oldRSs {
			rs := test.oldRSs[n]
			exp := test.expectedOld[n]
			if exp.Spec.Replicas != rs.Spec.Replicas {
				t.Errorf("%s: expected old (%s) replicas: %d, got: %d", test.name, rs.Name, exp.Spec.Replicas, rs.Spec.Replicas)
			}
		}
	}
}

func TestDeploymentController_cleanupDeployment(t *testing.T) {
	selector := map[string]string{"foo": "bar"}

	tests := []struct {
		oldRSs               []*exp.ReplicaSet
		revisionHistoryLimit int
		expectedDeletions    int
	}{
		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
				newRSWithStatus("foo-3", 0, 0, selector),
			},
			revisionHistoryLimit: 1,
			expectedDeletions:    2,
		},
		{
			// Only delete the replica set with Spec.Replicas = Status.Replicas = 0.
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 1, selector),
				newRSWithStatus("foo-3", 1, 0, selector),
				newRSWithStatus("foo-4", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    1,
		},

		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    2,
		},
		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 1, 1, selector),
				newRSWithStatus("foo-2", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    0,
		},
	}

	for i, test := range tests {
		fake := &fake.Clientset{}
		controller := NewDeploymentController(fake, controller.NoResyncPeriodFunc)

		controller.eventRecorder = &record.FakeRecorder{}
		controller.rsStoreSynced = alwaysReady
		controller.podStoreSynced = alwaysReady
		for _, rs := range test.oldRSs {
			controller.rsStore.Add(rs)
		}

		d := newDeployment(1, &tests[i].revisionHistoryLimit)
		controller.cleanupDeployment(test.oldRSs, d)

		gotDeletions := 0
		for _, action := range fake.Actions() {
			if "delete" == action.GetVerb() {
				gotDeletions++
			}
		}
		if gotDeletions != test.expectedDeletions {
			t.Errorf("expect %v old replica sets been deleted, but got %v", test.expectedDeletions, gotDeletions)
			continue
		}
	}
}
