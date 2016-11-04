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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	testclient "k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func maxSurge(val int) *intstr.IntOrString {
	surge := intstr.FromInt(val)
	return &surge
}

func TestScale(t *testing.T) {
	newTimestamp := unversioned.Date(2016, 5, 20, 2, 0, 0, 0, time.UTC)
	oldTimestamp := unversioned.Date(2016, 5, 20, 1, 0, 0, 0, time.UTC)
	olderTimestamp := unversioned.Date(2016, 5, 20, 0, 0, 0, 0, time.UTC)

	var updatedTemplate = func(replicas int) *extensions.Deployment {
		d := newDeployment("foo", replicas, nil, nil, nil, map[string]string{"foo": "bar"})
		d.Spec.Template.Labels["another"] = "label"
		return d
	}

	tests := []struct {
		name          string
		deployment    *extensions.Deployment
		oldDeployment *extensions.Deployment

		newRS  *extensions.ReplicaSet
		oldRSs []*extensions.ReplicaSet

		expectedNew  *extensions.ReplicaSet
		expectedOld  []*extensions.ReplicaSet
		wasntUpdated map[string]bool

		desiredReplicasAnnotations map[string]int32
	}{
		{
			name:          "normal scaling event: 10 -> 12",
			deployment:    newDeployment("foo", 12, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 10, nil, nil, nil, nil),

			newRS:  rs("foo-v1", 10, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{},

			expectedNew: rs("foo-v1", 12, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{},
		},
		{
			name:          "normal scaling event: 10 -> 5",
			deployment:    newDeployment("foo", 5, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 10, nil, nil, nil, nil),

			newRS:  rs("foo-v1", 10, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{},

			expectedNew: rs("foo-v1", 5, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{},
		},
		{
			name:          "proportional scaling: 5 -> 10",
			deployment:    newDeployment("foo", 10, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 5, nil, nil, nil, nil),

			newRS:  rs("foo-v2", 2, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v1", 3, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 4, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v1", 6, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 5 -> 3",
			deployment:    newDeployment("foo", 3, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 5, nil, nil, nil, nil),

			newRS:  rs("foo-v2", 2, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v1", 3, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 1, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v1", 2, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 9 -> 4",
			deployment:    newDeployment("foo", 4, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 9, nil, nil, nil, nil),

			newRS:  rs("foo-v2", 8, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v1", 1, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 4, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v1", 0, nil, oldTimestamp)},
		},
		{
			name:          "proportional scaling: 7 -> 10",
			deployment:    newDeployment("foo", 10, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 7, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 3, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 3, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 4, nil, oldTimestamp), rs("foo-v1", 3, nil, olderTimestamp)},
		},
		{
			name:          "proportional scaling: 13 -> 8",
			deployment:    newDeployment("foo", 8, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 13, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 8, nil, oldTimestamp), rs("foo-v1", 3, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 1, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 5, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},
		},
		// Scales up the new replica set.
		{
			name:          "leftover distribution: 3 -> 4",
			deployment:    newDeployment("foo", 4, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 3, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 1, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 2, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},
		},
		// Scales down the older replica set.
		{
			name:          "leftover distribution: 3 -> 2",
			deployment:    newDeployment("foo", 2, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 3, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 1, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 1, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
		},
		// Scales up the latest replica set first.
		{
			name:          "proportional scaling (no new rs): 4 -> 5",
			deployment:    newDeployment("foo", 5, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 4, nil, nil, nil, nil),

			newRS:  nil,
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},

			expectedNew: nil,
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 3, nil, oldTimestamp), rs("foo-v1", 2, nil, olderTimestamp)},
		},
		// Scales down to zero
		{
			name:          "proportional scaling: 6 -> 0",
			deployment:    newDeployment("foo", 0, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 6, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 3, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew: rs("foo-v3", 0, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
		},
		// Scales up from zero
		{
			name:          "proportional scaling: 0 -> 6",
			deployment:    newDeployment("foo", 6, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 6, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 0, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},

			expectedNew:  rs("foo-v3", 6, nil, newTimestamp),
			expectedOld:  []*extensions.ReplicaSet{rs("foo-v2", 0, nil, oldTimestamp), rs("foo-v1", 0, nil, olderTimestamp)},
			wasntUpdated: map[string]bool{"foo-v2": true, "foo-v1": true},
		},
		// Scenario: deployment.spec.replicas == 3 ( foo-v1.spec.replicas == foo-v2.spec.replicas == foo-v3.spec.replicas == 1 )
		// Deployment is scaled to 5. foo-v3.spec.replicas and foo-v2.spec.replicas should increment by 1 but foo-v2 fails to
		// update.
		{
			name:          "failed rs update",
			deployment:    newDeployment("foo", 5, nil, nil, nil, nil),
			oldDeployment: newDeployment("foo", 5, nil, nil, nil, nil),

			newRS:  rs("foo-v3", 2, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 1, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},

			expectedNew:  rs("foo-v3", 2, nil, newTimestamp),
			expectedOld:  []*extensions.ReplicaSet{rs("foo-v2", 2, nil, oldTimestamp), rs("foo-v1", 1, nil, olderTimestamp)},
			wasntUpdated: map[string]bool{"foo-v3": true, "foo-v1": true},

			desiredReplicasAnnotations: map[string]int32{"foo-v2": int32(3)},
		},
		{
			name:          "deployment with surge pods",
			deployment:    newDeployment("foo", 20, nil, maxSurge(2), nil, nil),
			oldDeployment: newDeployment("foo", 10, nil, maxSurge(2), nil, nil),

			newRS:  rs("foo-v2", 6, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v1", 6, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 11, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v1", 11, nil, oldTimestamp)},
		},
		{
			name:          "change both surge and size",
			deployment:    newDeployment("foo", 50, nil, maxSurge(6), nil, nil),
			oldDeployment: newDeployment("foo", 10, nil, maxSurge(3), nil, nil),

			newRS:  rs("foo-v2", 5, nil, newTimestamp),
			oldRSs: []*extensions.ReplicaSet{rs("foo-v1", 8, nil, oldTimestamp)},

			expectedNew: rs("foo-v2", 22, nil, newTimestamp),
			expectedOld: []*extensions.ReplicaSet{rs("foo-v1", 34, nil, oldTimestamp)},
		},
		{
			name:          "change both size and template",
			deployment:    updatedTemplate(14),
			oldDeployment: newDeployment("foo", 10, nil, nil, nil, map[string]string{"foo": "bar"}),

			newRS:  nil,
			oldRSs: []*extensions.ReplicaSet{rs("foo-v2", 7, nil, newTimestamp), rs("foo-v1", 3, nil, oldTimestamp)},

			expectedNew: nil,
			expectedOld: []*extensions.ReplicaSet{rs("foo-v2", 10, nil, newTimestamp), rs("foo-v1", 4, nil, oldTimestamp)},
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

		// Construct the nameToSize map that will hold all the sizes we got our of tests
		// Skip updating the map if the replica set wasn't updated since there will be
		// no update action for it.
		nameToSize := make(map[string]int32)
		if test.newRS != nil {
			nameToSize[test.newRS.Name] = test.newRS.Spec.Replicas
		}
		for i := range test.oldRSs {
			rs := test.oldRSs[i]
			nameToSize[rs.Name] = rs.Spec.Replicas
		}
		// Get all the UPDATE actions and update nameToSize with all the updated sizes.
		for _, action := range fake.Actions() {
			rs := action.(testclient.UpdateAction).GetObject().(*extensions.ReplicaSet)
			if !test.wasntUpdated[rs.Name] {
				nameToSize[rs.Name] = rs.Spec.Replicas
			}
		}

		if test.expectedNew != nil && test.newRS != nil && test.expectedNew.Spec.Replicas != nameToSize[test.newRS.Name] {
			t.Errorf("%s: expected new replicas: %d, got: %d", test.name, test.expectedNew.Spec.Replicas, nameToSize[test.newRS.Name])
			continue
		}
		if len(test.expectedOld) != len(test.oldRSs) {
			t.Errorf("%s: expected %d old replica sets, got %d", test.name, len(test.expectedOld), len(test.oldRSs))
			continue
		}
		for n := range test.oldRSs {
			rs := test.oldRSs[n]
			expected := test.expectedOld[n]
			if expected.Spec.Replicas != nameToSize[rs.Name] {
				t.Errorf("%s: expected old (%s) replicas: %d, got: %d", test.name, rs.Name, expected.Spec.Replicas, nameToSize[rs.Name])
			}
		}
	}
}

func TestDeploymentController_cleanupDeployment(t *testing.T) {
	selector := map[string]string{"foo": "bar"}

	tests := []struct {
		oldRSs               []*extensions.ReplicaSet
		revisionHistoryLimit int32
		expectedDeletions    int
	}{
		{
			oldRSs: []*extensions.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
				newRSWithStatus("foo-3", 0, 0, selector),
			},
			revisionHistoryLimit: 1,
			expectedDeletions:    2,
		},
		{
			// Only delete the replica set with Spec.Replicas = Status.Replicas = 0.
			oldRSs: []*extensions.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 1, selector),
				newRSWithStatus("foo-3", 1, 0, selector),
				newRSWithStatus("foo-4", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    1,
		},

		{
			oldRSs: []*extensions.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    2,
		},
		{
			oldRSs: []*extensions.ReplicaSet{
				newRSWithStatus("foo-1", 1, 1, selector),
				newRSWithStatus("foo-2", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    0,
		},
	}

	for i := range tests {
		test := tests[i]
		fake := &fake.Clientset{}
		informers := informers.NewSharedInformerFactory(fake, controller.NoResyncPeriodFunc())
		controller := NewDeploymentController(informers.Deployments(), informers.ReplicaSets(), informers.Pods(), fake)

		controller.eventRecorder = &record.FakeRecorder{}
		controller.dListerSynced = alwaysReady
		controller.rsListerSynced = alwaysReady
		controller.podListerSynced = alwaysReady
		for _, rs := range test.oldRSs {
			controller.rsLister.Indexer.Add(rs)
		}

		stopCh := make(chan struct{})
		defer close(stopCh)
		informers.Start(stopCh)

		d := newDeployment("foo", 1, &test.revisionHistoryLimit, nil, nil, map[string]string{"foo": "bar"})
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
