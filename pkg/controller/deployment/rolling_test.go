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
	"context"
	"errors"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/utils/ptr"
)

func TestDeploymentController_rolloutRolling(t *testing.T) {
	scaleErr := errors.New("ReplicaSet update failed")
	statusErr := errors.New("Deployment status update failed")
	tests := []struct {
		name            string
		oldAvailable    int32
		newReplicas     int32
		progressExpired bool
		statusErr       error
	}{
		{
			name:            "healthy old replicas time out",
			oldAvailable:    1,
			newReplicas:     4,
			progressExpired: true,
		},
		{
			name:            "unhealthy old replicas time out",
			oldAvailable:    0,
			newReplicas:     4,
			progressExpired: true,
		},
		{
			name:         "status is updated before the deadline",
			oldAvailable: 1,
			newReplicas:  4,
		},
		{
			name:            "scaling and status update errors are returned",
			oldAvailable:    1,
			newReplicas:     4,
			progressExpired: true,
			statusErr:       statusErr,
		},
		{
			name:            "new ReplicaSet scale-up failure times out",
			oldAvailable:    1,
			newReplicas:     3,
			progressExpired: true,
		},
		{
			name:         "new ReplicaSet scale-up failure updates status before the deadline",
			oldAvailable: 1,
			newReplicas:  3,
		},
		{
			name:            "new ReplicaSet scale-up and status update errors are returned",
			oldAvailable:    1,
			newReplicas:     3,
			progressExpired: true,
			statusErr:       statusErr,
		},
		{
			name:            "new ReplicaSet scale-down failure times out",
			oldAvailable:    1,
			newReplicas:     5,
			progressExpired: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			maxSurge := intstr.FromInt32(1)
			maxUnavailable := intstr.FromInt32(1)
			selector := map[string]string{"app": "foo"}
			d := newDeployment("foo", 4, nil, &maxSurge, &maxUnavailable, selector)
			d.Generation = 2
			d.Spec.ProgressDeadlineSeconds = ptr.To[int32](600)
			d.Annotations[deploymentutil.RevisionAnnotation] = "2"

			newRS := rs("foo-new", test.newReplicas, selector, noTimestamp)
			newRS.UID = "foo-new"
			newRS.Spec.Template = *d.Spec.Template.DeepCopy()
			newRS.Annotations = map[string]string{deploymentutil.RevisionAnnotation: "2"}
			newRS.Status = apps.ReplicaSetStatus{Replicas: test.newReplicas, ReadyReplicas: test.newReplicas, AvailableReplicas: test.newReplicas}
			oldRS := rs("foo-old", 1, selector, noTimestamp)
			oldRS.UID = "foo-old"
			oldRS.Spec.Template = *d.Spec.Template.DeepCopy()
			oldRS.Spec.Template.Spec.Containers[0].Image = "foo/old"
			oldRS.Annotations = map[string]string{deploymentutil.RevisionAnnotation: "1"}
			oldRS.Status = apps.ReplicaSetStatus{Replicas: 1, ReadyReplicas: test.oldAvailable, AvailableReplicas: test.oldAvailable}
			allRSs := []*apps.ReplicaSet{oldRS, newRS}

			d.Status = calculateStatus(allRSs, newRS, d)
			d.Status.ObservedGeneration = 1
			condition := deploymentutil.NewDeploymentCondition(apps.DeploymentProgressing, v1.ConditionTrue, deploymentutil.ReplicaSetUpdatedReason, "Deployment is progressing.")
			if test.progressExpired {
				condition.LastUpdateTime = metav1.NewTime(time.Now().Add(-time.Hour))
			}
			deploymentutil.SetDeploymentCondition(&d.Status, *condition)

			client := fake.NewClientset(d, oldRS, newRS)
			client.PrependReactor("update", "replicasets", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, scaleErr
			})
			if test.statusErr != nil {
				client.PrependReactor("update", "deployments", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, test.statusErr
				})
			}
			dc := &DeploymentController{
				client:        client,
				eventRecorder: &record.FakeRecorder{},
			}
			_, ctx := ktesting.NewTestContext(t)
			err := dc.rolloutRolling(ctx, d, allRSs)
			if !errors.Is(err, scaleErr) {
				t.Errorf("expected scaling error %v, got %v", scaleErr, err)
			}
			if test.statusErr != nil && !errors.Is(err, test.statusErr) {
				t.Errorf("expected status update error %v, got %v", test.statusErr, err)
			}

			actions := client.Actions()
			if len(actions) != 2 {
				t.Fatalf("expected ReplicaSet update followed by Deployment status update, got %v", actions)
			}
			if !actions[0].Matches("update", "replicasets") {
				t.Fatalf("expected ReplicaSet update, got %v", actions[0])
			}
			scaledRS := actions[0].(core.UpdateAction).GetObject().(*apps.ReplicaSet)
			expectedRS, expectedReplicas := oldRS.Name, int32(0)
			if test.newReplicas != *d.Spec.Replicas {
				expectedRS, expectedReplicas = newRS.Name, *d.Spec.Replicas
			}
			if scaledRS.Name != expectedRS || *scaledRS.Spec.Replicas != expectedReplicas {
				t.Errorf("expected ReplicaSet %s to scale to %d replicas, got %s with %d replicas", expectedRS, expectedReplicas, scaledRS.Name, *scaledRS.Spec.Replicas)
			}
			if !actions[1].Matches("update", "deployments") || actions[1].GetSubresource() != "status" {
				t.Fatalf("expected Deployment status update, got %v", actions[1])
			}
			updated := actions[1].(core.UpdateAction).GetObject().(*apps.Deployment)
			progressing := deploymentutil.GetDeploymentCondition(updated.Status, apps.DeploymentProgressing)
			if progressing == nil {
				t.Fatal("expected Progressing condition")
			}
			expectedReason := deploymentutil.ReplicaSetUpdatedReason
			expectedStatus := v1.ConditionTrue
			if test.progressExpired {
				expectedReason = deploymentutil.TimedOutReason
				expectedStatus = v1.ConditionFalse
			}
			if progressing.Reason != expectedReason || progressing.Status != expectedStatus {
				t.Errorf("expected Progressing condition %s with reason %s, got %+v", expectedStatus, expectedReason, progressing)
			}
			if updated.Status.ObservedGeneration != d.Generation {
				t.Errorf("expected observed generation %d, got %d", d.Generation, updated.Status.ObservedGeneration)
			}
			if test.statusErr == nil {
				stored, err := client.AppsV1().Deployments(d.Namespace).Get(ctx, d.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				storedCondition := deploymentutil.GetDeploymentCondition(stored.Status, apps.DeploymentProgressing)
				if storedCondition == nil || storedCondition.Status != expectedStatus || storedCondition.Reason != expectedReason {
					t.Errorf("expected persisted Progressing condition %s with reason %s, got %+v", expectedStatus, expectedReason, storedCondition)
				}
			}
		})
	}
}

func TestDeploymentController_reconcileNewReplicaSet(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int32
		maxSurge            intstr.IntOrString
		oldReplicas         int32
		newReplicas         int32
		scaleExpected       bool
		expectedNewReplicas int32
	}{
		{
			// Should not scale up.
			deploymentReplicas: 10,
			maxSurge:           intstr.FromInt32(0),
			oldReplicas:        10,
			newReplicas:        0,
			scaleExpected:      false,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt32(2),
			oldReplicas:         10,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 2,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt32(2),
			oldReplicas:         5,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 7,
		},
		{
			deploymentReplicas: 10,
			maxSurge:           intstr.FromInt32(2),
			oldReplicas:        10,
			newReplicas:        2,
			scaleExpected:      false,
		},
		{
			// Should scale down.
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt32(2),
			oldReplicas:         2,
			newReplicas:         11,
			scaleExpected:       true,
			expectedNewReplicas: 10,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Logf("executing scenario %d", i)
		newRS := rs("foo-v2", test.newReplicas, nil, noTimestamp)
		oldRS := rs("foo-v2", test.oldReplicas, nil, noTimestamp)
		allRSs := []*apps.ReplicaSet{newRS, oldRS}
		maxUnavailable := intstr.FromInt32(0)
		deployment := newDeployment("foo", test.deploymentReplicas, nil, &test.maxSurge, &maxUnavailable, map[string]string{"foo": "bar"})
		fake := fake.Clientset{}
		controller := &DeploymentController{
			client:        &fake,
			eventRecorder: &record.FakeRecorder{},
		}
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		scaled, err := controller.reconcileNewReplicaSet(ctx, allRSs, newRS, deployment)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !test.scaleExpected {
			if scaled || len(fake.Actions()) > 0 {
				t.Errorf("unexpected scaling: %v", fake.Actions())
			}
			continue
		}
		if test.scaleExpected && !scaled {
			t.Errorf("expected scaling to occur")
			continue
		}
		if len(fake.Actions()) != 1 {
			t.Errorf("expected 1 action during scale, got: %v", fake.Actions())
			continue
		}
		updated := fake.Actions()[0].(core.UpdateAction).GetObject().(*apps.ReplicaSet)
		if e, a := test.expectedNewReplicas, *(updated.Spec.Replicas); e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}

func TestDeploymentController_reconcileOldReplicaSets(t *testing.T) {
	errFakeRSUpdate := errors.New("fake ReplicaSet update failure")

	tests := []struct {
		deploymentReplicas  int32
		maxUnavailable      intstr.IntOrString
		oldReplicas         int32
		newReplicas         int32
		readyPodsFromOldRS  int
		readyPodsFromNewRS  int
		scaleExpected       bool
		expectedOldReplicas int32
		// If set, ReplicaSet updates fail with this error; reconcileOldReplicaSets must return it.
		updateErr error
	}{
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(0),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  10,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 9,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  10,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // expect unhealthy replicas from old replica sets been cleaned up
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  8,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // expect 1 unhealthy replica from old replica sets been cleaned up, and 1 ready pod been scaled down
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  9,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // the unavailable pods from the newRS would not make us scale down old RSs in a further step
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			oldReplicas:        8,
			newReplicas:        2,
			readyPodsFromOldRS: 8,
			readyPodsFromNewRS: 0,
			scaleExpected:      false,
		},
		{ // ready old replicas: scaleDownOldReplicaSetsForRollingUpdate update failure must surface
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			oldReplicas:        10,
			newReplicas:        0,
			readyPodsFromOldRS: 10,
			readyPodsFromNewRS: 0,
			updateErr:          errFakeRSUpdate,
		},
		{ // unavailable old replicas: cleanupUnhealthyReplicas update failure must surface
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			oldReplicas:        10,
			newReplicas:        0,
			readyPodsFromOldRS: 8,
			readyPodsFromNewRS: 0,
			updateErr:          errFakeRSUpdate,
		},
	}
	for i := range tests {
		test := tests[i]
		t.Logf("executing scenario %d", i)

		newSelector := map[string]string{"foo": "new"}
		oldSelector := map[string]string{"foo": "old"}
		newRS := rs("foo-new", test.newReplicas, newSelector, noTimestamp)
		newRS.Status.AvailableReplicas = int32(test.readyPodsFromNewRS)
		oldRS := rs("foo-old", test.oldReplicas, oldSelector, noTimestamp)
		oldRS.Status.AvailableReplicas = int32(test.readyPodsFromOldRS)
		oldRSs := []*apps.ReplicaSet{oldRS}
		allRSs := []*apps.ReplicaSet{oldRS, newRS}
		maxSurge := intstr.FromInt32(0)
		deployment := newDeployment("foo", test.deploymentReplicas, nil, &maxSurge, &test.maxUnavailable, newSelector)
		fakeClientset := fake.Clientset{}
		if test.updateErr != nil {
			updateErr := test.updateErr
			fakeClientset.PrependReactor("update", "replicasets", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, updateErr
			})
		}
		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}
		_, ctx := ktesting.NewTestContext(t)
		scaled, err := controller.reconcileOldReplicaSets(ctx, allRSs, oldRSs, newRS, deployment)
		if test.updateErr != nil {
			if !errors.Is(err, test.updateErr) {
				t.Errorf("expected error %v, got %v", test.updateErr, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !test.scaleExpected && scaled {
			t.Errorf("unexpected scaling: %v", fakeClientset.Actions())
		}
		if test.scaleExpected && !scaled {
			t.Errorf("expected scaling to occur")
			continue
		}
		continue
	}
}

func TestDeploymentController_cleanupUnhealthyReplicas(t *testing.T) {
	tests := []struct {
		oldReplicas          int32
		readyPods            int
		unHealthyPods        int
		maxCleanupCount      int
		cleanupCountExpected int
	}{
		{
			oldReplicas:          10,
			readyPods:            8,
			unHealthyPods:        2,
			maxCleanupCount:      1,
			cleanupCountExpected: 1,
		},
		{
			oldReplicas:          10,
			readyPods:            8,
			unHealthyPods:        2,
			maxCleanupCount:      3,
			cleanupCountExpected: 2,
		},
		{
			oldReplicas:          10,
			readyPods:            8,
			unHealthyPods:        2,
			maxCleanupCount:      0,
			cleanupCountExpected: 0,
		},
		{
			oldReplicas:          10,
			readyPods:            10,
			unHealthyPods:        0,
			maxCleanupCount:      3,
			cleanupCountExpected: 0,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		oldRS := rs("foo-v2", test.oldReplicas, nil, noTimestamp)
		oldRS.Status.AvailableReplicas = int32(test.readyPods)
		oldRSs := []*apps.ReplicaSet{oldRS}
		maxSurge := intstr.FromInt32(2)
		maxUnavailable := intstr.FromInt32(2)
		deployment := newDeployment("foo", 10, nil, &maxSurge, &maxUnavailable, nil)
		fakeClientset := fake.Clientset{}

		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}
		_, ctx := ktesting.NewTestContext(t)
		_, cleanupCount, err := controller.cleanupUnhealthyReplicas(ctx, oldRSs, deployment, int32(test.maxCleanupCount))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if int(cleanupCount) != test.cleanupCountExpected {
			t.Errorf("expected %v unhealthy replicas been cleaned up, got %v", test.cleanupCountExpected, cleanupCount)
			continue
		}
	}
}

func TestDeploymentController_scaleDownOldReplicaSetsForRollingUpdate(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int32
		maxUnavailable      intstr.IntOrString
		readyPods           int
		oldReplicas         int32
		scaleExpected       bool
		expectedOldReplicas int32
	}{
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(0),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 9,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt32(2),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			readyPods:          8,
			oldReplicas:        10,
			scaleExpected:      false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			readyPods:          10,
			oldReplicas:        0,
			scaleExpected:      false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt32(2),
			readyPods:          1,
			oldReplicas:        10,
			scaleExpected:      false,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Logf("executing scenario %d", i)
		oldRS := rs("foo-v2", test.oldReplicas, nil, noTimestamp)
		oldRS.Status.AvailableReplicas = int32(test.readyPods)
		allRSs := []*apps.ReplicaSet{oldRS}
		oldRSs := []*apps.ReplicaSet{oldRS}
		maxSurge := intstr.FromInt32(0)
		deployment := newDeployment("foo", test.deploymentReplicas, nil, &maxSurge, &test.maxUnavailable, map[string]string{"foo": "bar"})
		fakeClientset := fake.Clientset{}
		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}
		_, ctx := ktesting.NewTestContext(t)
		scaled, err := controller.scaleDownOldReplicaSetsForRollingUpdate(ctx, allRSs, oldRSs, deployment)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !test.scaleExpected {
			if scaled != 0 {
				t.Errorf("unexpected scaling: %v", fakeClientset.Actions())
			}
			continue
		}
		if test.scaleExpected && scaled == 0 {
			t.Errorf("expected scaling to occur; actions: %v", fakeClientset.Actions())
			continue
		}
		// There are both list and update actions logged, so extract the update
		// action for verification.
		var updateAction core.UpdateAction
		for _, action := range fakeClientset.Actions() {
			switch a := action.(type) {
			case core.UpdateAction:
				if updateAction != nil {
					t.Errorf("expected only 1 update action; had %v and found %v", updateAction, a)
				} else {
					updateAction = a
				}
			}
		}
		if updateAction == nil {
			t.Errorf("expected an update action")
			continue
		}
		updated := updateAction.GetObject().(*apps.ReplicaSet)
		if e, a := test.expectedOldReplicas, *(updated.Spec.Replicas); e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}
