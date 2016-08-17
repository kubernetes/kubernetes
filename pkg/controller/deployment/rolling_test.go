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

	"k8s.io/kubernetes/pkg/api"
	exp "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestDeploymentController_reconcileNewReplicaSet(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int
		maxSurge            intstr.IntOrString
		oldReplicas         int
		newReplicas         int
		scaleExpected       bool
		expectedNewReplicas int
	}{
		{
			// Should not scale up.
			deploymentReplicas: 10,
			maxSurge:           intstr.FromInt(0),
			oldReplicas:        10,
			newReplicas:        0,
			scaleExpected:      false,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt(2),
			oldReplicas:         10,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 2,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt(2),
			oldReplicas:         5,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 7,
		},
		{
			deploymentReplicas: 10,
			maxSurge:           intstr.FromInt(2),
			oldReplicas:        10,
			newReplicas:        2,
			scaleExpected:      false,
		},
		{
			// Should scale down.
			deploymentReplicas:  10,
			maxSurge:            intstr.FromInt(2),
			oldReplicas:         2,
			newReplicas:         11,
			scaleExpected:       true,
			expectedNewReplicas: 10,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		newRS := rs("foo-v2", test.newReplicas, nil, noTimestamp)
		oldRS := rs("foo-v2", test.oldReplicas, nil, noTimestamp)
		allRSs := []*exp.ReplicaSet{newRS, oldRS}
		deployment := deployment("foo", test.deploymentReplicas, test.maxSurge, intstr.FromInt(0), nil)
		fake := fake.Clientset{}
		controller := &DeploymentController{
			client:        &fake,
			eventRecorder: &record.FakeRecorder{},
		}
		scaled, err := controller.reconcileNewReplicaSet(allRSs, newRS, &deployment)
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
		updated := fake.Actions()[0].(core.UpdateAction).GetObject().(*exp.ReplicaSet)
		if e, a := test.expectedNewReplicas, int(updated.Spec.Replicas); e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}

func TestDeploymentController_reconcileOldReplicaSets(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int
		maxUnavailable      intstr.IntOrString
		oldReplicas         int
		newReplicas         int
		readyPodsFromOldRS  int
		readyPodsFromNewRS  int
		scaleExpected       bool
		expectedOldReplicas int
	}{
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(0),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  10,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 9,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  10,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // expect unhealthy replicas from old replica sets been cleaned up
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  8,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // expect 1 unhealthy replica from old replica sets been cleaned up, and 1 ready pod been scaled down
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(2),
			oldReplicas:         10,
			newReplicas:         0,
			readyPodsFromOldRS:  9,
			readyPodsFromNewRS:  0,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{ // the unavailable pods from the newRS would not make us scale down old RSs in a further step
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			oldReplicas:        8,
			newReplicas:        2,
			readyPodsFromOldRS: 8,
			readyPodsFromNewRS: 0,
			scaleExpected:      false,
		},
	}
	for i, test := range tests {
		t.Logf("executing scenario %d", i)

		newSelector := map[string]string{"foo": "new"}
		oldSelector := map[string]string{"foo": "old"}
		newRS := rs("foo-new", test.newReplicas, newSelector, noTimestamp)
		oldRS := rs("foo-old", test.oldReplicas, oldSelector, noTimestamp)
		oldRSs := []*exp.ReplicaSet{oldRS}
		allRSs := []*exp.ReplicaSet{oldRS, newRS}

		deployment := deployment("foo", test.deploymentReplicas, intstr.FromInt(0), test.maxUnavailable, newSelector)
		fakeClientset := fake.Clientset{}
		fakeClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case core.ListAction:
				podList := &api.PodList{}
				for podIndex := 0; podIndex < test.readyPodsFromOldRS; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:   fmt.Sprintf("%s-oldReadyPod-%d", oldRS.Name, podIndex),
							Labels: oldSelector,
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionTrue,
								},
							},
						},
					})
				}
				for podIndex := 0; podIndex < test.oldReplicas-test.readyPodsFromOldRS; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:   fmt.Sprintf("%s-oldUnhealthyPod-%d", oldRS.Name, podIndex),
							Labels: oldSelector,
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionFalse,
								},
							},
						},
					})
				}
				for podIndex := 0; podIndex < test.readyPodsFromNewRS; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:   fmt.Sprintf("%s-newReadyPod-%d", oldRS.Name, podIndex),
							Labels: newSelector,
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionTrue,
								},
							},
						},
					})
				}
				for podIndex := 0; podIndex < test.oldReplicas-test.readyPodsFromOldRS; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:   fmt.Sprintf("%s-newUnhealthyPod-%d", oldRS.Name, podIndex),
							Labels: newSelector,
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionFalse,
								},
							},
						},
					})
				}
				return true, podList, nil
			}
			return false, nil, nil
		})
		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}

		scaled, err := controller.reconcileOldReplicaSets(allRSs, oldRSs, newRS, &deployment)
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
		oldReplicas          int
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
		oldRSs := []*exp.ReplicaSet{oldRS}
		deployment := deployment("foo", 10, intstr.FromInt(2), intstr.FromInt(2), nil)
		fakeClientset := fake.Clientset{}
		fakeClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case core.ListAction:
				podList := &api.PodList{}
				for podIndex := 0; podIndex < test.readyPods; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: fmt.Sprintf("%s-readyPod-%d", oldRS.Name, podIndex),
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionTrue,
								},
							},
						},
					})
				}
				for podIndex := 0; podIndex < test.unHealthyPods; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: fmt.Sprintf("%s-unHealthyPod-%d", oldRS.Name, podIndex),
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionFalse,
								},
							},
						},
					})
				}
				return true, podList, nil
			}
			return false, nil, nil
		})

		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}
		_, cleanupCount, err := controller.cleanupUnhealthyReplicas(oldRSs, &deployment, 0, int32(test.maxCleanupCount))
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
		deploymentReplicas  int
		maxUnavailable      intstr.IntOrString
		readyPods           int
		oldReplicas         int
		scaleExpected       bool
		expectedOldReplicas int
		errorExpected       bool
	}{
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(0),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 9,
			errorExpected:       false,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(2),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 8,
			errorExpected:       false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			readyPods:          8,
			oldReplicas:        10,
			scaleExpected:      false,
			errorExpected:      false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			readyPods:          10,
			oldReplicas:        0,
			scaleExpected:      false,
			errorExpected:      true,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			readyPods:          1,
			oldReplicas:        10,
			scaleExpected:      false,
			errorExpected:      false,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		oldRS := rs("foo-v2", test.oldReplicas, nil, noTimestamp)
		allRSs := []*exp.ReplicaSet{oldRS}
		oldRSs := []*exp.ReplicaSet{oldRS}
		deployment := deployment("foo", test.deploymentReplicas, intstr.FromInt(0), test.maxUnavailable, map[string]string{"foo": "bar"})
		fakeClientset := fake.Clientset{}
		fakeClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case core.ListAction:
				podList := &api.PodList{}
				for podIndex := 0; podIndex < test.readyPods; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name:   fmt.Sprintf("%s-pod-%d", oldRS.Name, podIndex),
							Labels: map[string]string{"foo": "bar"},
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Type:   api.PodReady,
									Status: api.ConditionTrue,
								},
							},
						},
					})
				}
				return true, podList, nil
			}
			return false, nil, nil
		})
		controller := &DeploymentController{
			client:        &fakeClientset,
			eventRecorder: &record.FakeRecorder{},
		}
		scaled, err := controller.scaleDownOldReplicaSetsForRollingUpdate(allRSs, oldRSs, &deployment)
		if !test.errorExpected && err != nil {
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
		updated := updateAction.GetObject().(*exp.ReplicaSet)
		if e, a := test.expectedOldReplicas, int(updated.Spec.Replicas); e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}
