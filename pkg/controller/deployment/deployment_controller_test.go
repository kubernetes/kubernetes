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
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	exp "k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func TestDeploymentController_reconcileNewRC(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int
		maxSurge            util.IntOrString
		oldReplicas         int
		newReplicas         int
		scaleExpected       bool
		expectedNewReplicas int
	}{
		{
			// Should not scale up.
			deploymentReplicas: 10,
			maxSurge:           util.NewIntOrStringFromInt(0),
			oldReplicas:        10,
			newReplicas:        0,
			scaleExpected:      false,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            util.NewIntOrStringFromInt(2),
			oldReplicas:         10,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 2,
		},
		{
			deploymentReplicas:  10,
			maxSurge:            util.NewIntOrStringFromInt(2),
			oldReplicas:         5,
			newReplicas:         0,
			scaleExpected:       true,
			expectedNewReplicas: 7,
		},
		{
			deploymentReplicas: 10,
			maxSurge:           util.NewIntOrStringFromInt(2),
			oldReplicas:        10,
			newReplicas:        2,
			scaleExpected:      false,
		},
		{
			// Should scale down.
			deploymentReplicas:  10,
			maxSurge:            util.NewIntOrStringFromInt(2),
			oldReplicas:         2,
			newReplicas:         11,
			scaleExpected:       true,
			expectedNewReplicas: 10,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		newRc := rc("foo-v2", test.newReplicas)
		oldRc := rc("foo-v2", test.oldReplicas)
		allRcs := []*api.ReplicationController{newRc, oldRc}
		deployment := deployment("foo", test.deploymentReplicas, test.maxSurge, util.NewIntOrStringFromInt(0))
		fake := &testclient.Fake{}
		controller := &DeploymentController{
			client:        fake,
			eventRecorder: &record.FakeRecorder{},
		}
		scaled, err := controller.reconcileNewRC(allRcs, newRc, deployment)
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
		updated := fake.Actions()[0].(testclient.UpdateAction).GetObject().(*api.ReplicationController)
		if e, a := test.expectedNewReplicas, updated.Spec.Replicas; e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}

func TestDeploymentController_reconcileOldRCs(t *testing.T) {
	tests := []struct {
		deploymentReplicas  int
		maxUnavailable      util.IntOrString
		readyPods           int
		oldReplicas         int
		scaleExpected       bool
		expectedOldReplicas int
	}{
		{
			deploymentReplicas: 10,
			maxUnavailable:     util.NewIntOrStringFromInt(0),
			readyPods:          10,
			oldReplicas:        10,
			scaleExpected:      false,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      util.NewIntOrStringFromInt(2),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     util.NewIntOrStringFromInt(2),
			readyPods:          8,
			oldReplicas:        10,
			scaleExpected:      false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     util.NewIntOrStringFromInt(2),
			readyPods:          10,
			oldReplicas:        0,
			scaleExpected:      false,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		oldRc := rc("foo-v2", test.oldReplicas)
		allRcs := []*api.ReplicationController{oldRc}
		oldRcs := []*api.ReplicationController{oldRc}
		deployment := deployment("foo", test.deploymentReplicas, util.NewIntOrStringFromInt(0), test.maxUnavailable)
		fake := &testclient.Fake{}
		fake.AddReactor("list", "pods", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case testclient.ListAction:
				podList := &api.PodList{}
				for podIndex := 0; podIndex < test.readyPods; podIndex++ {
					podList.Items = append(podList.Items, api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: fmt.Sprintf("%s-pod-%d", oldRc.Name, podIndex),
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
			client:        fake,
			eventRecorder: &record.FakeRecorder{},
		}
		scaled, err := controller.reconcileOldRCs(allRcs, oldRcs, nil, deployment)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !test.scaleExpected {
			if scaled {
				t.Errorf("unexpected scaling: %v", fake.Actions())
			}
			continue
		}
		if test.scaleExpected && !scaled {
			t.Errorf("expected scaling to occur; actions: %v", fake.Actions())
			continue
		}
		// There are both list and update actions logged, so extract the update
		// action for verification.
		var updateAction testclient.UpdateAction
		for _, action := range fake.Actions() {
			switch a := action.(type) {
			case testclient.UpdateAction:
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
		updated := updateAction.GetObject().(*api.ReplicationController)
		if e, a := test.expectedOldReplicas, updated.Spec.Replicas; e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}

func rc(name string, replicas int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Template: &api.PodTemplateSpec{},
		},
	}
}

func deployment(name string, replicas int, maxSurge, maxUnavailable util.IntOrString) exp.Deployment {
	return exp.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: exp.DeploymentSpec{
			Replicas: replicas,
			Strategy: exp.DeploymentStrategy{
				Type: exp.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &exp.RollingUpdateDeployment{
					MaxSurge:       maxSurge,
					MaxUnavailable: maxUnavailable,
				},
			},
		},
	}
}
