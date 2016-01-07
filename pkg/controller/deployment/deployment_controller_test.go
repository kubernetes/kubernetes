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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	exp "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestDeploymentController_reconcileNewRC(t *testing.T) {
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
		newRc := rc("foo-v2", test.newReplicas)
		oldRc := rc("foo-v2", test.oldReplicas)
		allRcs := []*api.ReplicationController{newRc, oldRc}
		deployment := deployment("foo", test.deploymentReplicas, test.maxSurge, intstr.FromInt(0))
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
		maxUnavailable      intstr.IntOrString
		readyPods           int
		oldReplicas         int
		scaleExpected       bool
		expectedOldReplicas int
	}{
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(0),
			readyPods:          10,
			oldReplicas:        10,
			scaleExpected:      false,
		},
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(2),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 8,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			readyPods:          8,
			oldReplicas:        10,
			scaleExpected:      false,
		},
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
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
		deployment := deployment("foo", test.deploymentReplicas, intstr.FromInt(0), test.maxUnavailable)
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
		scaled, err := controller.reconcileOldRCs(allRcs, oldRcs, nil, deployment, false)
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

func deployment(name string, replicas int, maxSurge, maxUnavailable intstr.IntOrString) exp.Deployment {
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

var alwaysReady = func() bool { return true }

func newDeployment(replicas int) *exp.Deployment {
	d := exp.Deployment{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             util.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: exp.DeploymentSpec{
			Strategy: exp.DeploymentStrategy{
				Type:          exp.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &exp.RollingUpdateDeployment{},
			},
			Replicas: replicas,
			Selector: map[string]string{"foo": "bar"},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
						},
					},
				},
			},
		},
	}
	return &d
}

func getKey(d *exp.Deployment, t *testing.T) string {
	if key, err := controller.KeyFunc(d); err != nil {
		t.Errorf("Unexpected error getting key for deployment %v: %v", d.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicationController(d *exp.Deployment, name string, replicas int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Template: &d.Spec.Template,
		},
	}

}

type fixture struct {
	t *testing.T

	client *testclient.Fake

	// Objects to put in the store.
	dStore   []*exp.Deployment
	rcStore  []*api.ReplicationController
	podStore []*api.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []testclient.Action
	objects *api.List
}

func (f *fixture) expectUpdateDeploymentAction(d *exp.Deployment) {
	f.actions = append(f.actions, testclient.NewUpdateAction("deployments", d.Namespace, d))
	f.objects.Items = append(f.objects.Items, d)
}

func (f *fixture) expectCreateRCAction(rc *api.ReplicationController) {
	f.actions = append(f.actions, testclient.NewCreateAction("replicationcontrollers", rc.Namespace, rc))
	f.objects.Items = append(f.objects.Items, rc)
}

func (f *fixture) expectUpdateRCAction(rc *api.ReplicationController) {
	f.actions = append(f.actions, testclient.NewUpdateAction("replicationcontrollers", rc.Namespace, rc))
	f.objects.Items = append(f.objects.Items, rc)
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = &api.List{}
	return f
}

func (f *fixture) run(deploymentName string) {
	f.client = testclient.NewSimpleFake(f.objects)
	c := NewDeploymentController(f.client, controller.NoResyncPeriodFunc)
	c.eventRecorder = &record.FakeRecorder{}
	c.rcStoreSynced = alwaysReady
	c.podStoreSynced = alwaysReady
	for _, d := range f.dStore {
		c.dStore.Store.Add(d)
	}
	for _, rc := range f.rcStore {
		c.rcStore.Store.Add(rc)
	}
	for _, pod := range f.podStore {
		c.podStore.Store.Add(pod)
	}

	err := c.syncDeployment(deploymentName)
	if err != nil {
		f.t.Errorf("error syncing deployment: %v", err)
	}

	actions := f.client.Actions()
	for i, action := range actions {
		if len(f.actions) < i+1 {
			f.t.Errorf("%d unexpected actions: %+v", len(actions)-len(f.actions), actions[i:])
			break
		}

		expectedAction := f.actions[i]
		if !expectedAction.Matches(action.GetVerb(), action.GetResource()) {
			f.t.Errorf("Expected\n\t%#v\ngot\n\t%#v", expectedAction, action)
			continue
		}
	}

	if len(f.actions) > len(actions) {
		f.t.Errorf("%d additional expected actions:%+v", len(f.actions)-len(actions), f.actions[len(actions):])
	}
}

func TestSyncDeploymentCreatesRC(t *testing.T) {
	f := newFixture(t)

	d := newDeployment(1)
	f.dStore = append(f.dStore, d)

	// expect that one rc with zero replicas is created
	// then is updated to 1 replica
	rc := newReplicationController(d, "deploymentrc-4186632231", 0)
	updatedRC := newReplicationController(d, "deploymentrc-4186632231", 1)

	f.expectCreateRCAction(rc)
	f.expectUpdateRCAction(updatedRC)
	f.expectUpdateDeploymentAction(d)

	f.run(getKey(d, t))
}
