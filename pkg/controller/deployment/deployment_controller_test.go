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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util"
)

var alwaysReady = func() bool { return true }

func newDeployment(replicas int) *experimental.Deployment {
	d := experimental.Deployment{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.Version()},
		ObjectMeta: api.ObjectMeta{
			UID:             util.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: experimental.DeploymentSpec{
			Strategy: experimental.DeploymentStrategy{
				Type:          experimental.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &experimental.RollingUpdateDeployment{},
			},
			Replicas: replicas,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
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

func getKey(d *experimental.Deployment, t *testing.T) string {
	if key, err := controller.KeyFunc(d); err != nil {
		t.Errorf("Unexpected error getting key for deployment %v: %v", d.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicationController(d *experimental.Deployment, name string, replicas int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Template: d.Spec.Template,
		},
	}

}

type fixture struct {
	t *testing.T

	client *testclient.Fake

	// Objects to put in the store.
	dStore   []*experimental.Deployment
	rcStore  []*api.ReplicationController
	podStore []*api.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []testclient.Action
	objects *api.List
}

func (f *fixture) expectUpdateDeploymentAction(d *experimental.Deployment) {
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
	c := NewDeploymentController(f.client)
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
