/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
)

var (
	alwaysReady = func() bool { return true }
	noTimestamp = unversioned.Time{}
)

func rs(name string, replicas int, selector map[string]string, timestamp unversioned.Time) *exp.ReplicaSet {
	return &exp.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			CreationTimestamp: timestamp,
			Namespace:         api.NamespaceDefault,
		},
		Spec: exp.ReplicaSetSpec{
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: selector},
			Template: api.PodTemplateSpec{},
		},
	}
}

func newRSWithStatus(name string, specReplicas, statusReplicas int, selector map[string]string) *exp.ReplicaSet {
	rs := rs(name, specReplicas, selector, noTimestamp)
	rs.Status = exp.ReplicaSetStatus{
		Replicas: int32(statusReplicas),
	}
	return rs
}

func deployment(name string, replicas int, maxSurge, maxUnavailable intstr.IntOrString, selector map[string]string) exp.Deployment {
	return exp.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: exp.DeploymentSpec{
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: selector},
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

func newDeployment(replicas int, revisionHistoryLimit *int) *exp.Deployment {
	var v *int32
	if revisionHistoryLimit != nil {
		v = new(int32)
		*v = int32(*revisionHistoryLimit)
	}
	d := exp.Deployment{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: exp.DeploymentSpec{
			Strategy: exp.DeploymentStrategy{
				Type:          exp.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &exp.RollingUpdateDeployment{},
			},
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
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
			RevisionHistoryLimit: v,
		},
	}
	return &d
}

// TODO: Consolidate all deployment helpers into one.
func newDeploymentEnhanced(replicas int, maxSurge intstr.IntOrString) *exp.Deployment {
	d := newDeployment(replicas, nil)
	d.Spec.Strategy.RollingUpdate.MaxSurge = maxSurge
	return d
}

func newReplicaSet(d *exp.Deployment, name string, replicas int) *exp.ReplicaSet {
	return &exp.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: exp.ReplicaSetSpec{
			Replicas: int32(replicas),
			Template: d.Spec.Template,
		},
	}
}

func getKey(d *exp.Deployment, t *testing.T) string {
	if key, err := controller.KeyFunc(d); err != nil {
		t.Errorf("Unexpected error getting key for deployment %v: %v", d.Name, err)
		return ""
	} else {
		return key
	}
}

type fixture struct {
	t *testing.T

	client *fake.Clientset
	// Objects to put in the store.
	dStore   []*exp.Deployment
	rsStore  []*exp.ReplicaSet
	podStore []*api.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []core.Action
	objects []runtime.Object
}

func (f *fixture) expectUpdateDeploymentAction(d *exp.Deployment) {
	f.actions = append(f.actions, core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "deployments"}, d.Namespace, d))
}

func (f *fixture) expectUpdateDeploymentStatusAction(d *exp.Deployment) {
	action := core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
	action.Subresource = "status"
	f.actions = append(f.actions, action)
}

func (f *fixture) expectCreateRSAction(rs *exp.ReplicaSet) {
	f.actions = append(f.actions, core.NewCreateAction(unversioned.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
}

func (f *fixture) expectUpdateRSAction(rs *exp.ReplicaSet) {
	f.actions = append(f.actions, core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
}

func (f *fixture) expectListPodAction(namespace string, opt api.ListOptions) {
	f.actions = append(f.actions, core.NewListAction(unversioned.GroupVersionResource{Resource: "pods"}, namespace, opt))
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = []runtime.Object{}
	return f
}

func (f *fixture) run(deploymentName string) {
	f.client = fake.NewSimpleClientset(f.objects...)
	c := NewDeploymentController(f.client, controller.NoResyncPeriodFunc)
	c.eventRecorder = &record.FakeRecorder{}
	c.rsStoreSynced = alwaysReady
	c.podStoreSynced = alwaysReady
	for _, d := range f.dStore {
		c.dStore.Indexer.Add(d)
	}
	for _, rs := range f.rsStore {
		c.rsStore.Store.Add(rs)
	}
	for _, pod := range f.podStore {
		c.podStore.Indexer.Add(pod)
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
		if !expectedAction.Matches(action.GetVerb(), action.GetResource().Resource) {
			f.t.Errorf("Expected\n\t%#v\ngot\n\t%#v", expectedAction, action)
			continue
		}
	}

	if len(f.actions) > len(actions) {
		f.t.Errorf("%d additional expected actions:%+v", len(f.actions)-len(actions), f.actions[len(actions):])
	}
}

func TestSyncDeploymentCreatesReplicaSet(t *testing.T) {
	f := newFixture(t)

	d := newDeployment(1, nil)
	f.dStore = append(f.dStore, d)
	f.objects = append(f.objects, d)

	rs := newReplicaSet(d, "deploymentrs-4186632231", 1)

	f.expectCreateRSAction(rs)
	f.expectUpdateDeploymentAction(d)
	f.expectUpdateDeploymentStatusAction(d)

	f.run(getKey(d, t))
}

func TestSyncDeploymentDontDoAnythingDuringDeletion(t *testing.T) {
	f := newFixture(t)

	d := newDeployment(1, nil)
	now := unversioned.Now()
	d.DeletionTimestamp = &now
	f.dStore = append(f.dStore, d)

	f.run(getKey(d, t))
}

// issue: https://github.com/kubernetes/kubernetes/issues/23218
func TestDeploymentController_dontSyncDeploymentsWithEmptyPodSelector(t *testing.T) {
	fake := &fake.Clientset{}
	controller := NewDeploymentController(fake, controller.NoResyncPeriodFunc)

	controller.eventRecorder = &record.FakeRecorder{}
	controller.rsStoreSynced = alwaysReady
	controller.podStoreSynced = alwaysReady

	d := newDeployment(1, nil)
	empty := unversioned.LabelSelector{}
	d.Spec.Selector = &empty
	controller.dStore.Indexer.Add(d)
	// We expect the deployment controller to not take action here since it's configuration
	// is invalid, even though no replicasets exist that match it's selector.
	controller.syncDeployment(fmt.Sprintf("%s/%s", d.ObjectMeta.Namespace, d.ObjectMeta.Name))
	if len(fake.Actions()) == 0 {
		return
	}
	for _, action := range fake.Actions() {
		t.Logf("unexpected action: %#v", action)
	}
	t.Errorf("expected deployment controller to not take action")
}
