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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
)

var (
	alwaysReady = func() bool { return true }
	noTimestamp = unversioned.Time{}
)

func rs(name string, replicas int, selector map[string]string, timestamp unversioned.Time) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			CreationTimestamp: timestamp,
			Namespace:         api.NamespaceDefault,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: selector},
			Template: api.PodTemplateSpec{},
		},
	}
}

func newRSWithStatus(name string, specReplicas, statusReplicas int, selector map[string]string) *extensions.ReplicaSet {
	rs := rs(name, specReplicas, selector, noTimestamp)
	rs.Status = extensions.ReplicaSetStatus{
		Replicas: int32(statusReplicas),
	}
	return rs
}

func newDeployment(name string, replicas int, revisionHistoryLimit *int32, maxSurge, maxUnavailable *intstr.IntOrString, selector map[string]string) *extensions.Deployment {
	d := extensions.Deployment{
		TypeMeta: unversioned.TypeMeta{APIVersion: registered.GroupOrDie(extensions.GroupName).GroupVersion.String()},
		ObjectMeta: api.ObjectMeta{
			UID:       uuid.NewUUID(),
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.DeploymentSpec{
			Strategy: extensions.DeploymentStrategy{
				Type:          extensions.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &extensions.RollingUpdateDeployment{},
			},
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: selector},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: selector,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
						},
					},
				},
			},
			RevisionHistoryLimit: revisionHistoryLimit,
		},
	}
	if maxSurge != nil {
		d.Spec.Strategy.RollingUpdate.MaxSurge = *maxSurge
	}
	if maxUnavailable != nil {
		d.Spec.Strategy.RollingUpdate.MaxUnavailable = *maxUnavailable
	}
	return &d
}

func newReplicaSet(d *extensions.Deployment, name string, replicas int) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: int32(replicas),
			Template: d.Spec.Template,
		},
	}
}

func getKey(d *extensions.Deployment, t *testing.T) string {
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
	dLister   []*extensions.Deployment
	rsLister  []*extensions.ReplicaSet
	podLister []*api.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []core.Action
	objects []runtime.Object
}

func (f *fixture) expectUpdateDeploymentAction(d *extensions.Deployment) {
	f.actions = append(f.actions, core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "deployments"}, d.Namespace, d))
}

func (f *fixture) expectUpdateDeploymentStatusAction(d *extensions.Deployment) {
	action := core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
	action.Subresource = "status"
	f.actions = append(f.actions, action)
}

func (f *fixture) expectCreateRSAction(rs *extensions.ReplicaSet) {
	f.actions = append(f.actions, core.NewCreateAction(unversioned.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = []runtime.Object{}
	return f
}

func (f *fixture) run(deploymentName string) {
	f.client = fake.NewSimpleClientset(f.objects...)
	informers := informers.NewSharedInformerFactory(f.client, controller.NoResyncPeriodFunc())
	c := NewDeploymentController(informers.Deployments(), informers.ReplicaSets(), informers.Pods(), f.client)
	c.eventRecorder = &record.FakeRecorder{}
	c.dListerSynced = alwaysReady
	c.rsListerSynced = alwaysReady
	c.podListerSynced = alwaysReady
	for _, d := range f.dLister {
		c.dLister.Indexer.Add(d)
	}
	for _, rs := range f.rsLister {
		c.rsLister.Indexer.Add(rs)
	}
	for _, pod := range f.podLister {
		c.podLister.Indexer.Add(pod)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	err := c.syncDeployment(deploymentName)
	if err != nil {
		f.t.Errorf("error syncing deployment: %v", err)
	}

	actions := filterInformerActions(f.client.Actions())
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

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	rs := newReplicaSet(d, "deploymentrs-4186632231", 1)

	f.expectCreateRSAction(rs)
	f.expectUpdateDeploymentAction(d)
	f.expectUpdateDeploymentStatusAction(d)

	f.run(getKey(d, t))
}

func TestSyncDeploymentDontDoAnythingDuringDeletion(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	now := unversioned.Now()
	d.DeletionTimestamp = &now
	f.dLister = append(f.dLister, d)

	f.run(getKey(d, t))
}

// issue: https://github.com/kubernetes/kubernetes/issues/23218
func TestDeploymentController_dontSyncDeploymentsWithEmptyPodSelector(t *testing.T) {
	fake := &fake.Clientset{}
	informers := informers.NewSharedInformerFactory(fake, controller.NoResyncPeriodFunc())
	controller := NewDeploymentController(informers.Deployments(), informers.ReplicaSets(), informers.Pods(), fake)
	controller.eventRecorder = &record.FakeRecorder{}
	controller.dListerSynced = alwaysReady
	controller.rsListerSynced = alwaysReady
	controller.podListerSynced = alwaysReady

	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	empty := unversioned.LabelSelector{}
	d.Spec.Selector = &empty
	controller.dLister.Indexer.Add(d)
	// We expect the deployment controller to not take action here since it's configuration
	// is invalid, even though no replicasets exist that match it's selector.
	controller.syncDeployment(fmt.Sprintf("%s/%s", d.ObjectMeta.Namespace, d.ObjectMeta.Name))

	filteredActions := filterInformerActions(fake.Actions())
	if len(filteredActions) == 0 {
		return
	}
	for _, action := range filteredActions {
		t.Logf("unexpected action: %#v", action)
	}
	t.Errorf("expected deployment controller to not take action")
}

func filterInformerActions(actions []core.Action) []core.Action {
	ret := []core.Action{}
	for _, action := range actions {
		if len(action.GetNamespace()) == 0 &&
			(action.Matches("list", "pods") ||
				action.Matches("list", "deployments") ||
				action.Matches("list", "replicasets") ||
				action.Matches("watch", "pods") ||
				action.Matches("watch", "deployments") ||
				action.Matches("watch", "replicasets")) {
			continue
		}
		ret = append(ret, action)
	}

	return ret
}
