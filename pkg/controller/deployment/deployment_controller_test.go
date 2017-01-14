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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
)

var (
	alwaysReady = func() bool { return true }
	noTimestamp = metav1.Time{}
)

func rs(name string, replicas int, selector map[string]string, timestamp metav1.Time) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name:              name,
			CreationTimestamp: timestamp,
			Namespace:         v1.NamespaceDefault,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Selector: &metav1.LabelSelector{MatchLabels: selector},
			Template: v1.PodTemplateSpec{},
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
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(extensions.GroupName).GroupVersion.String()},
		ObjectMeta: v1.ObjectMeta{
			UID:         uuid.NewUUID(),
			Name:        name,
			Namespace:   v1.NamespaceDefault,
			Annotations: make(map[string]string),
		},
		Spec: extensions.DeploymentSpec{
			Strategy: extensions.DeploymentStrategy{
				Type: extensions.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &extensions.RollingUpdateDeployment{
					MaxUnavailable: func() *intstr.IntOrString { i := intstr.FromInt(0); return &i }(),
					MaxSurge:       func() *intstr.IntOrString { i := intstr.FromInt(0); return &i }(),
				},
			},
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Selector: &metav1.LabelSelector{MatchLabels: selector},
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: selector,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
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
		d.Spec.Strategy.RollingUpdate.MaxSurge = maxSurge
	}
	if maxUnavailable != nil {
		d.Spec.Strategy.RollingUpdate.MaxUnavailable = maxUnavailable
	}
	return &d
}

func newReplicaSet(d *extensions.Deployment, name string, replicas int) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			Labels:    d.Spec.Selector.MatchLabels,
		},
		Spec: extensions.ReplicaSetSpec{
			Selector: d.Spec.Selector,
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
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
	podLister []*v1.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []core.Action
	objects []runtime.Object
}

func (f *fixture) expectUpdateDeploymentStatusAction(d *extensions.Deployment) {
	action := core.NewUpdateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
	action.Subresource = "status"
	f.actions = append(f.actions, action)
}

func (f *fixture) expectCreateRSAction(rs *extensions.ReplicaSet) {
	f.actions = append(f.actions, core.NewCreateAction(schema.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = []runtime.Object{}
	return f
}

func (f *fixture) newController() (*DeploymentController, informers.SharedInformerFactory) {
	f.client = fake.NewSimpleClientset(f.objects...)
	informers := informers.NewSharedInformerFactory(f.client, nil, controller.NoResyncPeriodFunc())
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
	return c, informers
}

func (f *fixture) run(deploymentName string) {
	c, informers := f.newController()
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
	f.expectUpdateDeploymentStatusAction(d)
	f.expectUpdateDeploymentStatusAction(d)

	f.run(getKey(d, t))
}

func TestSyncDeploymentDontDoAnythingDuringDeletion(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	now := metav1.Now()
	d.DeletionTimestamp = &now
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	f.expectUpdateDeploymentStatusAction(d)
	f.run(getKey(d, t))
}

// issue: https://github.com/kubernetes/kubernetes/issues/23218
func TestDeploymentController_dontSyncDeploymentsWithEmptyPodSelector(t *testing.T) {
	fake := &fake.Clientset{}
	informers := informers.NewSharedInformerFactory(fake, nil, controller.NoResyncPeriodFunc())
	controller := NewDeploymentController(informers.Deployments(), informers.ReplicaSets(), informers.Pods(), fake)
	controller.eventRecorder = &record.FakeRecorder{}
	controller.dListerSynced = alwaysReady
	controller.rsListerSynced = alwaysReady
	controller.podListerSynced = alwaysReady

	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	empty := metav1.LabelSelector{}
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

// TestOverlappingDeployment ensures that an overlapping deployment will not be synced by
// the controller.
func TestOverlappingDeployment(t *testing.T) {
	f := newFixture(t)
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.CreationTimestamp = now
	bar := newDeployment("bar", 1, nil, nil, nil, map[string]string{"foo": "bar", "app": "baz"})
	bar.CreationTimestamp = later

	f.dLister = append(f.dLister, foo, bar)
	f.objects = append(f.objects, foo, bar)

	f.expectUpdateDeploymentStatusAction(bar)
	f.run(getKey(bar, t))

	for _, a := range filterInformerActions(f.client.Actions()) {
		action, ok := a.(core.UpdateAction)
		if !ok {
			continue
		}
		d, ok := action.GetObject().(*extensions.Deployment)
		if !ok {
			continue
		}
		if d.Name == "bar" && d.Annotations[util.OverlapAnnotation] != "foo" {
			t.Errorf("annotations weren't updated for the overlapping deployment: %v", d.Annotations)
		}
	}
}

// TestSyncOverlappedDeployment ensures that from two overlapping deployments, the older
// one will be synced and the newer will be marked as overlapping. Note that in reality it's
// not always the older deployment that is the one that works vs the rest but the one which
// has the selector unchanged for longer time.
func TestSyncOverlappedDeployment(t *testing.T) {
	f := newFixture(t)
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.CreationTimestamp = now
	bar := newDeployment("bar", 1, nil, nil, nil, map[string]string{"foo": "bar", "app": "baz"})
	bar.CreationTimestamp = later

	f.dLister = append(f.dLister, foo, bar)
	f.objects = append(f.objects, foo, bar)

	f.expectUpdateDeploymentStatusAction(bar)
	f.expectCreateRSAction(newReplicaSet(foo, "foo-rs", 1))
	f.expectUpdateDeploymentStatusAction(foo)
	f.expectUpdateDeploymentStatusAction(foo)
	f.run(getKey(foo, t))

	for _, a := range filterInformerActions(f.client.Actions()) {
		action, ok := a.(core.UpdateAction)
		if !ok {
			continue
		}
		d, ok := action.GetObject().(*extensions.Deployment)
		if !ok {
			continue
		}
		if d.Name == "bar" && d.Annotations[util.OverlapAnnotation] != "foo" {
			t.Errorf("annotations weren't updated for the overlapping deployment: %v", d.Annotations)
		}
	}
}

// TestSelectorUpdate ensures that from two overlapping deployments, the one that is working won't
// be marked as overlapping if its selector is updated but still overlaps with the other one.
func TestSelectorUpdate(t *testing.T) {
	f := newFixture(t)
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}
	selectorUpdated := metav1.Time{Time: later.Add(time.Minute)}

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.CreationTimestamp = now
	foo.Annotations = map[string]string{util.SelectorUpdateAnnotation: selectorUpdated.Format(time.RFC3339)}
	bar := newDeployment("bar", 1, nil, nil, nil, map[string]string{"foo": "bar", "app": "baz"})
	bar.CreationTimestamp = later
	bar.Annotations = map[string]string{util.OverlapAnnotation: "foo"}

	f.dLister = append(f.dLister, foo, bar)
	f.objects = append(f.objects, foo, bar)

	f.expectCreateRSAction(newReplicaSet(foo, "foo-rs", 1))
	f.expectUpdateDeploymentStatusAction(foo)
	f.expectUpdateDeploymentStatusAction(foo)
	f.run(getKey(foo, t))

	for _, a := range filterInformerActions(f.client.Actions()) {
		action, ok := a.(core.UpdateAction)
		if !ok {
			continue
		}
		d, ok := action.GetObject().(*extensions.Deployment)
		if !ok {
			continue
		}

		if d.Name == "foo" && len(d.Annotations[util.OverlapAnnotation]) > 0 {
			t.Errorf("deployment %q should not have the overlapping annotation", d.Name)
		}
		if d.Name == "bar" && len(d.Annotations[util.OverlapAnnotation]) == 0 {
			t.Errorf("deployment %q should have the overlapping annotation", d.Name)
		}
	}
}

// TestDeletedDeploymentShouldCleanupOverlaps ensures that the deletion of a deployment
// will cleanup any deployments that overlap with it.
func TestDeletedDeploymentShouldCleanupOverlaps(t *testing.T) {
	f := newFixture(t)
	now := metav1.Now()
	earlier := metav1.Time{Time: now.Add(-time.Minute)}
	later := metav1.Time{Time: now.Add(time.Minute)}

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.CreationTimestamp = earlier
	foo.DeletionTimestamp = &now
	bar := newDeployment("bar", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	bar.CreationTimestamp = later
	bar.Annotations = map[string]string{util.OverlapAnnotation: "foo"}

	f.dLister = append(f.dLister, foo, bar)
	f.objects = append(f.objects, foo, bar)

	f.expectUpdateDeploymentStatusAction(bar)
	f.expectUpdateDeploymentStatusAction(foo)
	f.run(getKey(foo, t))

	for _, a := range filterInformerActions(f.client.Actions()) {
		action, ok := a.(core.UpdateAction)
		if !ok {
			continue
		}
		d := action.GetObject().(*extensions.Deployment)
		if d.Name != "bar" {
			continue
		}

		if len(d.Annotations[util.OverlapAnnotation]) > 0 {
			t.Errorf("annotations weren't cleaned up for the overlapping deployment: %v", d.Annotations)
		}
	}
}

// TestDeletedDeploymentShouldNotCleanupOtherOverlaps ensures that the deletion of
// a deployment will not cleanup deployments that overlap with another deployment.
func TestDeletedDeploymentShouldNotCleanupOtherOverlaps(t *testing.T) {
	f := newFixture(t)
	now := metav1.Now()
	earlier := metav1.Time{Time: now.Add(-time.Minute)}
	later := metav1.Time{Time: now.Add(time.Minute)}

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.CreationTimestamp = earlier
	foo.DeletionTimestamp = &now
	bar := newDeployment("bar", 1, nil, nil, nil, map[string]string{"bla": "bla"})
	bar.CreationTimestamp = later
	// Notice this deployment is overlapping with another deployment
	bar.Annotations = map[string]string{util.OverlapAnnotation: "baz"}

	f.dLister = append(f.dLister, foo, bar)
	f.objects = append(f.objects, foo, bar)

	f.expectUpdateDeploymentStatusAction(foo)
	f.run(getKey(foo, t))

	for _, a := range filterInformerActions(f.client.Actions()) {
		action, ok := a.(core.UpdateAction)
		if !ok {
			continue
		}
		d := action.GetObject().(*extensions.Deployment)
		if d.Name != "bar" {
			continue
		}

		if len(d.Annotations[util.OverlapAnnotation]) == 0 {
			t.Errorf("overlapping annotation should not be cleaned up for bar: %v", d.Annotations)
		}
	}
}

// TestPodDeletionEnqueuesRecreateDeployment ensures that the deletion of a pod
// will requeue a Recreate deployment iff there is no other pod returned from the
// client.
func TestPodDeletionEnqueuesRecreateDeployment(t *testing.T) {
	f := newFixture(t)

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.Spec.Strategy.Type = extensions.RecreateDeploymentStrategyType
	rs := newReplicaSet(foo, "foo-1", 1)
	pod := generatePodFromRS(rs)

	f.dLister = append(f.dLister, foo)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, foo, rs)

	c, informers := f.newController()
	enqueued := false
	c.enqueueDeployment = func(d *extensions.Deployment) {
		if d.Name == "foo" {
			enqueued = true
		}
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	c.deletePod(pod)

	if !enqueued {
		t.Errorf("expected deployment %q to be queued after pod deletion", foo.Name)
	}
}

// TestPodDeletionDoesntEnqueueRecreateDeployment ensures that the deletion of a pod
// will not requeue a Recreate deployment iff there are other pods returned from the
// client.
func TestPodDeletionDoesntEnqueueRecreateDeployment(t *testing.T) {
	f := newFixture(t)

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.Spec.Strategy.Type = extensions.RecreateDeploymentStrategyType
	rs := newReplicaSet(foo, "foo-1", 1)
	pod := generatePodFromRS(rs)

	f.dLister = append(f.dLister, foo)
	f.rsLister = append(f.rsLister, rs)
	// Let's pretend this is a different pod. The gist is that the pod lister needs to
	// return a non-empty list.
	f.podLister = append(f.podLister, pod)

	c, informers := f.newController()
	enqueued := false
	c.enqueueDeployment = func(d *extensions.Deployment) {
		if d.Name == "foo" {
			enqueued = true
		}
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	c.deletePod(pod)

	if enqueued {
		t.Errorf("expected deployment %q not to be queued after pod deletion", foo.Name)
	}
}

// generatePodFromRS creates a pod, with the input ReplicaSet's selector and its template
func generatePodFromRS(rs *extensions.ReplicaSet) *v1.Pod {
	trueVar := true
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name:      rs.Name + "-pod",
			Namespace: rs.Namespace,
			Labels:    rs.Spec.Selector.MatchLabels,
			OwnerReferences: []metav1.OwnerReference{
				{UID: rs.UID, APIVersion: "v1beta1", Kind: "ReplicaSet", Name: rs.Name, Controller: &trueVar},
			},
		},
		Spec: rs.Spec.Template.Spec,
	}
}
