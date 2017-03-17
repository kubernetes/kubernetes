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
	"strconv"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

var (
	alwaysReady = func() bool { return true }
	noTimestamp = metav1.Time{}
)

func rs(name string, replicas int, selector map[string]string, timestamp metav1.Time) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: timestamp,
			Namespace:         metav1.NamespaceDefault,
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
		ObjectMeta: metav1.ObjectMeta{
			UID:         uuid.NewUUID(),
			Name:        name,
			Namespace:   metav1.NamespaceDefault,
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
				ObjectMeta: metav1.ObjectMeta{
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
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			UID:             uuid.NewUUID(),
			Namespace:       metav1.NamespaceDefault,
			Labels:          d.Spec.Selector.MatchLabels,
			OwnerReferences: []metav1.OwnerReference{*newControllerRef(d)},
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

func (f *fixture) expectUpdateDeploymentAction(d *extensions.Deployment) {
	action := core.NewUpdateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
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
	informers := informers.NewSharedInformerFactory(f.client, controller.NoResyncPeriodFunc())
	c := NewDeploymentController(informers.Extensions().V1beta1().Deployments(), informers.Extensions().V1beta1().ReplicaSets(), informers.Core().V1().Pods(), f.client)
	c.eventRecorder = &record.FakeRecorder{}
	c.dListerSynced = alwaysReady
	c.rsListerSynced = alwaysReady
	c.podListerSynced = alwaysReady
	for _, d := range f.dLister {
		informers.Extensions().V1beta1().Deployments().Informer().GetIndexer().Add(d)
	}
	for _, rs := range f.rsLister {
		informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	}
	for _, pod := range f.podLister {
		informers.Core().V1().Pods().Informer().GetIndexer().Add(pod)
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

func TestSyncDeploymentClearsOverlapAnnotation(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d.Annotations[util.OverlapAnnotation] = "overlap"
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	rs := newReplicaSet(d, "deploymentrs-4186632231", 1)

	f.expectUpdateDeploymentStatusAction(d)
	f.expectCreateRSAction(rs)
	f.expectUpdateDeploymentStatusAction(d)
	f.expectUpdateDeploymentStatusAction(d)

	f.run(getKey(d, t))

	d, err := f.client.ExtensionsV1beta1().Deployments(d.Namespace).Get(d.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("can't get deployment: %v", err)
	}
	if _, ok := d.Annotations[util.OverlapAnnotation]; ok {
		t.Errorf("OverlapAnnotation = %q, wanted absent", d.Annotations[util.OverlapAnnotation])
	}
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
func TestDontSyncDeploymentsWithEmptyPodSelector(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d.Spec.Selector = &metav1.LabelSelector{}
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	// Normally there should be a status update to sync observedGeneration but the fake
	// deployment has no generation set so there is no action happpening here.
	f.run(getKey(d, t))
}

func TestReentrantRollback(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	d.Spec.RollbackTo = &extensions.RollbackConfig{Revision: 0}
	d.Annotations = map[string]string{util.RevisionAnnotation: "2"}
	f.dLister = append(f.dLister, d)

	rs1 := newReplicaSet(d, "deploymentrs-old", 0)
	rs1.Annotations = map[string]string{util.RevisionAnnotation: "1"}
	one := int64(1)
	rs1.Spec.Template.Spec.TerminationGracePeriodSeconds = &one
	rs1.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey] = "hash"

	rs2 := newReplicaSet(d, "deploymentrs-new", 1)
	rs2.Annotations = map[string]string{util.RevisionAnnotation: "2"}
	rs2.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey] = "hash"

	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, d, rs1, rs2)

	// Rollback is done here
	f.expectUpdateDeploymentAction(d)
	// Expect no update on replica sets though
	f.run(getKey(d, t))
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

	c, _ := f.newController()
	enqueued := false
	c.enqueueDeployment = func(d *extensions.Deployment) {
		if d.Name == "foo" {
			enqueued = true
		}
	}

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

	c, _ := f.newController()
	enqueued := false
	c.enqueueDeployment = func(d *extensions.Deployment) {
		if d.Name == "foo" {
			enqueued = true
		}
	}

	c.deletePod(pod)

	if enqueued {
		t.Errorf("expected deployment %q not to be queued after pod deletion", foo.Name)
	}
}

func TestGetReplicaSetsForDeployment(t *testing.T) {
	f := newFixture(t)

	// Two Deployments with same labels.
	d1 := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("bar", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// Two ReplicaSets that match labels for both Deployments,
	// but have ControllerRefs to make ownership explicit.
	rs1 := newReplicaSet(d1, "rs1", 1)
	rs2 := newReplicaSet(d2, "rs2", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, d1, d2, rs1, rs2)

	// Start the fixture.
	c, informers := f.newController()
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	rsList, err := c.getReplicaSetsForDeployment(d1)
	if err != nil {
		t.Fatalf("getReplicaSetsForDeployment() error: %v", err)
	}
	rsNames := []string{}
	for _, rs := range rsList {
		rsNames = append(rsNames, rs.Name)
	}
	if len(rsNames) != 1 || rsNames[0] != rs1.Name {
		t.Errorf("getReplicaSetsForDeployment() = %v, want [%v]", rsNames, rs1.Name)
	}

	rsList, err = c.getReplicaSetsForDeployment(d2)
	if err != nil {
		t.Fatalf("getReplicaSetsForDeployment() error: %v", err)
	}
	rsNames = []string{}
	for _, rs := range rsList {
		rsNames = append(rsNames, rs.Name)
	}
	if len(rsNames) != 1 || rsNames[0] != rs2.Name {
		t.Errorf("getReplicaSetsForDeployment() = %v, want [%v]", rsNames, rs2.Name)
	}
}

func TestGetReplicaSetsForDeploymentAdoptRelease(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// RS with matching labels, but orphaned. Should be adopted and returned.
	rsAdopt := newReplicaSet(d, "rsAdopt", 1)
	rsAdopt.OwnerReferences = nil
	// RS with matching ControllerRef, but wrong labels. Should be released.
	rsRelease := newReplicaSet(d, "rsRelease", 1)
	rsRelease.Labels = map[string]string{"foo": "notbar"}

	f.dLister = append(f.dLister, d)
	f.rsLister = append(f.rsLister, rsAdopt, rsRelease)
	f.objects = append(f.objects, d, rsAdopt, rsRelease)

	// Start the fixture.
	c, informers := f.newController()
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	rsList, err := c.getReplicaSetsForDeployment(d)
	if err != nil {
		t.Fatalf("getReplicaSetsForDeployment() error: %v", err)
	}
	rsNames := []string{}
	for _, rs := range rsList {
		rsNames = append(rsNames, rs.Name)
	}
	if len(rsNames) != 1 || rsNames[0] != rsAdopt.Name {
		t.Errorf("getReplicaSetsForDeployment() = %v, want [%v]", rsNames, rsAdopt.Name)
	}
}

func TestGetPodMapForReplicaSets(t *testing.T) {
	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	rs1 := newReplicaSet(d, "rs1", 1)
	rs2 := newReplicaSet(d, "rs2", 1)

	// Add a Pod for each ReplicaSet.
	pod1 := generatePodFromRS(rs1)
	pod2 := generatePodFromRS(rs2)
	// Add a Pod that has matching labels, but no ControllerRef.
	pod3 := generatePodFromRS(rs1)
	pod3.Name = "pod3"
	pod3.OwnerReferences = nil
	// Add a Pod that has matching labels and ControllerRef, but is inactive.
	pod4 := generatePodFromRS(rs1)
	pod4.Name = "pod4"
	pod4.Status.Phase = v1.PodFailed

	f.dLister = append(f.dLister, d)
	f.rsLister = append(f.rsLister, rs1, rs2)
	f.podLister = append(f.podLister, pod1, pod2, pod3, pod4)
	f.objects = append(f.objects, d, rs1, rs2, pod1, pod2, pod3, pod4)

	// Start the fixture.
	c, informers := f.newController()
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)

	podMap, err := c.getPodMapForDeployment(d, f.rsLister)
	if err != nil {
		t.Fatalf("getPodMapForDeployment() error: %v", err)
	}
	podCount := 0
	for _, podList := range podMap {
		podCount += len(podList.Items)
	}
	if got, want := podCount, 2; got != want {
		t.Errorf("podCount = %v, want %v", got, want)
	}

	if got, want := len(podMap), 2; got != want {
		t.Errorf("len(podMap) = %v, want %v", got, want)
	}
	if got, want := len(podMap[rs1.UID].Items), 1; got != want {
		t.Errorf("len(podMap[rs1]) = %v, want %v", got, want)
	}
	if got, want := podMap[rs1.UID].Items[0].Name, "rs1-pod"; got != want {
		t.Errorf("podMap[rs1] = [%v], want [%v]", got, want)
	}
	if got, want := len(podMap[rs2.UID].Items), 1; got != want {
		t.Errorf("len(podMap[rs2]) = %v, want %v", got, want)
	}
	if got, want := podMap[rs2.UID].Items[0].Name, "rs2-pod"; got != want {
		t.Errorf("podMap[rs2] = [%v], want [%v]", got, want)
	}
}

func TestAddReplicaSet(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// Two ReplicaSets that match labels for both Deployments,
	// but have ControllerRefs to make ownership explicit.
	rs1 := newReplicaSet(d1, "rs1", 1)
	rs2 := newReplicaSet(d2, "rs2", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.objects = append(f.objects, d1, d2, rs1, rs2)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	dc.addReplicaSet(rs1)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	dc.addReplicaSet(rs2)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestAddReplicaSetOrphan(t *testing.T) {
	f := newFixture(t)

	// 2 will match the RS, 1 won't.
	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d3 := newDeployment("d3", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d3.Spec.Selector.MatchLabels = map[string]string{"foo": "notbar"}

	// Make the RS an orphan. Expect matching Deployments to be queued.
	rs := newReplicaSet(d1, "rs1", 1)
	rs.OwnerReferences = nil

	f.dLister = append(f.dLister, d1, d2, d3)
	f.objects = append(f.objects, d1, d2, d3)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	dc.addReplicaSet(rs)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSet(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// Two ReplicaSets that match labels for both Deployments,
	// but have ControllerRefs to make ownership explicit.
	rs1 := newReplicaSet(d1, "rs1", 1)
	rs2 := newReplicaSet(d2, "rs2", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, d1, d2, rs1, rs2)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	prev := *rs1
	next := *rs1
	bumpResourceVersion(&next)
	dc.updateReplicaSet(&prev, &next)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	prev = *rs2
	next = *rs2
	bumpResourceVersion(&next)
	dc.updateReplicaSet(&prev, &next)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetOrphanWithNewLabels(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// RS matches both, but is an orphan.
	rs := newReplicaSet(d1, "rs1", 1)
	rs.OwnerReferences = nil

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	// Change labels and expect all matching controllers to queue.
	prev := *rs
	prev.Labels = map[string]string{"foo": "notbar"}
	next := *rs
	bumpResourceVersion(&next)
	dc.updateReplicaSet(&prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetChangeControllerRef(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	rs := newReplicaSet(d1, "rs1", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	// Change ControllerRef and expect both old and new to queue.
	prev := *rs
	prev.OwnerReferences = []metav1.OwnerReference{*newControllerRef(d2)}
	next := *rs
	bumpResourceVersion(&next)
	dc.updateReplicaSet(&prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetRelease(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	rs := newReplicaSet(d1, "rs1", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	// Remove ControllerRef and expect all matching controller to sync orphan.
	prev := *rs
	next := *rs
	next.OwnerReferences = nil
	bumpResourceVersion(&next)
	dc.updateReplicaSet(&prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestDeleteReplicaSet(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// Two ReplicaSets that match labels for both Deployments,
	// but have ControllerRefs to make ownership explicit.
	rs1 := newReplicaSet(d1, "rs1", 1)
	rs2 := newReplicaSet(d2, "rs2", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, d1, d2, rs1, rs2)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	dc.deleteReplicaSet(rs1)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	dc.deleteReplicaSet(rs2)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestDeleteReplicaSetOrphan(t *testing.T) {
	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	// Make the RS an orphan. Expect matching Deployments to be queued.
	rs := newReplicaSet(d1, "rs1", 1)
	rs.OwnerReferences = nil

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _ := f.newController()

	dc.deleteReplicaSet(rs)
	if got, want := dc.queue.Len(), 0; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func bumpResourceVersion(obj metav1.Object) {
	ver, _ := strconv.ParseInt(obj.GetResourceVersion(), 10, 32)
	obj.SetResourceVersion(strconv.FormatInt(ver+1, 10))
}

// generatePodFromRS creates a pod, with the input ReplicaSet's selector and its template
func generatePodFromRS(rs *extensions.ReplicaSet) *v1.Pod {
	trueVar := true
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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
