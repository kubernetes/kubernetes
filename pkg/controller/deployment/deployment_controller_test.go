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
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/policy/install"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/utils/ptr"
)

var (
	alwaysReady = func() bool { return true }
	noTimestamp = metav1.Time{}
)

func rs(name string, replicas int32, selector map[string]string, timestamp metav1.Time) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: timestamp,
			Namespace:         metav1.NamespaceDefault,
		},
		Spec: apps.ReplicaSetSpec{
			Replicas: ptr.To(replicas),
			Selector: &metav1.LabelSelector{MatchLabels: selector},
			Template: v1.PodTemplateSpec{},
		},
	}
}

func newRSWithStatus(name string, specReplicas, statusReplicas int32, selector map[string]string) *apps.ReplicaSet {
	rs := rs(name, specReplicas, selector, noTimestamp)
	rs.Status = apps.ReplicaSetStatus{
		Replicas: statusReplicas,
	}
	return rs
}

func newDeployment(name string, replicas int32, revisionHistoryLimit *int32, maxSurge, maxUnavailable *intstr.IntOrString, selector map[string]string) *apps.Deployment {
	d := apps.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			UID:         uuid.NewUUID(),
			Name:        name,
			Namespace:   metav1.NamespaceDefault,
			Annotations: make(map[string]string),
		},
		Spec: apps.DeploymentSpec{
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxUnavailable: ptr.To(intstr.FromInt32(0)),
					MaxSurge:       ptr.To(intstr.FromInt32(0)),
				},
			},
			Replicas: ptr.To(replicas),
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

func newReplicaSet(d *apps.Deployment, name string, replicas int32) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		TypeMeta: metav1.TypeMeta{Kind: "ReplicaSet"},
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			UID:             uuid.NewUUID(),
			Namespace:       metav1.NamespaceDefault,
			Labels:          d.Spec.Selector.MatchLabels,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(d, controllerKind)},
		},
		Spec: apps.ReplicaSetSpec{
			Selector: d.Spec.Selector,
			Replicas: ptr.To(replicas),
			Template: d.Spec.Template,
		},
	}
}

type fixture struct {
	t testing.TB

	client *fake.Clientset
	// Objects to put in the store.
	dLister   []*apps.Deployment
	rsLister  []*apps.ReplicaSet
	podLister []*v1.Pod

	// Actions expected to happen on the client. Objects from here are also
	// preloaded into NewSimpleFake.
	actions []core.Action
	objects []runtime.Object
}

func (f *fixture) expectGetDeploymentAction(d *apps.Deployment) {
	action := core.NewGetAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d.Name)
	f.actions = append(f.actions, action)
}

func (f *fixture) expectUpdateDeploymentStatusAction(d *apps.Deployment) {
	action := core.NewUpdateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
	action.Subresource = "status"
	f.actions = append(f.actions, action)
}

func (f *fixture) expectUpdateDeploymentAction(d *apps.Deployment) {
	action := core.NewUpdateAction(schema.GroupVersionResource{Resource: "deployments"}, d.Namespace, d)
	f.actions = append(f.actions, action)
}

func (f *fixture) expectCreateRSAction(rs *apps.ReplicaSet) {
	f.actions = append(f.actions, core.NewCreateAction(schema.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
}

func newFixture(t testing.TB) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = []runtime.Object{}
	return f
}

func (f *fixture) newController(ctx context.Context) (*DeploymentController, informers.SharedInformerFactory, error) {
	f.client = fake.NewSimpleClientset(f.objects...)
	informers := informers.NewSharedInformerFactory(f.client, controller.NoResyncPeriodFunc())
	c, err := NewDeploymentController(ctx, informers.Apps().V1().Deployments(), informers.Apps().V1().ReplicaSets(), informers.Core().V1().Pods(), informers.Scheduling().V1beta1().Workloads(), f.client)
	if err != nil {
		return nil, nil, err
	}
	c.eventRecorder = &record.FakeRecorder{}
	c.dListerSynced = alwaysReady
	c.rsListerSynced = alwaysReady
	c.podListerSynced = alwaysReady
	for _, d := range f.dLister {
		informers.Apps().V1().Deployments().Informer().GetIndexer().Add(d)
	}
	for _, rs := range f.rsLister {
		informers.Apps().V1().ReplicaSets().Informer().GetIndexer().Add(rs)
	}
	for _, pod := range f.podLister {
		informers.Core().V1().Pods().Informer().GetIndexer().Add(pod)
	}
	return c, informers, nil
}

func (f *fixture) runExpectError(ctx context.Context, deploymentName string, startInformers bool) {
	f.run_(ctx, deploymentName, startInformers, true)
}

func (f *fixture) run(ctx context.Context, deploymentName string) {
	f.run_(ctx, deploymentName, true, false)
}

func (f *fixture) run_(ctx context.Context, deploymentName string, startInformers bool, expectError bool) {
	c, informers, err := f.newController(ctx)
	if err != nil {
		f.t.Fatalf("error creating Deployment controller: %v", err)
	}
	if startInformers {
		stopCh := make(chan struct{})
		defer close(stopCh)
		informers.StartWithContext(ctx)
	}

	err = c.syncDeployment(ctx, deploymentName)
	if !expectError && err != nil {
		f.t.Errorf("error syncing deployment: %v", err)
	} else if expectError && err == nil {
		f.t.Error("expected error syncing deployment, got nil")
	}

	actions := filterInformerActions(f.client.Actions())
	for i, action := range actions {
		if len(f.actions) < i+1 {
			f.t.Errorf("%d unexpected actions: %+v", len(actions)-len(f.actions), actions[i:])
			break
		}

		expectedAction := f.actions[i]
		if !(expectedAction.Matches(action.GetVerb(), action.GetResource().Resource) && action.GetSubresource() == expectedAction.GetSubresource()) {
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

func TestSyncDeploymentAnnotationUpdateError(t *testing.T) {
	statusErr := errors.New("deployment status update failed")
	for _, tc := range []struct {
		name       string
		strategy   apps.DeploymentStrategyType
		state      string
		statusErr  error
		wantStatus v1.ConditionStatus
		wantReason string
		wantWrites int
	}{
		{name: "rolling update times out", wantStatus: v1.ConditionFalse, wantReason: util.TimedOutReason, wantWrites: 1},
		{name: "recreate times out", strategy: apps.RecreateDeploymentStrategyType, wantStatus: v1.ConditionFalse, wantReason: util.TimedOutReason, wantWrites: 1},
		{name: "both update errors are returned", statusErr: statusErr, wantStatus: v1.ConditionFalse, wantReason: util.TimedOutReason, wantWrites: 1},
		{name: "before deadline", state: "recent", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason, wantWrites: 1},
		{name: "new availability is progress", state: "progress", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason, wantWrites: 1},
		{name: "completed rollout stays complete", state: "complete", wantStatus: v1.ConditionTrue, wantReason: util.NewRSAvailableReason, wantWrites: 1},
		{name: "pending minReadySeconds cannot establish completion", state: "pending minReadySeconds", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
		{name: "stale new ReplicaSet status cannot establish completion", state: "stale new status", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
		{name: "stale old ReplicaSet status cannot establish completion", state: "stale old status", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
		{name: "stale status cannot advance progress", state: "stale progress", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
		{name: "unobserved ReplicaSet status is not used", state: "unobserved status", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
		{name: "deadline resumes after status is observed", state: "stale then observed", wantStatus: v1.ConditionTrue, wantReason: util.ReplicaSetUpdatedReason},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			d := newDeployment("annotation-sync", 3, nil, ptr.To(intstr.FromInt32(1)), ptr.To(intstr.FromInt32(0)), map[string]string{"app": "annotation-sync"})
			d.Generation = 2
			d.Annotations[util.RevisionAnnotation] = "2"
			d.Spec.ProgressDeadlineSeconds = ptr.To[int32](600)
			newRS := newReplicaSet(d, "annotation-sync-new", 1)
			newRS.Generation = 1
			newRS.Annotations = map[string]string{util.RevisionAnnotation: "2", util.DesiredReplicasAnnotation: "3", util.MaxReplicasAnnotation: "4"}
			newRS.Status.Replicas = 1
			newRS.Status.ObservedGeneration = newRS.Generation
			oldRS := newReplicaSet(d, "annotation-sync-old", 3)
			oldRS.Generation = 1
			oldRS.Annotations = map[string]string{util.RevisionAnnotation: "1", util.DesiredReplicasAnnotation: "3", util.MaxReplicasAnnotation: "4"}
			oldRS.Spec.Template = *d.Spec.Template.DeepCopy()
			oldRS.Spec.Template.Spec.Containers[0].Image = "foo/old"
			oldRS.Status = apps.ReplicaSetStatus{Replicas: 3, ReadyReplicas: 3, AvailableReplicas: 3, ObservedGeneration: oldRS.Generation}
			if tc.strategy == apps.RecreateDeploymentStrategyType {
				d.Spec.Strategy = apps.DeploymentStrategy{Type: tc.strategy}
				oldRS.Spec.Replicas = ptr.To[int32](0)
				oldRS.Status = apps.ReplicaSetStatus{ObservedGeneration: oldRS.Generation}
				oldRS.Annotations[util.MaxReplicasAnnotation] = "3"
				newRS.Annotations[util.MaxReplicasAnnotation] = "3"
			}
			if tc.state == "complete" || tc.state == "pending minReadySeconds" || tc.state == "stale new status" || tc.state == "stale old status" {
				oldRS.Spec.Replicas = ptr.To[int32](0)
				oldRS.Status = apps.ReplicaSetStatus{ObservedGeneration: oldRS.Generation}
				newRS.Spec.Replicas = ptr.To[int32](3)
				newRS.Status = apps.ReplicaSetStatus{Replicas: 3, ReadyReplicas: 3, AvailableReplicas: 3, ObservedGeneration: newRS.Generation}
				if tc.state != "complete" {
					newRS.Status.ReadyReplicas = 2
					newRS.Status.AvailableReplicas = 2
				}
			}
			allRSs := []*apps.ReplicaSet{oldRS, newRS}
			d.Status = calculateStatus(allRSs, newRS, d)
			condition := util.NewDeploymentCondition(apps.DeploymentProgressing, v1.ConditionTrue, util.ReplicaSetUpdatedReason, "Deployment is progressing")
			condition.LastUpdateTime = metav1.NewTime(time.Now().Add(-time.Hour))
			if tc.state == "recent" {
				condition.LastUpdateTime = metav1.Now()
			}
			if tc.state == "complete" {
				condition.Reason = util.NewRSAvailableReason
			}
			condition.LastTransitionTime = condition.LastUpdateTime
			util.SetDeploymentCondition(&d.Status, *condition)
			if tc.state == "progress" || tc.state == "stale progress" {
				newRS.Status.ReadyReplicas = 1
				newRS.Status.AvailableReplicas = 1
			}
			if tc.state == "pending minReadySeconds" {
				d.Spec.MinReadySeconds = 10
				newRS.Status.ReadyReplicas = 3
				newRS.Status.AvailableReplicas = 3
			}
			if tc.state == "stale new status" {
				d.Spec.MinReadySeconds = 10
				newRS.Spec.MinReadySeconds = 10
				newRS.Generation++
				newRS.Status.ReadyReplicas = 3
				newRS.Status.AvailableReplicas = 3
			}
			if tc.state == "stale old status" {
				oldRS.Spec.Replicas = ptr.To[int32](1)
				oldRS.Generation++
				newRS.Status.ReadyReplicas = 3
				newRS.Status.AvailableReplicas = 3
			}
			if tc.state == "stale progress" || tc.state == "stale then observed" {
				newRS.Generation++
			}
			if tc.state == "unobserved status" {
				newRS.Status.ObservedGeneration = 0
			}
			d.Annotations["example.com/change-cause"] = "updated"
			d.Generation++

			f := newFixture(t)
			f.dLister = []*apps.Deployment{d}
			f.rsLister = allRSs
			f.objects = []runtime.Object{d, oldRS, newRS}
			dc, informerSet, err := f.newController(ctx)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(dc.queue.ShutDown)
			rsErr := apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "replicasets"}, newRS.Name, errors.New("annotation update rejected"))
			var attempts int
			f.client.PrependReactor("update", "replicasets", func(action core.Action) (bool, runtime.Object, error) {
				attempts++
				updated := action.(core.UpdateAction).GetObject().(*apps.ReplicaSet)
				if updated.Name != newRS.Name || *updated.Spec.Replicas != *newRS.Spec.Replicas || updated.Annotations["example.com/change-cause"] != "updated" {
					t.Errorf("expected annotation synchronization without scaling, got %+v", updated)
				}
				return true, nil, rsErr
			})
			var statusWrites []*apps.Deployment
			f.client.PrependReactor("update", "deployments", func(action core.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "status" {
					t.Fatalf("unexpected Deployment update: %v", action)
				}
				statusWrites = append(statusWrites, action.(core.UpdateAction).GetObject().(*apps.Deployment).DeepCopy())
				return tc.statusErr != nil, nil, tc.statusErr
			})
			err = dc.syncDeployment(ctx, d.Namespace+"/"+d.Name)
			if !errors.Is(err, rsErr) || (tc.statusErr != nil && !errors.Is(err, tc.statusErr)) {
				t.Errorf("update error was lost: %v", err)
			}
			if tc.statusErr == nil && err != rsErr {
				t.Errorf("original API error identity changed: %v", err)
			}
			if attempts != 1 || len(statusWrites) != tc.wantWrites {
				t.Errorf("ReplicaSet updates=%d, status updates=%d; want 1, %d", attempts, len(statusWrites), tc.wantWrites)
			}
			got := d
			if len(statusWrites) != 0 {
				got = statusWrites[len(statusWrites)-1]
				if got.Status.ObservedGeneration != d.Generation {
					t.Errorf("observed generation = %d, want %d", got.Status.ObservedGeneration, d.Generation)
				}
			}
			progressing := util.GetDeploymentCondition(got.Status, apps.DeploymentProgressing)
			if progressing == nil || progressing.Status != tc.wantStatus || progressing.Reason != tc.wantReason {
				t.Fatalf("Progressing = %+v, want %s/%s", progressing, tc.wantStatus, tc.wantReason)
			}
			if tc.state == "progress" {
				if !progressing.LastUpdateTime.After(condition.LastUpdateTime.Time) {
					t.Error("new availability did not advance the progress timestamp")
				}
			} else if tc.wantStatus == v1.ConditionTrue && !progressing.LastUpdateTime.Equal(&condition.LastUpdateTime) {
				t.Error("annotation synchronization changed the progress timestamp without new progress")
			}
			if newRS.Annotations["example.com/change-cause"] != "" {
				t.Error("cached ReplicaSet was mutated")
			}
			if tc.state == "stale new status" || tc.state == "stale then observed" {
				observed := newRS.DeepCopy()
				observed.Status.ObservedGeneration = observed.Generation
				if err := f.client.Tracker().Update(apps.SchemeGroupVersion.WithResource("replicasets"), observed, observed.Namespace); err != nil {
					t.Fatal(err)
				}
				if err := informerSet.Apps().V1().ReplicaSets().Informer().GetIndexer().Update(observed); err != nil {
					t.Fatal(err)
				}
				if err := dc.syncDeployment(ctx, d.Namespace+"/"+d.Name); err != rsErr {
					t.Fatalf("error after observing status = %v, want original ReplicaSet error", err)
				}
				if attempts != 2 || len(statusWrites) != 1 {
					t.Fatalf("after observing status: ReplicaSet updates=%d, status updates=%d; want 2, 1", attempts, len(statusWrites))
				}
				updated := statusWrites[0]
				wantReason, wantStatus := util.TimedOutReason, v1.ConditionFalse
				if tc.state == "stale new status" {
					wantReason, wantStatus = util.NewRSAvailableReason, v1.ConditionTrue
				}
				condition := util.GetDeploymentCondition(updated.Status, apps.DeploymentProgressing)
				if condition == nil || condition.Reason != wantReason || condition.Status != wantStatus || updated.Status.ObservedGeneration != d.Generation {
					t.Fatalf("status reconciliation did not resume: %+v", updated.Status)
				}
			}
		})
	}
}

func TestSyncDeploymentCreatesReplicaSet(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	rs := newReplicaSet(d, "deploymentrs-4186632231", 1)

	f.expectCreateRSAction(rs)
	f.expectUpdateDeploymentStatusAction(d)
	f.expectUpdateDeploymentStatusAction(d)

	f.run(ctx, testutil.GetKey(d, t))
}

func TestSyncDeploymentDontDoAnythingDuringDeletion(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	now := metav1.Now()
	d.DeletionTimestamp = &now
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	f.expectUpdateDeploymentStatusAction(d)
	f.run(ctx, testutil.GetKey(d, t))
}

func TestSyncDeploymentDeletionRace(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := *d
	// Lister (cache) says NOT deleted.
	f.dLister = append(f.dLister, d)
	// Bare client says it IS deleted. This should be presumed more up-to-date.
	now := metav1.Now()
	d2.DeletionTimestamp = &now
	f.objects = append(f.objects, &d2)

	// The recheck is only triggered if a matching orphan exists.
	rs := newReplicaSet(d, "rs1", 1)
	rs.OwnerReferences = nil
	f.objects = append(f.objects, rs)
	f.rsLister = append(f.rsLister, rs)

	// Expect to only recheck DeletionTimestamp.
	f.expectGetDeploymentAction(d)
	// Sync should fail and requeue to let cache catch up.
	// Don't start informers, since we don't want cache to catch up for this test.
	f.runExpectError(ctx, testutil.GetKey(d, t), false)
}

// issue: https://github.com/kubernetes/kubernetes/issues/23218
func TestDontSyncDeploymentsWithEmptyPodSelector(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d.Spec.Selector = &metav1.LabelSelector{}
	f.dLister = append(f.dLister, d)
	f.objects = append(f.objects, d)

	// Normally there should be a status update to sync observedGeneration but the fake
	// deployment has no generation set so there is no action happening here.
	f.run(ctx, testutil.GetKey(d, t))
}

func TestReentrantRollback(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d.Annotations = map[string]string{util.RevisionAnnotation: "2"}
	setRollbackTo(d, new(int64))
	f.dLister = append(f.dLister, d)

	rs1 := newReplicaSet(d, "deploymentrs-old", 0)
	rs1.Annotations = map[string]string{util.RevisionAnnotation: "1"}
	one := int64(1)
	rs1.Spec.Template.Spec.TerminationGracePeriodSeconds = &one
	rs1.Spec.Selector.MatchLabels[apps.DefaultDeploymentUniqueLabelKey] = "hash"

	rs2 := newReplicaSet(d, "deploymentrs-new", 1)
	rs2.Annotations = map[string]string{util.RevisionAnnotation: "2"}
	rs2.Spec.Selector.MatchLabels[apps.DefaultDeploymentUniqueLabelKey] = "hash"

	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, d, rs1, rs2)

	// Rollback is done here
	f.expectUpdateDeploymentAction(d)
	// Expect no update on replica sets though
	f.run(ctx, testutil.GetKey(d, t))
}

// TestPodDeletionEnqueuesRecreateDeployment ensures that the deletion of a pod
// will requeue a Recreate deployment regardless of whether other pods exist.
func TestPodDeletionEnqueuesRecreateDeployment(t *testing.T) {
	tests := []struct {
		name      string
		otherPods bool
	}{
		{
			name:      "last pod deleted",
			otherPods: false,
		},
		{
			name:      "pod with siblings deleted",
			otherPods: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			f := newFixture(t)

			foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
			foo.Spec.Strategy.Type = apps.RecreateDeploymentStrategyType
			rs := newReplicaSet(foo, "foo-1", 1)
			pod := generatePodFromRS(rs)

			f.dLister = append(f.dLister, foo)
			f.rsLister = append(f.rsLister, rs)
			f.objects = append(f.objects, foo, rs)
			f.podLister = append(f.podLister, pod)

			if tc.otherPods {
				// Add a sibling pod
				pod2 := generatePodFromRS(rs)
				pod2.Name = "foo-2"
				f.podLister = append(f.podLister, pod2)
			}

			c, _, err := f.newController(ctx)
			if err != nil {
				t.Fatalf("error creating Deployment controller: %v", err)
			}
			enqueued := false
			c.enqueueDeployment = func(d *apps.Deployment) {
				if d.Name == "foo" {
					enqueued = true
				}
			}

			c.deletePod(logger, pod)

			if !enqueued {
				t.Errorf("expected deployment %q to be queued after pod deletion", foo.Name)
			}
		})
	}
}

// TestPodDeletionPartialReplicaSetOwnershipEnqueueRecreateDeployment ensures that
// the deletion of a pod will requeue a Recreate deployment iff there is no other
// pod returned from the client in the case where a deployment has multiple replica
// sets, some of which have empty owner references.
func TestPodDeletionPartialReplicaSetOwnershipEnqueueRecreateDeployment(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	foo := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	foo.Spec.Strategy.Type = apps.RecreateDeploymentStrategyType
	rs1 := newReplicaSet(foo, "foo-1", 1)
	rs2 := newReplicaSet(foo, "foo-2", 2)
	rs2.OwnerReferences = nil
	pod := generatePodFromRS(rs1)

	f.dLister = append(f.dLister, foo)
	f.rsLister = append(f.rsLister, rs1, rs2)
	f.objects = append(f.objects, foo, rs1, rs2)

	c, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	enqueued := false
	c.enqueueDeployment = func(d *apps.Deployment) {
		if d.Name == "foo" {
			enqueued = true
		}
	}

	c.deletePod(logger, pod)

	if !enqueued {
		t.Errorf("expected deployment %q to be queued after pod deletion", foo.Name)
	}
}

func TestGetReplicaSetsForDeployment(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

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
	c, informers, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.StartWithContext(ctx)

	rsList, err := c.getReplicaSetsForDeployment(ctx, d1)
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

	rsList, err = c.getReplicaSetsForDeployment(ctx, d2)
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
	_, ctx := ktesting.NewTestContext(t)

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
	c, informers, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.StartWithContext(ctx)

	rsList, err := c.getReplicaSetsForDeployment(ctx, d)
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
	_, ctx := ktesting.NewTestContext(t)

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
	c, informers, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.StartWithContext(ctx)

	podMap, err := c.getPodMapForDeployment(d, f.rsLister)
	if err != nil {
		t.Fatalf("getPodMapForDeployment() error: %v", err)
	}
	podCount := 0
	for _, podList := range podMap {
		podCount += len(podList)
	}
	if got, want := podCount, 3; got != want {
		t.Errorf("podCount = %v, want %v", got, want)
	}

	if got, want := len(podMap), 2; got != want {
		t.Errorf("len(podMap) = %v, want %v", got, want)
	}
	if got, want := len(podMap[rs1.UID]), 2; got != want {
		t.Errorf("len(podMap[rs1]) = %v, want %v", got, want)
	}
	expect := map[string]struct{}{"rs1-pod": {}, "pod4": {}}
	for _, pod := range podMap[rs1.UID] {
		if _, ok := expect[pod.Name]; !ok {
			t.Errorf("unexpected pod name for rs1: %s", pod.Name)
		}
	}
	if got, want := len(podMap[rs2.UID]), 1; got != want {
		t.Errorf("len(podMap[rs2]) = %v, want %v", got, want)
	}
	if got, want := podMap[rs2.UID][0].Name, "rs2-pod"; got != want {
		t.Errorf("podMap[rs2] = [%v], want [%v]", got, want)
	}
}

func TestAddReplicaSet(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	dc.addReplicaSet(klog.FromContext(ctx), rs1)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	dc.addReplicaSet(logger, rs2)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestAddReplicaSetOrphan(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	dc.addReplicaSet(logger, rs)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSet(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	prev := *rs1
	next := *rs1
	bumpResourceVersion(&next)
	dc.updateReplicaSet(logger, &prev, &next)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	prev = *rs2
	next = *rs2
	bumpResourceVersion(&next)
	dc.updateReplicaSet(logger, &prev, &next)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetOrphanWithNewLabels(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	// Change labels and expect all matching controllers to queue.
	prev := *rs
	prev.Labels = map[string]string{"foo": "notbar"}
	next := *rs
	bumpResourceVersion(&next)
	dc.updateReplicaSet(logger, &prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetChangeControllerRef(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	rs := newReplicaSet(d1, "rs1", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	// Change ControllerRef and expect both old and new to queue.
	prev := *rs
	prev.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(d2, controllerKind)}
	next := *rs
	bumpResourceVersion(&next)
	dc.updateReplicaSet(logger, &prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdateReplicaSetRelease(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	f := newFixture(t)

	d1 := newDeployment("d1", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	d2 := newDeployment("d2", 1, nil, nil, nil, map[string]string{"foo": "bar"})

	rs := newReplicaSet(d1, "rs1", 1)

	f.dLister = append(f.dLister, d1, d2)
	f.rsLister = append(f.rsLister, rs)
	f.objects = append(f.objects, d1, d2, rs)

	// Create the fixture but don't start it,
	// so nothing happens in the background.
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	// Remove ControllerRef and expect all matching controller to sync orphan.
	prev := *rs
	next := *rs
	next.OwnerReferences = nil
	bumpResourceVersion(&next)
	dc.updateReplicaSet(logger, &prev, &next)
	if got, want := dc.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestDeleteReplicaSet(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	dc.deleteReplicaSet(logger, rs1)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs1.Name)
	}
	expectedKey, _ := controller.KeyFunc(d1)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	dc.deleteReplicaSet(logger, rs2)
	if got, want := dc.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = dc.queue.Get()
	if key == "" || done {
		t.Fatalf("failed to enqueue controller for rs %v", rs2.Name)
	}
	expectedKey, _ = controller.KeyFunc(d2)
	if got, want := key, expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestDeleteReplicaSetOrphan(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

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
	dc, _, err := f.newController(ctx)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}

	dc.deleteReplicaSet(logger, rs)
	if got, want := dc.queue.Len(), 0; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func BenchmarkGetPodMapForDeployment(b *testing.B) {
	cases := []struct {
		name      string
		numPods   int
		extraPods int // Pods in namespace not owned by this deployment
	}{
		{name: "10-Pods-No-Noise", numPods: 10, extraPods: 0},
		{name: "10-Pods-With-Noise", numPods: 10, extraPods: 100},
		{name: "10000-Pods-No-Noise", numPods: 10000, extraPods: 0},
		{name: "10000-Pods-With-Noise", numPods: 10000, extraPods: 100000},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			_, ctx := ktesting.NewTestContext(b)

			f := newFixture(b)

			d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})

			rs1 := newReplicaSet(d, "rs1", 1)
			rs2 := newReplicaSet(d, "rs2", 1)

			var pods []*v1.Pod
			var objects []runtime.Object
			for i := 0; i < tc.numPods; i++ {
				p1, p2 := generatePodFromRS(rs1), generatePodFromRS(rs2)
				p1.Name, p2.Name = p1.Name+fmt.Sprintf("-%d", i), p2.Name+fmt.Sprintf("-%d", i)
				pods = append(pods, p1, p2)
				objects = append(objects, p1, p2)
			}

			// Add extra pods that don't belong to the deployment
			for i := 0; i < tc.extraPods; i++ {
				p := generatePodFromRS(rs1)
				p.Name = fmt.Sprintf("extra-pod-%d", i)
				p.OwnerReferences = nil // Orphaned/unrelated
				p.Labels = map[string]string{"foo": "notbar"}
				pods = append(pods, p)
				objects = append(objects, p)
			}

			f.dLister = append(f.dLister, d)
			f.rsLister = append(f.rsLister, rs1, rs2)
			f.podLister = append(f.podLister, pods...)
			f.objects = append(f.objects, d, rs1, rs2)
			f.objects = append(f.objects, objects...)

			// Start the fixture.
			c, informers, err := f.newController(ctx)
			if err != nil {
				b.Fatalf("error creating Deployment controller: %v", err)
			}
			stopCh := make(chan struct{})
			defer close(stopCh)
			informers.StartWithContext(ctx)

			b.ReportAllocs()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				m, err := c.getPodMapForDeployment(d, f.rsLister)
				if err != nil {
					b.Fatalf("getPodMapForDeployment() error: %v", err)
				}
				if len(m) != 2 {
					b.Errorf("Invalid map size, expected 2, got: %d", len(m))
				}
			}
		})
	}
}

func bumpResourceVersion(obj metav1.Object) {
	ver, _ := strconv.ParseInt(obj.GetResourceVersion(), 10, 32)
	obj.SetResourceVersion(strconv.FormatInt(ver+1, 10))
}

// generatePodFromRS creates a pod, with the input ReplicaSet's selector and its template
func generatePodFromRS(rs *apps.ReplicaSet) *v1.Pod {
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
