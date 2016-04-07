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
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func rs(name string, replicas int, selector map[string]string) *exp.ReplicaSet {
	return &exp.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: exp.ReplicaSetSpec{
			Replicas: int32(replicas),
			Selector: &unversioned.LabelSelector{MatchLabels: selector},
			Template: api.PodTemplateSpec{},
		},
	}
}

func newRSWithStatus(name string, specReplicas, statusReplicas int, selector map[string]string) *exp.ReplicaSet {
	rs := rs(name, specReplicas, selector)
	rs.Status = exp.ReplicaSetStatus{
		Replicas: int32(statusReplicas),
	}
	return rs
}

func deployment(name string, replicas int, maxSurge, maxUnavailable intstr.IntOrString) exp.Deployment {
	return exp.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: exp.DeploymentSpec{
			Replicas: int32(replicas),
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

func newDeployment(replicas int, revisionHistoryLimit *int) *exp.Deployment {
	var v *int32
	if revisionHistoryLimit != nil {
		v = new(int32)
		*v = int32(*revisionHistoryLimit)
	}
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

func newListOptions() api.ListOptions {
	return api.ListOptions{}
}

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
		newRS := rs("foo-v2", test.newReplicas, nil)
		oldRS := rs("foo-v2", test.oldReplicas, nil)
		allRSs := []*exp.ReplicaSet{newRS, oldRS}
		deployment := deployment("foo", test.deploymentReplicas, test.maxSurge, intstr.FromInt(0))
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
		newRS := rs("foo-new", test.newReplicas, newSelector)
		oldRS := rs("foo-old", test.oldReplicas, oldSelector)
		oldRSs := []*exp.ReplicaSet{oldRS}
		allRSs := []*exp.ReplicaSet{oldRS, newRS}

		deployment := deployment("foo", test.deploymentReplicas, intstr.FromInt(0), test.maxUnavailable)
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
		oldRS := rs("foo-v2", test.oldReplicas, nil)
		oldRSs := []*exp.ReplicaSet{oldRS}
		deployment := deployment("foo", 10, intstr.FromInt(2), intstr.FromInt(2))
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
		_, cleanupCount, err := controller.cleanupUnhealthyReplicas(oldRSs, &deployment, int32(test.maxCleanupCount))
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
	}{
		{
			deploymentReplicas:  10,
			maxUnavailable:      intstr.FromInt(0),
			readyPods:           10,
			oldReplicas:         10,
			scaleExpected:       true,
			expectedOldReplicas: 9,
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
		{
			deploymentReplicas: 10,
			maxUnavailable:     intstr.FromInt(2),
			readyPods:          1,
			oldReplicas:        10,
			scaleExpected:      false,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		oldRS := rs("foo-v2", test.oldReplicas, nil)
		allRSs := []*exp.ReplicaSet{oldRS}
		oldRSs := []*exp.ReplicaSet{oldRS}
		deployment := deployment("foo", test.deploymentReplicas, intstr.FromInt(0), test.maxUnavailable)
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
		updated := updateAction.GetObject().(*exp.ReplicaSet)
		if e, a := test.expectedOldReplicas, int(updated.Spec.Replicas); e != a {
			t.Errorf("expected update to %d replicas, got %d", e, a)
		}
	}
}

func TestDeploymentController_cleanupOldReplicaSets(t *testing.T) {
	selector := map[string]string{"foo": "bar"}

	tests := []struct {
		oldRSs               []*exp.ReplicaSet
		revisionHistoryLimit int
		expectedDeletions    int
	}{
		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
				newRSWithStatus("foo-3", 0, 0, selector),
			},
			revisionHistoryLimit: 1,
			expectedDeletions:    2,
		},
		{
			// Only delete the replica set with Spec.Replicas = Status.Replicas = 0.
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 1, selector),
				newRSWithStatus("foo-3", 1, 0, selector),
				newRSWithStatus("foo-4", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    1,
		},

		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 0, 0, selector),
				newRSWithStatus("foo-2", 0, 0, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    2,
		},
		{
			oldRSs: []*exp.ReplicaSet{
				newRSWithStatus("foo-1", 1, 1, selector),
				newRSWithStatus("foo-2", 1, 1, selector),
			},
			revisionHistoryLimit: 0,
			expectedDeletions:    0,
		},
	}

	for i, test := range tests {
		fake := &fake.Clientset{}
		controller := NewDeploymentController(fake, controller.NoResyncPeriodFunc)

		controller.eventRecorder = &record.FakeRecorder{}
		controller.rsStoreSynced = alwaysReady
		controller.podStoreSynced = alwaysReady
		for _, rs := range test.oldRSs {
			controller.rsStore.Add(rs)
		}

		d := newDeployment(1, &tests[i].revisionHistoryLimit)
		controller.cleanupOldReplicaSets(test.oldRSs, d)

		gotDeletions := 0
		for _, action := range fake.Actions() {
			if "delete" == action.GetVerb() {
				gotDeletions++
			}
		}
		if gotDeletions != test.expectedDeletions {
			t.Errorf("expect %v old replica sets been deleted, but got %v", test.expectedDeletions, gotDeletions)
			continue
		}
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
	objects *api.List
}

func (f *fixture) expectUpdateDeploymentAction(d *exp.Deployment) {
	f.actions = append(f.actions, core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "deployments"}, d.Namespace, d))
	f.objects.Items = append(f.objects.Items, d)
}

func (f *fixture) expectCreateRSAction(rs *exp.ReplicaSet) {
	f.actions = append(f.actions, core.NewCreateAction(unversioned.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
	f.objects.Items = append(f.objects.Items, rs)
}

func (f *fixture) expectUpdateRSAction(rs *exp.ReplicaSet) {
	f.actions = append(f.actions, core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "replicasets"}, rs.Namespace, rs))
	f.objects.Items = append(f.objects.Items, rs)
}

func (f *fixture) expectListPodAction(namespace string, opt api.ListOptions) {
	f.actions = append(f.actions, core.NewListAction(unversioned.GroupVersionResource{Resource: "pods"}, namespace, opt))
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{}
	f.t = t
	f.objects = &api.List{}
	return f
}

func (f *fixture) run(deploymentName string) {
	f.client = fake.NewSimpleClientset(f.objects)
	c := NewDeploymentController(f.client, controller.NoResyncPeriodFunc)
	c.eventRecorder = &record.FakeRecorder{}
	c.rsStoreSynced = alwaysReady
	c.podStoreSynced = alwaysReady
	for _, d := range f.dStore {
		c.dStore.Store.Add(d)
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

	// expect that one ReplicaSet with zero replicas is created
	// then is updated to 1 replica
	rs := newReplicaSet(d, "deploymentrs-4186632231", 0)
	updatedRS := newReplicaSet(d, "deploymentrs-4186632231", 1)
	opt := newListOptions()

	f.expectCreateRSAction(rs)
	f.expectUpdateDeploymentAction(d)
	f.expectUpdateRSAction(updatedRS)
	f.expectListPodAction(rs.Namespace, opt)
	f.expectUpdateDeploymentAction(d)

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
	controller.dStore.Store.Add(d)
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
