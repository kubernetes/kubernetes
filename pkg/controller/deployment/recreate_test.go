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
	"errors"
	"fmt"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/utils/ptr"
)

func TestDeploymentController_rolloutRecreate(t *testing.T) {
	scaleErr := errors.New("ReplicaSet update failed")
	statusErr := errors.New("Deployment status update failed")
	tests := []struct {
		name        string
		oldReplicas int32
		hasNewRS    bool
	}{
		{name: "old ReplicaSet scale-down failure", oldReplicas: 1, hasNewRS: true},
		{name: "old ReplicaSet scale-down failure without a new ReplicaSet", oldReplicas: 1},
		{name: "new ReplicaSet scale-up failure", hasNewRS: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			statusTests := []struct {
				name            string
				progressExpired bool
				statusErr       error
			}{
				{name: "rollout times out", progressExpired: true},
				{name: "status is updated before the deadline"},
				{name: "scaling and status update errors are returned", progressExpired: true, statusErr: statusErr},
			}
			for _, statusTest := range statusTests {
				t.Run(statusTest.name, func(t *testing.T) {
					selector := map[string]string{"app": "foo"}
					d := newDeployment("foo", 4, nil, nil, nil, selector)
					d.Generation = 2
					d.Spec.Strategy = apps.DeploymentStrategy{Type: apps.RecreateDeploymentStrategyType}
					d.Spec.ProgressDeadlineSeconds = ptr.To[int32](600)
					d.Annotations[deploymentutil.RevisionAnnotation] = "2"

					oldRS := newReplicaSet(d, "foo-old", test.oldReplicas)
					oldRS.Spec.Template = *d.Spec.Template.DeepCopy()
					oldRS.Spec.Template.Spec.Containers[0].Image = "foo/old"
					oldRS.Annotations = map[string]string{deploymentutil.RevisionAnnotation: "1"}
					oldRS.Status = apps.ReplicaSetStatus{Replicas: test.oldReplicas, ReadyReplicas: test.oldReplicas, AvailableReplicas: test.oldReplicas}
					var newRS *apps.ReplicaSet
					allRSs := []*apps.ReplicaSet{oldRS}
					objects := []runtime.Object{d, oldRS}
					if test.hasNewRS {
						newRS = newReplicaSet(d, "foo-new", 0)
						newRS.Annotations = map[string]string{deploymentutil.RevisionAnnotation: "2"}
						allRSs = append(allRSs, newRS)
						objects = append(objects, newRS)
					}

					d.Status = calculateStatus(allRSs, newRS, d)
					d.Status.ObservedGeneration = 1
					condition := deploymentutil.NewDeploymentCondition(apps.DeploymentProgressing, v1.ConditionTrue, deploymentutil.ReplicaSetUpdatedReason, "Deployment is progressing.")
					if statusTest.progressExpired {
						condition.LastUpdateTime = metav1.NewTime(time.Now().Add(-time.Hour))
					}
					deploymentutil.SetDeploymentCondition(&d.Status, *condition)

					client := fake.NewClientset(objects...)
					client.PrependReactor("update", "replicasets", func(action core.Action) (bool, runtime.Object, error) {
						return true, nil, scaleErr
					})
					if statusTest.statusErr != nil {
						client.PrependReactor("update", "deployments", func(action core.Action) (bool, runtime.Object, error) {
							return true, nil, statusTest.statusErr
						})
					}
					dc := &DeploymentController{
						client:        client,
						eventRecorder: &record.FakeRecorder{},
					}
					_, ctx := ktesting.NewTestContext(t)
					err := dc.rolloutRecreate(ctx, d, allRSs, nil)
					if !errors.Is(err, scaleErr) {
						t.Errorf("expected scaling error %v, got %v", scaleErr, err)
					}
					if statusTest.statusErr != nil && !errors.Is(err, statusTest.statusErr) {
						t.Errorf("expected status update error %v, got %v", statusTest.statusErr, err)
					}

					actions := client.Actions()
					if len(actions) != 2 {
						t.Fatalf("expected ReplicaSet update followed by Deployment status update, got %v", actions)
					}
					if !actions[0].Matches("update", "replicasets") {
						t.Fatalf("expected ReplicaSet update, got %v", actions[0])
					}
					scaledRS := actions[0].(core.UpdateAction).GetObject().(*apps.ReplicaSet)
					expectedRS, expectedReplicas := oldRS.Name, int32(0)
					if test.oldReplicas == 0 {
						expectedRS, expectedReplicas = newRS.Name, *d.Spec.Replicas
					}
					if scaledRS.Name != expectedRS || *scaledRS.Spec.Replicas != expectedReplicas {
						t.Errorf("expected ReplicaSet %s to scale to %d replicas, got %s with %d replicas", expectedRS, expectedReplicas, scaledRS.Name, *scaledRS.Spec.Replicas)
					}
					if !actions[1].Matches("update", "deployments") || actions[1].GetSubresource() != "status" {
						t.Fatalf("expected Deployment status update, got %v", actions[1])
					}
					updated := actions[1].(core.UpdateAction).GetObject().(*apps.Deployment)
					progressing := deploymentutil.GetDeploymentCondition(updated.Status, apps.DeploymentProgressing)
					if progressing == nil {
						t.Fatal("expected Progressing condition")
					}
					expectedReason, expectedStatus := deploymentutil.ReplicaSetUpdatedReason, v1.ConditionTrue
					if statusTest.progressExpired {
						expectedReason, expectedStatus = deploymentutil.TimedOutReason, v1.ConditionFalse
					}
					if progressing.Reason != expectedReason || progressing.Status != expectedStatus {
						t.Errorf("expected Progressing condition %s with reason %s, got %+v", expectedStatus, expectedReason, progressing)
					}
					if updated.Status.ObservedGeneration != d.Generation {
						t.Errorf("expected observed generation %d, got %d", d.Generation, updated.Status.ObservedGeneration)
					}
					if statusTest.statusErr == nil {
						stored, err := client.AppsV1().Deployments(d.Namespace).Get(ctx, d.Name, metav1.GetOptions{})
						if err != nil {
							t.Fatal(err)
						}
						storedCondition := deploymentutil.GetDeploymentCondition(stored.Status, apps.DeploymentProgressing)
						if storedCondition == nil || storedCondition.Status != expectedStatus || storedCondition.Reason != expectedReason {
							t.Errorf("expected persisted Progressing condition %s with reason %s, got %+v", expectedStatus, expectedReason, storedCondition)
						}
					}
				})
			}
		})
	}
}

func TestScaleDownOldReplicaSets(t *testing.T) {
	tests := []struct {
		oldRSSizes []int32
		d          *apps.Deployment
	}{
		{
			oldRSSizes: []int32{3},
			d:          newDeployment("foo", 3, nil, nil, nil, map[string]string{"foo": "bar"}),
		},
	}

	for i := range tests {
		t.Logf("running scenario %d", i)
		test := tests[i]

		var oldRSs []*apps.ReplicaSet
		var expected []runtime.Object

		for n, size := range test.oldRSSizes {
			rs := newReplicaSet(test.d, fmt.Sprintf("%s-%d", test.d.Name, n), size)
			oldRSs = append(oldRSs, rs)

			rsCopy := rs.DeepCopy()

			zero := int32(0)
			rsCopy.Spec.Replicas = &zero
			expected = append(expected, rsCopy)

			if *(oldRSs[n].Spec.Replicas) == *(expected[n].(*apps.ReplicaSet).Spec.Replicas) {
				t.Errorf("broken test - original and expected RS have the same size")
			}
		}

		kc := fake.NewSimpleClientset(expected...)
		informers := informers.NewSharedInformerFactory(kc, controller.NoResyncPeriodFunc())
		_, ctx := ktesting.NewTestContext(t)
		c, err := NewDeploymentController(ctx, informers.Apps().V1().Deployments(), informers.Apps().V1().ReplicaSets(), informers.Core().V1().Pods(), kc)
		if err != nil {
			t.Fatalf("error creating Deployment controller: %v", err)
		}
		c.eventRecorder = &record.FakeRecorder{}

		c.scaleDownOldReplicaSetsForRecreate(ctx, oldRSs, test.d)
		for j := range oldRSs {
			rs := oldRSs[j]

			if *rs.Spec.Replicas != 0 {
				t.Errorf("rs %q has non-zero replicas", rs.Name)
			}
		}
	}
}

func TestOldPodsRunning(t *testing.T) {
	tests := []struct {
		name string

		newRS  *apps.ReplicaSet
		oldRSs []*apps.ReplicaSet
		podMap map[types.UID][]*v1.Pod

		hasOldPodsRunning bool
	}{
		{
			name:              "no old RSs",
			hasOldPodsRunning: false,
		},
		{
			name:              "old RSs with running pods",
			oldRSs:            []*apps.ReplicaSet{rsWithUID("some-uid"), rsWithUID("other-uid")},
			podMap:            podMapWithUIDs([]string{"some-uid", "other-uid"}),
			hasOldPodsRunning: true,
		},
		{
			name:              "old RSs without pods but with non-zero status replicas",
			oldRSs:            []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 1, nil)},
			hasOldPodsRunning: true,
		},
		{
			name:              "old RSs without pods or non-zero status replicas",
			oldRSs:            []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			hasOldPodsRunning: false,
		},
		{
			name:   "old RSs with zero status replicas but pods in terminal state are present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID][]*v1.Pod{
				"uid-1": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodFailed,
						},
					},
					{
						Status: v1.PodStatus{
							Phase: v1.PodSucceeded,
						},
					},
				},
			},
			hasOldPodsRunning: false,
		},
		{
			name:   "old RSs with zero status replicas but pod in unknown phase present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID][]*v1.Pod{
				"uid-1": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodUnknown,
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas with pending pod present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID][]*v1.Pod{
				"uid-1": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodPending,
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas with running pod present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID][]*v1.Pod{
				"uid-1": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodRunning,
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
		{
			name:   "old RSs with zero status replicas but pods in terminal state and pending are present",
			oldRSs: []*apps.ReplicaSet{newRSWithStatus("rs-1", 0, 0, nil)},
			podMap: map[types.UID][]*v1.Pod{
				"uid-1": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodFailed,
						},
					},
					{
						Status: v1.PodStatus{
							Phase: v1.PodSucceeded,
						},
					},
				},
				"uid-2": {},
				"uid-3": {
					{
						Status: v1.PodStatus{
							Phase: v1.PodPending,
						},
					},
				},
			},
			hasOldPodsRunning: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if expected, got := test.hasOldPodsRunning, oldPodsRunning(test.newRS, test.oldRSs, test.podMap); expected != got {
				t.Errorf("%s: expected %t, got %t", test.name, expected, got)
			}
		})
	}
}

func rsWithUID(uid string) *apps.ReplicaSet {
	d := newDeployment("foo", 1, nil, nil, nil, map[string]string{"foo": "bar"})
	rs := newReplicaSet(d, fmt.Sprintf("foo-%s", uid), 0)
	rs.UID = types.UID(uid)
	return rs
}

func podMapWithUIDs(uids []string) map[types.UID][]*v1.Pod {
	podMap := make(map[types.UID][]*v1.Pod)
	for _, uid := range uids {
		podMap[types.UID(uid)] = []*v1.Pod{
			{ /* supposedly a pod */ },
			{ /* supposedly another pod pod */ },
		}
	}
	return podMap
}
