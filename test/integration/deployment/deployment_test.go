/*
Copyright 2017 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

func TestNewDeployment(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-new-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int32(20)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	tester.deployment.Spec.MinReadySeconds = 4

	tester.deployment.Annotations = map[string]string{"test": "should-copy-to-replica-set", v1.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Make sure the Deployment completes while manually marking Deployment pods as ready at the same time.
	// Use soft check because this deployment was just created and rolling update strategy might be violated.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// Check new RS annotations
	newRS, err := tester.expectNewReplicaSet()
	if err != nil {
		t.Fatal(err)
	}
	if newRS.Annotations["test"] != "should-copy-to-replica-set" {
		t.Errorf("expected new ReplicaSet annotations copied from Deployment %s, got: %v", tester.deployment.Name, newRS.Annotations)
	}
	if newRS.Annotations[v1.LastAppliedConfigAnnotation] != "" {
		t.Errorf("expected new ReplicaSet last-applied annotation not copied from Deployment %s", tester.deployment.Name)
	}

	// New RS should contain pod-template-hash in its selector, label, and template label
	rsHash, err := checkRSHashLabels(newRS)
	if err != nil {
		t.Error(err)
	}

	// All pods targeted by the deployment should contain pod-template-hash in their labels
	selector, err := metav1.LabelSelectorAsSelector(tester.deployment.Spec.Selector)
	if err != nil {
		t.Fatalf("failed to parse deployment %s selector: %v", name, err)
	}
	pods, err := c.CoreV1().Pods(ns.Name).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		t.Fatalf("failed to list pods of deployment %s: %v", name, err)
	}
	if len(pods.Items) != int(replicas) {
		t.Errorf("expected %d pods, got %d pods", replicas, len(pods.Items))
	}
	podHash, err := checkPodsHashLabel(pods)
	if err != nil {
		t.Error(err)
	}
	if rsHash != podHash {
		t.Errorf("found mismatching pod-template-hash value: rs hash = %s whereas pod hash = %s", rsHash, podHash)
	}
}

// Deployments should support roll out, roll back, and roll over
func TestDeploymentRollingUpdate(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-rolling-update-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	replicas := int32(20)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	tester.deployment.Spec.MinReadySeconds = 4
	quarter := intstr.FromString("25%")
	tester.deployment.Spec.Strategy.RollingUpdate = &apps.RollingUpdateDeployment{
		MaxUnavailable: &quarter,
		MaxSurge:       &quarter,
	}

	// Create a deployment.
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}
	oriImage := tester.deployment.Spec.Template.Spec.Containers[0].Image
	if err := tester.waitForDeploymentRevisionAndImage("1", oriImage); err != nil {
		t.Fatal(err)
	}
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// 1. Roll out a new image.
	image := "new-image"
	if oriImage == image {
		t.Fatalf("bad test setup, deployment %s roll out with the same image", tester.deployment.Name)
	}
	imageFn := func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = image
	}
	tester.deployment, err = tester.updateDeployment(imageFn)
	if err != nil {
		t.Fatalf("failed to update deployment %s: %v", tester.deployment.Name, err)
	}
	if err := tester.waitForDeploymentRevisionAndImage("2", image); err != nil {
		t.Fatal(err)
	}
	if err := tester.waitForDeploymentCompleteAndCheckRollingAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// 2. Roll back to the last revision.
	revision := int64(0)
	rollback := newDeploymentRollback(tester.deployment.Name, nil, revision)
	if err = c.ExtensionsV1beta1().Deployments(ns.Name).Rollback(rollback); err != nil {
		t.Fatalf("failed to roll back deployment %s to last revision: %v", tester.deployment.Name, err)
	}
	// Wait for the deployment to start rolling back
	if err = tester.waitForDeploymentRollbackCleared(); err != nil {
		t.Fatalf("failed to roll back deployment %s to last revision: %v", tester.deployment.Name, err)
	}
	// Wait for the deployment to be rolled back to the template stored in revision 1 and rolled forward to revision 3.
	if err := tester.waitForDeploymentRevisionAndImage("3", oriImage); err != nil {
		t.Fatal(err)
	}
	if err := tester.waitForDeploymentCompleteAndCheckRollingAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// 3. Roll over a deployment before the previous rolling update finishes.
	image = "dont-finish"
	imageFn = func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = image
	}
	tester.deployment, err = tester.updateDeployment(imageFn)
	if err != nil {
		t.Fatalf("failed to update deployment %s: %v", tester.deployment.Name, err)
	}
	if err := tester.waitForDeploymentRevisionAndImage("4", image); err != nil {
		t.Fatal(err)
	}
	// We don't mark pods as ready so that rollout won't finish.
	// Before the rollout finishes, trigger another rollout.
	image = "rollover"
	imageFn = func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = image
	}
	tester.deployment, err = tester.updateDeployment(imageFn)
	if err != nil {
		t.Fatalf("failed to update deployment %s: %v", tester.deployment.Name, err)
	}
	if err := tester.waitForDeploymentRevisionAndImage("5", image); err != nil {
		t.Fatal(err)
	}
	if err := tester.waitForDeploymentCompleteAndCheckRollingAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed retrieving old replicasets of deployment %s: %v", tester.deployment.Name, err)
	}
	for _, oldRS := range allOldRSs {
		if *oldRS.Spec.Replicas != 0 {
			t.Errorf("expected old replicaset %s of deployment %s to have 0 replica, got %d", oldRS.Name, tester.deployment.Name, *oldRS.Spec.Replicas)
		}
	}
}

// selectors are IMMUTABLE for all API versions except apps/v1beta1 and extensions/v1beta1
func TestDeploymentSelectorImmutability(t *testing.T) {
	s, closeFn, c := dcSimpleSetup(t)
	defer closeFn()
	name := "test-deployment-selector-immutability"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, int32(20))}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create apps/v1 deployment %s: %v", tester.deployment.Name, err)
	}

	// test to ensure extensions/v1beta1 selector is mutable
	newSelectorLabels := map[string]string{"name_extensions_v1beta1": "test_extensions_v1beta1"}
	deploymentExtensionsV1beta1, err := c.ExtensionsV1beta1().Deployments(ns.Name).Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get extensions/v1beta deployment %s: %v", name, err)
	}
	deploymentExtensionsV1beta1.Spec.Selector.MatchLabels = newSelectorLabels
	deploymentExtensionsV1beta1.Spec.Template.Labels = newSelectorLabels
	updatedDeploymentExtensionsV1beta1, err := c.ExtensionsV1beta1().Deployments(ns.Name).Update(deploymentExtensionsV1beta1)
	if err != nil {
		t.Fatalf("failed to update extensions/v1beta1 deployment %s: %v", deploymentExtensionsV1beta1.Name, err)
	}
	if !reflect.DeepEqual(updatedDeploymentExtensionsV1beta1.Spec.Selector.MatchLabels, newSelectorLabels) {
		t.Errorf("selector should be changed for extensions/v1beta1, expected: %v, got: %v", newSelectorLabels, updatedDeploymentExtensionsV1beta1.Spec.Selector.MatchLabels)
	}

	// test to ensure apps/v1beta1 selector is mutable
	deploymentAppsV1beta1, err := c.AppsV1beta1().Deployments(ns.Name).Get(updatedDeploymentExtensionsV1beta1.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get apps/v1beta1 deployment %s: %v", updatedDeploymentExtensionsV1beta1.Name, err)
	}

	newSelectorLabels = map[string]string{"name_apps_v1beta1": "test_apps_v1beta1"}
	deploymentAppsV1beta1.Spec.Selector.MatchLabels = newSelectorLabels
	deploymentAppsV1beta1.Spec.Template.Labels = newSelectorLabels
	updatedDeploymentAppsV1beta1, err := c.AppsV1beta1().Deployments(ns.Name).Update(deploymentAppsV1beta1)
	if err != nil {
		t.Fatalf("failed to update apps/v1beta1 deployment %s: %v", deploymentAppsV1beta1.Name, err)
	}
	if !reflect.DeepEqual(updatedDeploymentAppsV1beta1.Spec.Selector.MatchLabels, newSelectorLabels) {
		t.Errorf("selector should be changed for apps/v1beta1, expected: %v, got: %v", newSelectorLabels, updatedDeploymentAppsV1beta1.Spec.Selector.MatchLabels)
	}

	// test to ensure apps/v1beta2 selector is immutable
	deploymentAppsV1beta2, err := c.AppsV1beta2().Deployments(ns.Name).Get(updatedDeploymentAppsV1beta1.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get apps/v1beta2 deployment %s: %v", updatedDeploymentAppsV1beta1.Name, err)
	}
	newSelectorLabels = map[string]string{"name_apps_v1beta2": "test_apps_v1beta2"}
	deploymentAppsV1beta2.Spec.Selector.MatchLabels = newSelectorLabels
	deploymentAppsV1beta2.Spec.Template.Labels = newSelectorLabels
	_, err = c.AppsV1beta2().Deployments(ns.Name).Update(deploymentAppsV1beta2)
	if err == nil {
		t.Fatalf("failed to provide validation error when changing immutable selector when updating apps/v1beta2 deployment %s", deploymentAppsV1beta2.Name)
	}
	expectedErrType := "Invalid value"
	expectedErrDetail := "field is immutable"
	if !strings.Contains(err.Error(), expectedErrType) || !strings.Contains(err.Error(), expectedErrDetail) {
		t.Errorf("error message does not match, expected type: %s, expected detail: %s, got: %s", expectedErrType, expectedErrDetail, err.Error())
	}

	// test to ensure apps/v1 selector is immutable
	deploymentAppsV1, err := c.AppsV1().Deployments(ns.Name).Get(updatedDeploymentAppsV1beta1.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get apps/v1 deployment %s: %v", updatedDeploymentAppsV1beta1.Name, err)
	}
	newSelectorLabels = map[string]string{"name_apps_v1": "test_apps_v1"}
	deploymentAppsV1.Spec.Selector.MatchLabels = newSelectorLabels
	deploymentAppsV1.Spec.Template.Labels = newSelectorLabels
	_, err = c.AppsV1().Deployments(ns.Name).Update(deploymentAppsV1)
	if err == nil {
		t.Fatalf("failed to provide validation error when changing immutable selector when updating apps/v1 deployment %s", deploymentAppsV1.Name)
	}
	if !strings.Contains(err.Error(), expectedErrType) || !strings.Contains(err.Error(), expectedErrDetail) {
		t.Errorf("error message does not match, expected type: %s, expected detail: %s, got: %s", expectedErrType, expectedErrDetail, err.Error())
	}
}

// Paused deployment should not start new rollout
func TestPausedDeployment(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-paused-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	tester.deployment.Spec.Paused = true
	tgps := int64(1)
	tester.deployment.Spec.Template.Spec.TerminationGracePeriodSeconds = &tgps

	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Verify that the paused deployment won't create new replica set.
	if err := tester.expectNoNewReplicaSet(); err != nil {
		t.Fatal(err)
	}

	// Resume the deployment
	tester.deployment, err = tester.updateDeployment(resumeFn)
	if err != nil {
		t.Fatalf("failed to resume deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the controller to notice the resume.
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Make sure the Deployment completes while manually marking Deployment pods as ready at the same time.
	// Use soft check because this deployment was just created and rolling update strategy might be violated.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// A new replicaset should be created.
	if _, err := tester.expectNewReplicaSet(); err != nil {
		t.Fatal(err)
	}

	// Pause the deployment.
	// The paused deployment shouldn't trigger a new rollout.
	tester.deployment, err = tester.updateDeployment(pauseFn)
	if err != nil {
		t.Fatalf("failed to pause deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the controller to notice the pause.
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Update the deployment template
	newTGPS := int64(0)
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Template.Spec.TerminationGracePeriodSeconds = &newTGPS
	})
	if err != nil {
		t.Fatalf("failed updating template of deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the controller to notice the rollout.
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Verify that the paused deployment won't create new replica set.
	if err := tester.expectNoNewReplicaSet(); err != nil {
		t.Fatal(err)
	}

	_, allOldRs, err := deploymentutil.GetOldReplicaSets(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed retrieving old replicasets of deployment %s: %v", tester.deployment.Name, err)
	}
	if len(allOldRs) != 1 {
		t.Errorf("expected an old replica set, got %v", allOldRs)
	}
	if *allOldRs[0].Spec.Template.Spec.TerminationGracePeriodSeconds == newTGPS {
		t.Errorf("TerminationGracePeriodSeconds on the replica set should be %d, got %d", tgps, newTGPS)
	}
}

// Paused deployment can be scaled
func TestScalePausedDeployment(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-scale-paused-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	tgps := int64(1)
	tester.deployment.Spec.Template.Spec.TerminationGracePeriodSeconds = &tgps

	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Make sure the Deployment completes while manually marking Deployment pods as ready at the same time.
	// Use soft check because this deployment was just created and rolling update strategy might be violated.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// A new replicaset should be created.
	if _, err := tester.expectNewReplicaSet(); err != nil {
		t.Fatal(err)
	}

	// Pause the deployment.
	tester.deployment, err = tester.updateDeployment(pauseFn)
	if err != nil {
		t.Fatalf("failed to pause deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the controller to notice the scale.
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Scale the paused deployment.
	newReplicas := int32(10)
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	if err != nil {
		t.Fatalf("failed updating deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the controller to notice the scale.
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Verify that the new replicaset is scaled.
	rs, err := tester.expectNewReplicaSet()
	if err != nil {
		t.Fatal(err)
	}
	if *rs.Spec.Replicas != newReplicas {
		t.Errorf("expected new replicaset replicas = %d, got %d", newReplicas, *rs.Spec.Replicas)
	}

	// Make sure the Deployment completes while manually marking Deployment pods as ready at the same time.
	// Use soft check because this deployment was just scaled and rolling update strategy might be violated.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}
}

// Deployment rollout shouldn't be blocked on hash collisions
func TestDeploymentHashCollision(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-hash-collision-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}

	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Mock a hash collision
	newRS, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed getting new replicaset of deployment %s: %v", tester.deployment.Name, err)
	}
	if newRS == nil {
		t.Fatalf("unable to find new replicaset of deployment %s", tester.deployment.Name)
	}
	_, err = tester.updateReplicaSet(newRS.Name, func(update *apps.ReplicaSet) {
		*update.Spec.Template.Spec.TerminationGracePeriodSeconds = int64(5)
	})
	if err != nil {
		t.Fatalf("failed updating replicaset %s template: %v", newRS.Name, err)
	}

	// Expect deployment collision counter to increment
	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		d, err := c.AppsV1().Deployments(ns.Name).Get(tester.deployment.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return d.Status.CollisionCount != nil && *d.Status.CollisionCount == int32(1), nil
	}); err != nil {
		t.Fatalf("Failed to increment collision counter for deployment %q: %v", tester.deployment.Name, err)
	}

	// Expect a new ReplicaSet to be created
	if err := tester.waitForDeploymentRevisionAndImage("2", fakeImage); err != nil {
		t.Fatal(err)
	}
}

// Deployment supports rollback even when there's old replica set without revision.
func TestRollbackDeploymentRSNoRevision(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-rollback-no-revision-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	// Create an old RS without revision
	rsName := "test-rollback-no-revision-controller"
	rsReplicas := int32(1)
	rs := newReplicaSet(rsName, ns.Name, rsReplicas)
	rs.Annotations = make(map[string]string)
	rs.Annotations["make"] = "difference"
	rs.Spec.Template.Spec.Containers[0].Image = "different-image"
	_, err := c.AppsV1().ReplicaSets(ns.Name).Create(rs)
	if err != nil {
		t.Fatalf("failed to create replicaset %s: %v", rsName, err)
	}

	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	oriImage := tester.deployment.Spec.Template.Spec.Containers[0].Image
	// Set absolute rollout limits (defaults changed to percentages)
	max := intstr.FromInt(1)
	tester.deployment.Spec.Strategy.RollingUpdate.MaxUnavailable = &max
	tester.deployment.Spec.Strategy.RollingUpdate.MaxSurge = &max

	// Create a deployment which have different template than the replica set created above.
	if tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment); err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err = tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// 1. Rollback to the last revision
	//    Since there's only 1 revision in history, it should still be revision 1
	revision := int64(0)
	rollback := newDeploymentRollback(tester.deployment.Name, nil, revision)
	if err = c.ExtensionsV1beta1().Deployments(ns.Name).Rollback(rollback); err != nil {
		t.Fatalf("failed to roll back deployment %s to last revision: %v", tester.deployment.Name, err)
	}

	// Wait for the deployment to start rolling back
	if err = tester.waitForDeploymentRollbackCleared(); err != nil {
		t.Fatalf("failed to roll back deployment %s to last revision: %v", tester.deployment.Name, err)
	}
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no last revision
	// Check if the deployment is still revision 1 and still has the old pod template
	err = tester.checkDeploymentRevisionAndImage("1", oriImage)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Update the deployment to revision 2.
	updatedImage := "update"
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedImage
		update.Spec.Template.Spec.Containers[0].Image = updatedImage
	})
	if err != nil {
		t.Fatalf("failed updating deployment %s: %v", tester.deployment.Name, err)
	}

	// Use observedGeneration to determine if the controller noticed the pod template update.
	// Wait for the controller to notice the resume.
	if err = tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Wait for it to be updated to revision 2
	if err = tester.waitForDeploymentRevisionAndImage("2", updatedImage); err != nil {
		t.Fatal(err)
	}

	// Wait for the Deployment to complete while manually marking Deployment pods as ready at the same time
	if err = tester.waitForDeploymentCompleteAndCheckRollingAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// 3. Update the deploymentRollback to rollback to revision 1
	revision = int64(1)
	rollback = newDeploymentRollback(tester.deployment.Name, nil, revision)
	if err = c.ExtensionsV1beta1().Deployments(ns.Name).Rollback(rollback); err != nil {
		t.Fatalf("failed to roll back deployment %s to revision %d: %v", tester.deployment.Name, revision, err)
	}

	// Wait for the deployment to start rolling back
	if err = tester.waitForDeploymentRollbackCleared(); err != nil {
		t.Fatalf("failed to roll back deployment %s to revision %d: %v", tester.deployment.Name, revision, err)
	}
	// TODO: report RollbackDone in deployment status and check it here

	// The pod template should be updated to the one in revision 1
	// Wait for it to be updated to revision 3
	if err = tester.waitForDeploymentRevisionAndImage("3", oriImage); err != nil {
		t.Fatal(err)
	}

	// Wait for the Deployment to complete while manually marking Deployment pods as ready at the same time
	if err = tester.waitForDeploymentCompleteAndCheckRollingAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}
}

func checkRSHashLabels(rs *apps.ReplicaSet) (string, error) {
	hash := rs.Labels[apps.DefaultDeploymentUniqueLabelKey]
	selectorHash := rs.Spec.Selector.MatchLabels[apps.DefaultDeploymentUniqueLabelKey]
	templateLabelHash := rs.Spec.Template.Labels[apps.DefaultDeploymentUniqueLabelKey]

	if hash != selectorHash || selectorHash != templateLabelHash {
		return "", fmt.Errorf("mismatching hash value found in replicaset %s: %#v", rs.Name, rs)
	}
	if len(hash) == 0 {
		return "", fmt.Errorf("unexpected replicaset %s missing required pod-template-hash labels", rs.Name)
	}

	if !strings.HasSuffix(rs.Name, hash) {
		return "", fmt.Errorf("unexpected replicaset %s name suffix doesn't match hash %s", rs.Name, hash)
	}

	return hash, nil
}

func checkPodsHashLabel(pods *v1.PodList) (string, error) {
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no pods given")
	}
	var hash string
	for _, pod := range pods.Items {
		podHash := pod.Labels[apps.DefaultDeploymentUniqueLabelKey]
		if len(podHash) == 0 {
			return "", fmt.Errorf("found pod %s missing pod-template-hash label: %#v", pod.Name, pods)
		}
		// Save the first valid hash
		if len(hash) == 0 {
			hash = podHash
		}
		if podHash != hash {
			return "", fmt.Errorf("found pod %s with mismatching pod-template-hash value %s: %#v", pod.Name, podHash, pods)
		}
	}
	return hash, nil
}

// Deployment should have a timeout condition when it fails to progress after given deadline.
func TestFailedDeployment(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-failed-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "progress-check"
	replicas := int32(1)
	three := int32(3)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	tester.deployment.Spec.ProgressDeadlineSeconds = &three
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	if err = tester.waitForDeploymentUpdatedReplicasGTE(replicas); err != nil {
		t.Fatal(err)
	}

	// Pods are not marked as Ready, therefore the deployment progress will eventually timeout after progressDeadlineSeconds has passed.
	// Wait for the deployment to have a progress timeout condition.
	if err = tester.waitForDeploymentWithCondition(deploymentutil.TimedOutReason, apps.DeploymentProgressing); err != nil {
		t.Fatal(err)
	}

	// Manually mark pods as Ready and wait for deployment to complete.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q fails to have its status becoming valid: %v", deploymentName, err)
	}

	// Wait for the deployment to have a progress complete condition.
	if err = tester.waitForDeploymentWithCondition(deploymentutil.NewRSAvailableReason, apps.DeploymentProgressing); err != nil {
		t.Fatal(err)
	}
}

func TestOverlappingDeployments(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-overlapping-deployments"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	replicas := int32(1)
	firstDeploymentName := "first-deployment"
	secondDeploymentName := "second-deployment"
	testers := []*deploymentTester{
		{t: t, c: c, deployment: newDeployment(firstDeploymentName, ns.Name, replicas)},
		{t: t, c: c, deployment: newDeployment(secondDeploymentName, ns.Name, replicas)},
	}
	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Create 2 deployments with overlapping selectors
	var err error
	var rss []*apps.ReplicaSet
	for _, tester := range testers {
		tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
		dname := tester.deployment.Name
		if err != nil {
			t.Fatalf("failed to create deployment %q: %v", dname, err)
		}
		// Wait for the deployment to be updated to revision 1
		if err = tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
			t.Fatalf("failed to update deployment %q to revision 1: %v", dname, err)
		}
		// Make sure the deployment completes while manually marking its pods as ready at the same time
		if err = tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
			t.Fatalf("deployment %q failed to complete: %v", dname, err)
		}
		// Get replicaset of the deployment
		newRS, err := tester.getNewReplicaSet()
		if err != nil {
			t.Fatalf("failed to get new replicaset of deployment %q: %v", dname, err)
		}
		if newRS == nil {
			t.Fatalf("unable to find new replicaset of deployment %q", dname)
		}
		// Store the replicaset for future usage
		rss = append(rss, newRS)
	}

	// Both deployments should proceed independently, so their respective replicaset should not be the same replicaset
	if rss[0].UID == rss[1].UID {
		t.Fatalf("overlapping deployments should not share the same replicaset")
	}

	// Scale only the first deployment by 1
	newReplicas := replicas + 1
	testers[0].deployment, err = testers[0].updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = &newReplicas
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", firstDeploymentName, err)
	}

	// Make sure the deployment completes after scaling
	if err := testers[0].waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q failed to complete after scaling: %v", firstDeploymentName, err)
	}

	// Verify replicaset of both deployments has updated number of replicas
	for i, tester := range testers {
		rs, err := c.AppsV1().ReplicaSets(ns.Name).Get(rss[i].Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get replicaset %q: %v", rss[i].Name, err)
		}
		if *rs.Spec.Replicas != *tester.deployment.Spec.Replicas {
			t.Errorf("expected replicaset %q of deployment %q has %d replicas, but found %d replicas", rs.Name, firstDeploymentName, *tester.deployment.Spec.Replicas, *rs.Spec.Replicas)
		}
	}
}

// Deployment should not block rollout when updating spec replica number and template at the same time.
func TestScaledRolloutDeployment(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-scaled-rollout-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Create a deployment with rolling update strategy, max surge = 3, and max unavailable = 2
	var err error
	replicas := int32(10)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	tester.deployment.Spec.Strategy.RollingUpdate.MaxSurge = intOrStrP(3)
	tester.deployment.Spec.Strategy.RollingUpdate.MaxUnavailable = intOrStrP(2)
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", name, err)
	}
	if err = tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}
	if err = tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q failed to complete: %v", name, err)
	}

	// Record current replicaset before starting new rollout
	firstRS, err := tester.expectNewReplicaSet()
	if err != nil {
		t.Fatal(err)
	}

	// Update the deployment with another new image but do not mark the pods as ready to block new replicaset
	fakeImage2 := "fakeimage2"
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = fakeImage2
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", name, err)
	}
	if err = tester.waitForDeploymentRevisionAndImage("2", fakeImage2); err != nil {
		t.Fatal(err)
	}

	// Verify the deployment has minimum available replicas after 2nd rollout
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get deployment %q: %v", name, err)
	}
	minAvailableReplicas := deploymentutil.MinAvailable(tester.deployment)
	if tester.deployment.Status.AvailableReplicas < minAvailableReplicas {
		t.Fatalf("deployment %q does not have minimum number of available replicas after 2nd rollout", name)
	}

	// Wait for old replicaset of 1st rollout to have desired replicas
	firstRS, err = c.AppsV1().ReplicaSets(ns.Name).Get(firstRS.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get replicaset %q: %v", firstRS.Name, err)
	}
	if err = tester.waitRSStable(firstRS); err != nil {
		t.Fatal(err)
	}

	// Wait for new replicaset of 2nd rollout to have desired replicas
	secondRS, err := tester.expectNewReplicaSet()
	if err != nil {
		t.Fatal(err)
	}
	if err = tester.waitRSStable(secondRS); err != nil {
		t.Fatal(err)
	}

	// Scale up the deployment and update its image to another new image simultaneously (this time marks all pods as ready)
	newReplicas := int32(20)
	fakeImage3 := "fakeimage3"
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = fakeImage3
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", name, err)
	}
	if err = tester.waitForDeploymentRevisionAndImage("3", fakeImage3); err != nil {
		t.Fatal(err)
	}
	if err = tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q failed to complete: %v", name, err)
	}

	// Verify every replicaset has correct desiredReplicas annotation after 3rd rollout
	thirdRS, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed getting new revision 3 replicaset for deployment %q: %v", name, err)
	}
	rss := []*apps.ReplicaSet{firstRS, secondRS, thirdRS}
	for _, curRS := range rss {
		curRS, err = c.AppsV1().ReplicaSets(ns.Name).Get(curRS.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get replicaset when checking desired replicas annotation: %v", err)
		}
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(curRS)
		if !ok {
			t.Fatalf("failed to retrieve desiredReplicas annotation for replicaset %q", curRS.Name)
		}
		if desired != *(tester.deployment.Spec.Replicas) {
			t.Fatalf("unexpected desiredReplicas annotation for replicaset %q: expected %d, got %d", curRS.Name, *(tester.deployment.Spec.Replicas), desired)
		}
	}

	// Update the deployment with another new image but do not mark the pods as ready to block new replicaset
	fakeImage4 := "fakeimage4"
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Template.Spec.Containers[0].Image = fakeImage4
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", name, err)
	}
	if err = tester.waitForDeploymentRevisionAndImage("4", fakeImage4); err != nil {
		t.Fatal(err)
	}

	// Verify the deployment has minimum available replicas after 4th rollout
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get deployment %q: %v", name, err)
	}
	minAvailableReplicas = deploymentutil.MinAvailable(tester.deployment)
	if tester.deployment.Status.AvailableReplicas < minAvailableReplicas {
		t.Fatalf("deployment %q does not have minimum number of available replicas after 4th rollout", name)
	}

	// Wait for old replicaset of 3rd rollout to have desired replicas
	thirdRS, err = c.AppsV1().ReplicaSets(ns.Name).Get(thirdRS.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get replicaset %q: %v", thirdRS.Name, err)
	}
	if err = tester.waitRSStable(thirdRS); err != nil {
		t.Fatal(err)
	}

	// Wait for new replicaset of 4th rollout to have desired replicas
	fourthRS, err := tester.expectNewReplicaSet()
	if err != nil {
		t.Fatal(err)
	}
	if err = tester.waitRSStable(fourthRS); err != nil {
		t.Fatal(err)
	}

	// Scale down the deployment and update its image to another new image simultaneously (this time marks all pods as ready)
	newReplicas = int32(5)
	fakeImage5 := "fakeimage5"
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = &newReplicas
		update.Spec.Template.Spec.Containers[0].Image = fakeImage5
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", name, err)
	}
	if err = tester.waitForDeploymentRevisionAndImage("5", fakeImage5); err != nil {
		t.Fatal(err)
	}
	if err = tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q failed to complete: %v", name, err)
	}

	// Verify every replicaset has correct desiredReplicas annotation after 5th rollout
	fifthRS, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed getting new revision 5 replicaset for deployment %q: %v", name, err)
	}
	rss = []*apps.ReplicaSet{thirdRS, fourthRS, fifthRS}
	for _, curRS := range rss {
		curRS, err = c.AppsV1().ReplicaSets(ns.Name).Get(curRS.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get replicaset when checking desired replicas annotation: %v", err)
		}
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(curRS)
		if !ok {
			t.Fatalf("failed to retrieve desiredReplicas annotation for replicaset %q", curRS.Name)
		}
		if desired != *(tester.deployment.Spec.Replicas) {
			t.Fatalf("unexpected desiredReplicas annotation for replicaset %q: expected %d, got %d", curRS.Name, *(tester.deployment.Spec.Replicas), desired)
		}
	}
}

func TestSpecReplicasChange(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-spec-replicas-change"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "deployment"
	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	tester.deployment.Spec.Strategy.Type = apps.RecreateDeploymentStrategyType
	tester.deployment.Spec.Strategy.RollingUpdate = nil
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Scale up/down deployment and verify its replicaset has matching .spec.replicas
	if err = tester.scaleDeployment(2); err != nil {
		t.Fatal(err)
	}
	if err = tester.scaleDeployment(0); err != nil {
		t.Fatal(err)
	}
	if err = tester.scaleDeployment(1); err != nil {
		t.Fatal(err)
	}

	// Add a template annotation change to test deployment's status does update
	// without .spec.replicas change
	var oldGeneration int64
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		oldGeneration = update.Generation
		update.Spec.RevisionHistoryLimit = pointer.Int32Ptr(4)
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", tester.deployment.Name, err)
	}

	savedGeneration := tester.deployment.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("Failed to verify .Generation has incremented for deployment %q", deploymentName)
	}
	if err = tester.waitForObservedDeployment(savedGeneration); err != nil {
		t.Fatal(err)
	}
}

func TestDeploymentAvailableCondition(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-deployment-available-condition"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "deployment"
	replicas := int32(10)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	// Assign a high value to the deployment's minReadySeconds
	tester.deployment.Spec.MinReadySeconds = 3600
	// progressDeadlineSeconds must be greater than minReadySeconds
	tester.deployment.Spec.ProgressDeadlineSeconds = pointer.Int32Ptr(7200)
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the deployment to be observed by the controller and has at least specified number of updated replicas
	if err = tester.waitForDeploymentUpdatedReplicasGTE(replicas); err != nil {
		t.Fatal(err)
	}

	// Wait for the deployment to have MinimumReplicasUnavailable reason because the pods are not marked as ready
	if err = tester.waitForDeploymentWithCondition(deploymentutil.MinimumReplicasUnavailable, apps.DeploymentAvailable); err != nil {
		t.Fatal(err)
	}

	// Verify all replicas fields of DeploymentStatus have desired counts
	if err = tester.checkDeploymentStatusReplicasFields(10, 10, 0, 0, 10); err != nil {
		t.Fatal(err)
	}

	// Mark the pods as ready without waiting for the deployment to complete
	if err = tester.markUpdatedPodsReadyWithoutComplete(); err != nil {
		t.Fatal(err)
	}

	// Wait for number of ready replicas to equal number of replicas.
	if err = tester.waitForReadyReplicas(); err != nil {
		t.Fatal(err)
	}

	// Wait for the deployment to still have MinimumReplicasUnavailable reason within minReadySeconds period
	if err = tester.waitForDeploymentWithCondition(deploymentutil.MinimumReplicasUnavailable, apps.DeploymentAvailable); err != nil {
		t.Fatal(err)
	}

	// Verify all replicas fields of DeploymentStatus have desired counts
	if err = tester.checkDeploymentStatusReplicasFields(10, 10, 10, 0, 10); err != nil {
		t.Fatal(err)
	}

	// Update the deployment's minReadySeconds to a small value
	tester.deployment, err = tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.MinReadySeconds = 1
	})
	if err != nil {
		t.Fatalf("failed updating deployment %q: %v", deploymentName, err)
	}

	// Wait for the deployment to notice minReadySeconds has changed
	if err := tester.waitForObservedDeployment(tester.deployment.Generation); err != nil {
		t.Fatal(err)
	}

	// Wait for the deployment to have MinimumReplicasAvailable reason after minReadySeconds period
	if err = tester.waitForDeploymentWithCondition(deploymentutil.MinimumReplicasAvailable, apps.DeploymentAvailable); err != nil {
		t.Fatal(err)
	}

	// Verify all replicas fields of DeploymentStatus have desired counts
	if err = tester.checkDeploymentStatusReplicasFields(10, 10, 10, 10, 0); err != nil {
		t.Fatal(err)
	}
}

// Wait for deployment to automatically patch incorrect ControllerRef of RS
func testRSControllerRefPatch(t *testing.T, tester *deploymentTester, rs *apps.ReplicaSet, ownerReference *metav1.OwnerReference, expectedOwnerReferenceNum int) {
	ns := rs.Namespace
	rsClient := tester.c.AppsV1().ReplicaSets(ns)
	rs, err := tester.updateReplicaSet(rs.Name, func(update *apps.ReplicaSet) {
		update.OwnerReferences = []metav1.OwnerReference{*ownerReference}
	})
	if err != nil {
		t.Fatalf("failed to update replicaset %q: %v", rs.Name, err)
	}

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return metav1.GetControllerOf(newRS) != nil, nil
	}); err != nil {
		t.Fatalf("failed to wait for controllerRef of the replicaset %q to become nil: %v", rs.Name, err)
	}

	newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to obtain replicaset %q: %v", rs.Name, err)
	}
	controllerRef := metav1.GetControllerOf(newRS)
	if controllerRef.UID != tester.deployment.UID {
		t.Fatalf("controllerRef of replicaset %q has a different UID: Expected %v, got %v", newRS.Name, tester.deployment.UID, controllerRef.UID)
	}
	ownerReferenceNum := len(newRS.GetOwnerReferences())
	if ownerReferenceNum != expectedOwnerReferenceNum {
		t.Fatalf("unexpected number of owner references for replicaset %q: Expected %d, got %d", newRS.Name, expectedOwnerReferenceNum, ownerReferenceNum)
	}
}

func TestGeneralReplicaSetAdoption(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-general-replicaset-adoption"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "deployment"
	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Ensure the deployment completes while marking its pods as ready simultaneously
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// Get replicaset of the deployment
	rs, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed to get replicaset of deployment %q: %v", deploymentName, err)
	}
	if rs == nil {
		t.Fatalf("unable to find replicaset of deployment %q", deploymentName)
	}

	// When the only OwnerReference of the RS points to another type of API object such as statefulset
	// with Controller=false, the deployment should add a second OwnerReference (ControllerRef) pointing to itself
	// with Controller=true
	var falseVar = false
	ownerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "apps/v1beta1", Kind: "StatefulSet", Name: deploymentName, Controller: &falseVar}
	testRSControllerRefPatch(t, tester, rs, &ownerReference, 2)

	// When the only OwnerReference of the RS points to the deployment with Controller=false,
	// the deployment should set Controller=true for the only OwnerReference
	ownerReference = metav1.OwnerReference{UID: tester.deployment.UID, APIVersion: "extensions/v1beta1", Kind: "Deployment", Name: deploymentName, Controller: &falseVar}
	testRSControllerRefPatch(t, tester, rs, &ownerReference, 1)
}

func testScalingUsingScaleSubresource(t *testing.T, tester *deploymentTester, replicas int32) {
	ns := tester.deployment.Namespace
	deploymentName := tester.deployment.Name
	deploymentClient := tester.c.AppsV1().Deployments(ns)
	deployment, err := deploymentClient.Get(deploymentName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain deployment %q: %v", deploymentName, err)
	}
	scale, err := tester.c.AppsV1().Deployments(ns).GetScale(deploymentName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain scale subresource for deployment %q: %v", deploymentName, err)
	}
	if scale.Spec.Replicas != *deployment.Spec.Replicas {
		t.Fatalf("Scale subresource for deployment %q does not match .Spec.Replicas: expected %d, got %d", deploymentName, *deployment.Spec.Replicas, scale.Spec.Replicas)
	}

	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		scale, err := tester.c.AppsV1().Deployments(ns).GetScale(deploymentName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		scale.Spec.Replicas = replicas
		_, err = tester.c.AppsV1().Deployments(ns).UpdateScale(deploymentName, scale)
		return err
	}); err != nil {
		t.Fatalf("Failed to set .Spec.Replicas of scale subresource for deployment %q: %v", deploymentName, err)
	}

	deployment, err = deploymentClient.Get(deploymentName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain deployment %q: %v", deploymentName, err)
	}
	if *deployment.Spec.Replicas != replicas {
		t.Fatalf(".Spec.Replicas of deployment %q does not match its scale subresource: expected %d, got %d", deploymentName, replicas, *deployment.Spec.Replicas)
	}
}

func TestDeploymentScaleSubresource(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-deployment-scale-subresource"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "deployment"
	replicas := int32(2)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Ensure the deployment completes while marking its pods as ready simultaneously
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// Use scale subresource to scale the deployment up to 3
	testScalingUsingScaleSubresource(t, tester, 3)
	// Use the scale subresource to scale the deployment down to 0
	testScalingUsingScaleSubresource(t, tester, 0)
}

// This test verifies that the Deployment does orphan a ReplicaSet when the ReplicaSet's
// .Labels field is changed to no longer match the Deployment's selector. It also partially
// verifies that collision avoidance mechanism is triggered when a Deployment's new ReplicaSet
// is orphaned, even without PodTemplateSpec change. Refer comment below for more info:
// https://github.com/kubernetes/kubernetes/pull/59212#discussion_r166465113
func TestReplicaSetOrphaningAndAdoptionWhenLabelsChange(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-replicaset-orphaning-and-adoption-when-labels-change"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	deploymentName := "deployment"
	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(deploymentName, ns.Name, replicas)}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	if err := tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// Ensure the deployment completes while marking its pods as ready simultaneously
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatal(err)
	}

	// Orphaning: deployment should remove OwnerReference from a RS when the RS's labels change to not match its labels

	// Get replicaset of the deployment
	rs, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
	if err != nil {
		t.Fatalf("failed to get replicaset of deployment %q: %v", deploymentName, err)
	}
	if rs == nil {
		t.Fatalf("unable to find replicaset of deployment %q", deploymentName)
	}

	// Verify controllerRef of the replicaset is not nil and pointing to the deployment
	controllerRef := metav1.GetControllerOf(rs)
	if controllerRef == nil {
		t.Fatalf("controllerRef of replicaset %q is nil", rs.Name)
	}
	if controllerRef.UID != tester.deployment.UID {
		t.Fatalf("controllerRef of replicaset %q has a different UID: Expected %v, got %v", rs.Name, tester.deployment.UID, controllerRef.UID)
	}

	// Change the replicaset's labels to not match the deployment's labels
	labelMap := map[string]string{"new-name": "new-test"}
	rs, err = tester.updateReplicaSet(rs.Name, func(update *apps.ReplicaSet) {
		update.Labels = labelMap
	})
	if err != nil {
		t.Fatalf("failed to update replicaset %q: %v", rs.Name, err)
	}

	// Wait for the controllerRef of the replicaset to become nil
	rsClient := tester.c.AppsV1().ReplicaSets(ns.Name)
	if err = wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		rs, err = rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return metav1.GetControllerOf(rs) == nil, nil
	}); err != nil {
		t.Fatalf("failed to wait for controllerRef of replicaset %q to become nil: %v", rs.Name, err)
	}

	// Wait for the deployment to create a new replicaset
	// This will trigger collision avoidance due to deterministic nature of replicaset name
	// i.e., the new replicaset will have a name with different hash to preserve name uniqueness
	var newRS *apps.ReplicaSet
	if err = wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newRS, err = deploymentutil.GetNewReplicaSet(tester.deployment, c.AppsV1())
		if err != nil {
			return false, fmt.Errorf("failed to get new replicaset of deployment %q after orphaning: %v", deploymentName, err)
		}
		return newRS != nil, nil
	}); err != nil {
		t.Fatalf("failed to wait for deployment %q to create a new replicaset after orphaning: %v", deploymentName, err)
	}
	if newRS.UID == rs.UID {
		t.Fatalf("expect deployment %q to create a new replicaset different from the orphaned one, but it isn't", deploymentName)
	}

	// Adoption: deployment should add controllerRef to a RS when the RS's labels change to match its labels

	// Change the old replicaset's labels to match the deployment's labels
	rs, err = tester.updateReplicaSet(rs.Name, func(update *apps.ReplicaSet) {
		update.Labels = testLabels()
	})
	if err != nil {
		t.Fatalf("failed to update replicaset %q: %v", rs.Name, err)
	}

	// Wait for the deployment to adopt the old replicaset
	if err = wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		rs, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		controllerRef = metav1.GetControllerOf(rs)
		return controllerRef != nil && controllerRef.UID == tester.deployment.UID, nil
	}); err != nil {
		t.Fatalf("failed waiting for replicaset adoption by deployment %q to complete: %v", deploymentName, err)
	}
}
