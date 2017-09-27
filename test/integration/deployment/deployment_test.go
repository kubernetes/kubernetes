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
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/integration/framework"
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
	tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
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

	// Make sure the Deployment status becomes valid while manually marking Deployment pods as ready at the same time
	if err := tester.waitForDeploymentStatusValidAndMarkPodsReady(); err != nil {
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
}

// selectors are IMMUTABLE for all API versions except apps/v1beta1 and extensions/v1beta1
func TestDeploymentSelectorImmutability(t *testing.T) {
	s, closeFn, c := dcSimpleSetup(t)
	defer closeFn()
	name := "test-deployment-selector-immutability"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, int32(20))}
	deploymentExtensionsV1beta1, err := c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create extensions/v1beta1 deployment %s: %v", tester.deployment.Name, err)
	}

	// test to ensure extensions/v1beta1 selector is mutable
	newSelectorLabels := map[string]string{"name_extensions_v1beta1": "test_extensions_v1beta1"}
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
	tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
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

	// Make sure the Deployment status becomes valid while manually marking Deployment pods as ready at the same time
	if err := tester.waitForDeploymentStatusValidAndMarkPodsReady(); err != nil {
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
	tester.deployment, err = tester.updateDeployment(func(update *v1beta1.Deployment) {
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

	_, allOldRs, err := deploymentutil.GetOldReplicaSets(tester.deployment, c.ExtensionsV1beta1())
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
	tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
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

	// Make sure the Deployment status becomes valid while manually marking Deployment pods as ready at the same time
	if err := tester.waitForDeploymentStatusValidAndMarkPodsReady(); err != nil {
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
	tester.deployment, err = tester.updateDeployment(func(update *v1beta1.Deployment) {
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

	// Make sure the Deployment status becomes valid while manually marking Deployment pods as ready at the same time
	if err := tester.waitForDeploymentStatusValidAndMarkPodsReady(); err != nil {
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
	tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
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
	newRS, err := deploymentutil.GetNewReplicaSet(tester.deployment, c.ExtensionsV1beta1())
	if err != nil {
		t.Fatalf("failed getting new replicaset of deployment %s: %v", tester.deployment.Name, err)
	}
	if newRS == nil {
		t.Fatalf("unable to find new replicaset of deployment %s", tester.deployment.Name)
	}
	_, err = tester.updateReplicaSet(newRS.Name, func(update *v1beta1.ReplicaSet) {
		*update.Spec.Template.Spec.TerminationGracePeriodSeconds = int64(5)
	})
	if err != nil {
		t.Fatalf("failed updating replicaset %s template: %v", newRS.Name, err)
	}

	// Expect deployment collision counter to increment
	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		d, err := c.ExtensionsV1beta1().Deployments(ns.Name).Get(tester.deployment.Name, metav1.GetOptions{})
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
