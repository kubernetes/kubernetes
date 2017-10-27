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
	_, err := c.ExtensionsV1beta1().ReplicaSets(ns.Name).Create(rs)
	if err != nil {
		t.Fatalf("failed to create replicaset %s: %v", rsName, err)
	}

	replicas := int32(1)
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	oriImage := tester.deployment.Spec.Template.Spec.Containers[0].Image

	// Create a deployment which have different template than the replica set created above.
	if tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment); err != nil {
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
	tester.deployment, err = tester.updateDeployment(func(update *v1beta1.Deployment) {
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

func checkRSHashLabels(rs *v1beta1.ReplicaSet) (string, error) {
	hash := rs.Labels[v1beta1.DefaultDeploymentUniqueLabelKey]
	selectorHash := rs.Spec.Selector.MatchLabels[v1beta1.DefaultDeploymentUniqueLabelKey]
	templateLabelHash := rs.Spec.Template.Labels[v1beta1.DefaultDeploymentUniqueLabelKey]

	if hash != selectorHash || selectorHash != templateLabelHash {
		return "", fmt.Errorf("mismatching hash value found in replicaset %s: %#v", rs.Name, rs)
	}
	if len(hash) == 0 {
		return "", fmt.Errorf("unexpected replicaset %s missing required pod-template-hash labels", rs.Name)
	}

	return hash, nil
}

func checkPodsHashLabel(pods *v1.PodList) (string, error) {
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no pods given")
	}
	var hash string
	for _, pod := range pods.Items {
		podHash := pod.Labels[v1beta1.DefaultDeploymentUniqueLabelKey]
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

// Deployment should label adopted ReplicaSets and Pods.
func TestDeploymentLabelAdopted(t *testing.T) {
	s, closeFn, rm, dc, informers, c := dcSetup(t)
	defer closeFn()
	name := "test-adopted-deployment"
	ns := framework.CreateTestingNamespace(name, s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Create a RS to be adopted by the deployment.
	rsName := "test-adopted-controller"
	replicas := int32(1)
	rs := newReplicaSet(rsName, ns.Name, replicas)
	_, err := c.ExtensionsV1beta1().ReplicaSets(ns.Name).Create(rs)
	if err != nil {
		t.Fatalf("failed to create replicaset %s: %v", rsName, err)
	}
	// Mark RS pods as ready.
	selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	if err != nil {
		t.Fatalf("failed to parse replicaset %s selector: %v", rsName, err)
	}
	if err = wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		pods, err := c.CoreV1().Pods(ns.Name).List(metav1.ListOptions{LabelSelector: selector.String()})
		if err != nil {
			return false, err
		}
		if len(pods.Items) != int(replicas) {
			return false, nil
		}
		for _, pod := range pods.Items {
			if err = markPodReady(c, ns.Name, &pod); err != nil {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to mark pods replicaset %s as ready: %v", rsName, err)
	}

	// Create a Deployment to adopt the old rs.
	tester := &deploymentTester{t: t, c: c, deployment: newDeployment(name, ns.Name, replicas)}
	if tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment); err != nil {
		t.Fatalf("failed to create deployment %s: %v", tester.deployment.Name, err)
	}

	// Wait for the Deployment to be updated to revision 1
	if err = tester.waitForDeploymentRevisionAndImage("1", fakeImage); err != nil {
		t.Fatal(err)
	}

	// The RS and pods should be relabeled after the Deployment finishes adopting it and completes.
	if err := tester.waitForDeploymentComplete(); err != nil {
		t.Fatal(err)
	}

	// There should be no old RSes (overlapping RS)
	oldRSs, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(tester.deployment, c.ExtensionsV1beta1())
	if err != nil {
		t.Fatalf("failed to get all replicasets owned by deployment %s: %v", name, err)
	}
	if len(oldRSs) != 0 || len(allOldRSs) != 0 {
		t.Errorf("expected deployment to have no old replicasets, got %d old replicasets", len(allOldRSs))
	}

	// New RS should be relabeled, i.e. contain pod-template-hash in its selector, label, and template label
	rsHash, err := checkRSHashLabels(newRS)
	if err != nil {
		t.Error(err)
	}

	// All pods targeted by the deployment should contain pod-template-hash in their labels, and there should be only 3 pods
	selector, err = metav1.LabelSelectorAsSelector(tester.deployment.Spec.Selector)
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
	tester.deployment, err = c.ExtensionsV1beta1().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %q: %v", deploymentName, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	if err = tester.waitForDeploymentUpdatedReplicasLTE(replicas); err != nil {
		t.Fatal(err)
	}

	// Pods are not marked as Ready, therefore the deployment progress will eventually timeout after progressDeadlineSeconds has passed.
	// Wait for the deployment to have a progress timeout condition.
	if err = tester.waitForDeploymentWithCondition(deploymentutil.TimedOutReason, v1beta1.DeploymentProgressing); err != nil {
		t.Fatal(err)
	}

	// Manually mark pods as Ready and wait for deployment to complete.
	if err := tester.waitForDeploymentCompleteAndMarkPodsReady(); err != nil {
		t.Fatalf("deployment %q fails to have its status becoming valid: %v", deploymentName, err)
	}

	// Wait for the deployment to have a progress complete condition.
	if err = tester.waitForDeploymentWithCondition(deploymentutil.NewRSAvailableReason, v1beta1.DeploymentProgressing); err != nil {
		t.Fatal(err)
	}
}
