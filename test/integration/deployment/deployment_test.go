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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	deploy, err := c.Extensions().Deployments(ns.Name).Create(tester.deployment)
	if err != nil {
		t.Fatalf("failed to create deployment %s: %v", deploy.Name, err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	go dc.Run(5, stopCh)

	// Wait for the Deployment to be updated to revision 1
	err = tester.waitForDeploymentRevisionAndImage("1", fakeImage)
	if err != nil {
		t.Fatalf("failed to wait for Deployment revision %s: %v", deploy.Name, err)
	}

	// Make sure the Deployment status becomes valid while manually marking Deployment pods as ready at the same time
	tester.waitForDeploymentStatusValidAndMarkPodsReady()

	// Check new RS annotations
	newRS, err := deploymentutil.GetNewReplicaSet(deploy, c.ExtensionsV1beta1())
	if err != nil {
		t.Fatalf("failed to get new ReplicaSet of Deployment %s: %v", deploy.Name, err)
	}
	if newRS.Annotations["test"] != "should-copy-to-replica-set" {
		t.Errorf("expected new ReplicaSet annotations copied from Deployment %s, got: %v", deploy.Name, newRS.Annotations)
	}
	if newRS.Annotations[v1.LastAppliedConfigAnnotation] != "" {
		t.Errorf("expected new ReplicaSet last-applied annotation not copied from Deployment %s", deploy.Name)
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
