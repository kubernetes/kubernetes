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

package federation

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/integration/federation/fixture"
)

func TestFederationCRUD(t *testing.T) {
	fedFixture := fixture.FederationFixture{ClusterCount: 2}
	fedFixture.Setup(t)
	defer fedFixture.Teardown(t)

	resourceFixtures := []fixture.ResourceFixture{
		&fixture.SecretControllerFixture{},
		&fixture.ConfigMapControllerFixture{},
	}
	for _, fixture := range resourceFixtures {
		kind := fixture.Kind()
		t.Run(kind, func(t *testing.T) {
			testConfig := fedFixture.NewRestConfig(fmt.Sprintf("test-%s", kind))
			testClient := federationclientset.NewForConfigOrDie(testConfig)
			controllerConfig := fedFixture.NewRestConfig(fmt.Sprintf("controller-%s", kind))
			controllerClient := federationclientset.NewForConfigOrDie(controllerConfig)
			fixture.Setup(t, testClient, controllerClient)
			defer fixture.Teardown(t)
			validateCRUDForKind(t, fixture, fedFixture.Clusters)
		})
	}
}

func validateCRUDForKind(t *testing.T, f fixture.ResourceFixture, clusters []*fixture.MemberCluster) {
	adapter := f.GetAdapter()
	kind := adapter.Kind()
	namespace := fmt.Sprintf("validate-crud-%s", kind)

	// Create federated object
	obj, err := adapter.FedCreate(f.New(namespace))
	if err != nil {
		t.Fatalf("Error creating federated %s: %v", kind, err)
	}

	// Check that the object is propagated to member clusters
	for _, cluster := range clusters {
		err = waitForResource(adapter, cluster.Client, obj, 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}

	nsName := adapter.GetNamespacedName(obj)

	// The resource will have been updated to include deletion finalizers as
	// part of propagation.  It is necessary to get the latest version to avoid
	// a conflict on update.
	objWithFinalizers, err := adapter.FedGet(nsName)
	if err != nil {
		t.Fatal(err)
	}

	// Update the federated object
	mutatedObj := f.Mutate(objWithFinalizers)
	updatedObj, err := adapter.FedUpdate(mutatedObj)
	if err != nil {
		t.Fatalf("Error updating federated %s: %v", kind, err)
	}

	// Check that the object is updated in member clusters
	for _, cluster := range clusters {
		err = waitForResource(adapter, cluster.Client, updatedObj, 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Delete the federated object
	orphanDependents := false
	err = adapter.FedDelete(nsName, &metav1.DeleteOptions{OrphanDependents: &orphanDependents})
	if err != nil {
		t.Fatalf("Error deleting federated %s: %v", kind, err)
	}

	// Check that the object is removed from member clusters
	for _, cluster := range clusters {
		err = waitForDeletion(adapter, cluster.Client, nsName, 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func waitForResource(adapter fixture.ResourceAdapter, client clientset.Interface, obj pkgruntime.Object, timeout time.Duration) error {
	nsName := adapter.GetNamespacedName(obj)
	err := wait.PollImmediate(2*time.Second, timeout, func() (bool, error) {
		clusterObj, err := adapter.Get(client, nsName)
		if err == nil && adapter.Equivalent(clusterObj, obj) {
			return true, nil
		} else if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	})
	return err
}

func waitForDeletion(adapter fixture.ResourceAdapter, client clientset.Interface, nsName types.NamespacedName, timeout time.Duration) error {
	err := wait.PollImmediate(2*time.Second, timeout, func() (bool, error) {
		_, err := adapter.Get(client, nsName)
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	return err
}
