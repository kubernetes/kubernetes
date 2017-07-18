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

package crudtester

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
)

const (
	AnnotationTestFederationCRUDUpdate string = "federation.kubernetes.io/test-federation-crud-update"
)

// TestLogger defines operations common across different types of testing
type TestLogger interface {
	Fatalf(format string, args ...interface{})
	Fatal(msg string)
	Logf(format string, args ...interface{})
}

// FederatedTypeCRUDTester exercises Create/Read/Update/Delete operations for
// federated types via the Federation API and validates that the
// results of those operations are propagated to clusters that are
// members of a federation.
type FederatedTypeCRUDTester struct {
	tl             TestLogger
	adapter        federatedtypes.FederatedTypeAdapter
	kind           string
	clusterClients []clientset.Interface
	waitInterval   time.Duration
	// Federation operations will use wait.ForeverTestTimeout.  Any
	// operation that involves member clusters may take longer due to
	// propagation latency.
	clusterWaitTimeout time.Duration
}

func NewFederatedTypeCRUDTester(testLogger TestLogger, adapter federatedtypes.FederatedTypeAdapter, clusterClients []clientset.Interface, waitInterval, clusterWaitTimeout time.Duration) *FederatedTypeCRUDTester {
	return &FederatedTypeCRUDTester{
		tl:                 testLogger,
		adapter:            adapter,
		kind:               adapter.Kind(),
		clusterClients:     clusterClients,
		waitInterval:       waitInterval,
		clusterWaitTimeout: clusterWaitTimeout,
	}
}

func (c *FederatedTypeCRUDTester) CheckLifecycle(desiredObject pkgruntime.Object) {
	obj := c.CheckCreate(desiredObject)
	c.CheckUpdate(obj)

	// Validate the golden path - removal of dependents
	orphanDependents := false
	c.CheckDelete(obj, &orphanDependents)
}

func (c *FederatedTypeCRUDTester) Create(desiredObject pkgruntime.Object) pkgruntime.Object {
	namespace := c.adapter.ObjectMeta(desiredObject).Namespace
	resourceMsg := fmt.Sprintf("federated %s", c.kind)
	if len(namespace) > 0 {
		resourceMsg = fmt.Sprintf("%s in namespace %q", resourceMsg, namespace)
	}

	c.tl.Logf("Creating new %s", resourceMsg)

	obj, err := c.adapter.FedCreate(desiredObject)
	if err != nil {
		c.tl.Fatalf("Error creating %s: %v", resourceMsg, err)
	}

	qualifiedName := c.adapter.QualifiedName(obj)
	c.tl.Logf("Created new federated %s %q", c.kind, qualifiedName)

	return obj
}

func (c *FederatedTypeCRUDTester) CheckCreate(desiredObject pkgruntime.Object) pkgruntime.Object {
	obj := c.Create(desiredObject)

	c.CheckPropagation(obj)

	return obj
}

func (c *FederatedTypeCRUDTester) CheckUpdate(obj pkgruntime.Object) {
	qualifiedName := c.adapter.QualifiedName(obj)

	var initialAnnotation string
	meta := c.adapter.ObjectMeta(obj)
	if meta.Annotations != nil {
		initialAnnotation = meta.Annotations[AnnotationTestFederationCRUDUpdate]
	}

	c.tl.Logf("Updating federated %s %q", c.kind, qualifiedName)
	updatedObj, err := c.updateFedObject(obj)
	if err != nil {
		c.tl.Fatalf("Error updating federated %s %q: %v", c.kind, qualifiedName, err)
	}

	// updateFedObject is expected to have changed the value of the annotation
	meta = c.adapter.ObjectMeta(updatedObj)
	updatedAnnotation := meta.Annotations[AnnotationTestFederationCRUDUpdate]
	if updatedAnnotation == initialAnnotation {
		c.tl.Fatalf("Federated %s %q not mutated", c.kind, qualifiedName)
	}

	c.CheckPropagation(updatedObj)
}

func (c *FederatedTypeCRUDTester) CheckDelete(obj pkgruntime.Object, orphanDependents *bool) {
	qualifiedName := c.adapter.QualifiedName(obj)

	c.tl.Logf("Deleting federated %s %q", c.kind, qualifiedName)
	err := c.adapter.FedDelete(qualifiedName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil {
		c.tl.Fatalf("Error deleting federated %s %q: %v", c.kind, qualifiedName, err)
	}

	deletingInCluster := (orphanDependents != nil && *orphanDependents == false)

	waitTimeout := wait.ForeverTestTimeout
	if deletingInCluster {
		// May need extra time to delete both federation and cluster resources
		waitTimeout = c.clusterWaitTimeout
	}

	// Wait for deletion.  The federation resource will only be removed once orphan deletion has been
	// completed or deemed unnecessary.
	err = wait.PollImmediate(c.waitInterval, waitTimeout, func() (bool, error) {
		_, err := c.adapter.FedGet(qualifiedName)
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		c.tl.Fatalf("Error deleting federated %s %q: %v", c.kind, qualifiedName, err)
	}

	var stateMsg string = "present"
	if deletingInCluster {
		stateMsg = "not present"
	}
	for _, client := range c.clusterClients {
		_, err := c.adapter.ClusterGet(client, qualifiedName)
		switch {
		case !deletingInCluster && errors.IsNotFound(err):
			c.tl.Fatalf("Federated %s %q was unexpectedly deleted from a member cluster", c.kind, qualifiedName)
		case deletingInCluster && err == nil:
			c.tl.Fatalf("Federated %s %q was unexpectedly orphaned in a member cluster", c.kind, qualifiedName)
		case err != nil && !errors.IsNotFound(err):
			c.tl.Fatalf("Error while checking whether %s %q is %s in member clusters: %v", c.kind, qualifiedName, stateMsg, err)
		}
	}
}

// CheckPropagation checks propagation for the crud tester's clients
func (c *FederatedTypeCRUDTester) CheckPropagation(obj pkgruntime.Object) {
	c.CheckPropagationForClients(obj, c.clusterClients, true)
}

// CheckPropagationForClients checks propagation for the provided clients
func (c *FederatedTypeCRUDTester) CheckPropagationForClients(obj pkgruntime.Object, clusterClients []clientset.Interface, objExpected bool) {
	qualifiedName := c.adapter.QualifiedName(obj)

	c.tl.Logf("Waiting for %s %q in %d clusters", c.kind, qualifiedName, len(clusterClients))
	for _, client := range clusterClients {
		err := c.waitForResource(client, obj)
		switch {
		case err == wait.ErrWaitTimeout:
			if objExpected {
				c.tl.Fatalf("Timeout verifying %s %q in a member cluster: %v", c.kind, qualifiedName, err)
			}
		case err != nil:
			c.tl.Fatalf("Failed to verify %s %q in a member cluster: %v", c.kind, qualifiedName, err)
		case err == nil && !objExpected:
			c.tl.Fatalf("Found unexpected object %s %q in a member cluster: %v", c.kind, qualifiedName, err)
		}
	}
}

func (c *FederatedTypeCRUDTester) waitForResource(client clientset.Interface, obj pkgruntime.Object) error {
	qualifiedName := c.adapter.QualifiedName(obj)
	err := wait.PollImmediate(c.waitInterval, c.clusterWaitTimeout, func() (bool, error) {
		equivalenceFunc := c.adapter.Equivalent
		if c.adapter.IsSchedulingAdapter() {
			schedulingAdapter, ok := c.adapter.(federatedtypes.SchedulingAdapter)
			if !ok {
				c.tl.Fatalf("Adapter for kind %q does not properly implement SchedulingAdapter.", c.adapter.Kind())
			}
			equivalenceFunc = schedulingAdapter.EquivalentIgnoringSchedule
		}

		clusterObj, err := c.adapter.ClusterGet(client, qualifiedName)
		if err == nil && equivalenceFunc(clusterObj, obj) {
			return true, nil
		}
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	})
	return err
}

func (c *FederatedTypeCRUDTester) updateFedObject(obj pkgruntime.Object) (pkgruntime.Object, error) {
	err := wait.PollImmediate(c.waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		// Target the metadata for simplicity (it's type-agnostic)
		federatedtypes.SetAnnotation(c.adapter, obj, AnnotationTestFederationCRUDUpdate, "updated")

		_, err := c.adapter.FedUpdate(obj)
		if errors.IsConflict(err) {
			// The resource was updated by the federation controller.
			// Get the latest version and retry.
			qualifiedName := c.adapter.QualifiedName(obj)
			obj, err = c.adapter.FedGet(qualifiedName)
			return false, err
		}
		// Be tolerant of a slow server
		if errors.IsServerTimeout(err) {
			return false, nil
		}
		return (err == nil), err
	})
	return obj, err
}
