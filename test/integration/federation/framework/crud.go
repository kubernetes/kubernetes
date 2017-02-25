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

package framework

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

type CRUDHelper struct {
	t              TestLogger
	adapter        ResourceAdapter
	kind           string
	clusterClients []clientset.Interface
	waitInterval   time.Duration
	// Federation operations will use wait.ForeverTestTimeout.  Any
	// operation that involves member clusters may take longer due to
	// propagation latency.
	clusterWaitTimeout time.Duration
}

func NewCRUDHelperWithTimeout(logger TestLogger, adapter ResourceAdapter, clusterClients []clientset.Interface, waitInterval, clusterWaitTimeout time.Duration) *CRUDHelper {
	return &CRUDHelper{
		t:                  logger,
		adapter:            adapter,
		kind:               adapter.Kind(),
		clusterClients:     clusterClients,
		waitInterval:       waitInterval,
		clusterWaitTimeout: clusterWaitTimeout,
	}
}

func NewCRUDHelper(t *testing.T, adapter ResourceAdapter, clusterClients []clientset.Interface) *CRUDHelper {
	logger := &IntegrationLogger{t}
	return NewCRUDHelperWithTimeout(logger, adapter, clusterClients, DefaultWaitInterval, wait.ForeverTestTimeout)
}

func (c *CRUDHelper) CheckLifecycle(obj pkgruntime.Object) {
	updatedObj := c.CheckCreate(obj)
	c.CheckUpdate(updatedObj)

	// Validate the golden path - removal of dependents
	orphanDependents := false
	c.CheckDelete(updatedObj, &orphanDependents)
}

func (c *CRUDHelper) CheckCreate(obj pkgruntime.Object) pkgruntime.Object {
	nsName := c.adapter.NamespacedName(obj)

	c.t.Logf("Creating federated %s %q", c.kind, nsName)

	obj, err := c.adapter.FedCreate(obj)
	if err != nil {
		c.t.Fatalf("Error creating federated %s %q : %v", c.kind, nsName, err)
	}

	c.checkPropagation(obj)

	return obj
}

func (c *CRUDHelper) CheckUpdate(obj pkgruntime.Object) {
	nsName := c.adapter.NamespacedName(obj)

	// Insure that the provided object does not have labels, which
	// would break the simple check for mutation.
	meta := c.adapter.ObjectMeta(obj)
	if len(meta.Labels) != 0 {
		c.t.Fatalf("Expected federated %s %q to have been created without labels", c.kind, nsName)
	}

	c.t.Logf("Updating federated %s %q", c.kind, nsName)
	updatedObj, err := c.updateFedObject(obj)
	if err != nil {
		c.t.Fatalf("Error updating federated %s %q: %v", c.kind, nsName, err)
	}

	// updateFedObject is expected to have added a label
	meta = c.adapter.ObjectMeta(updatedObj)
	if len(meta.Labels) == 0 {
		c.t.Fatalf("Federated %s %q not mutated", c.kind, nsName)
	}

	c.checkPropagation(updatedObj)
}

func (c *CRUDHelper) CheckDelete(obj pkgruntime.Object, orphanDependents *bool) {
	nsName := c.adapter.NamespacedName(obj)

	c.t.Logf("Deleting federated %s %q", c.kind, nsName)
	err := c.adapter.FedDelete(nsName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil {
		c.t.Fatalf("Error deleting federated %s %q: %v", c.kind, nsName, err)
	}

	// Wait for deletion.  The federation resource will only be removed once orphan deletion has been
	// completed or deemed unnecessary.
	err = wait.PollImmediate(c.waitInterval, c.clusterWaitTimeout, func() (bool, error) {
		_, err := c.adapter.FedGet(nsName)
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		c.t.Fatalf("Error deleting federated %s %q: %v", c.kind, nsName, err)
	}

	// Resource should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	var stateMsg string = "present"
	if !shouldExist {
		stateMsg = "not present"
	}
	for _, client := range c.clusterClients {
		_, err := c.adapter.Get(client, nsName)
		switch {
		case shouldExist && errors.IsNotFound(err):
			c.t.Fatalf("Federated %s %q was unexpectedly deleted from a member cluster", c.kind, nsName)
		case !shouldExist && err == nil:
			c.t.Fatalf("Federated %s %q was unexpectedly orphaned in a member cluster", c.kind, nsName)
		case err != nil && !errors.IsNotFound(err):
			c.t.Fatalf("Error while checking whether %s %q is %s in member clusters: %v", c.kind, nsName, stateMsg, err)
		}
	}
}

func (c *CRUDHelper) checkPropagation(obj pkgruntime.Object) {
	nsName := c.adapter.NamespacedName(obj)

	c.t.Logf("Waiting for %s %q in %d clusters", c.kind, nsName, len(c.clusterClients))
	for _, client := range c.clusterClients {
		err := c.waitForResource(client, obj)
		if err != nil {
			c.t.Fatalf("Failed to verify %s %q in a member cluster: %v", c.kind, nsName, err)
		}
	}
}

func (c *CRUDHelper) waitForResource(client clientset.Interface, obj pkgruntime.Object) error {
	nsName := c.adapter.NamespacedName(obj)
	err := wait.PollImmediate(c.waitInterval, c.clusterWaitTimeout, func() (bool, error) {
		clusterObj, err := c.adapter.Get(client, nsName)
		if err == nil && c.adapter.Equivalent(clusterObj, obj) {
			return true, nil
		}
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	})
	return err
}

func (c *CRUDHelper) updateFedObject(obj pkgruntime.Object) (pkgruntime.Object, error) {
	err := wait.PollImmediate(c.waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		// Target the metadata for simplicity (it's type-agnostic)
		meta := c.adapter.ObjectMeta(obj)
		if meta.Labels == nil {
			meta.Labels = make(map[string]string)
		}
		meta.Labels["foo"] = "bar"

		_, err := c.adapter.FedUpdate(obj)
		if errors.IsConflict(err) {
			// The resource was updated by the federation controller.
			// Get the latest version and retry.
			nsName := c.adapter.NamespacedName(obj)
			obj, err = c.adapter.FedGet(nsName)
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
