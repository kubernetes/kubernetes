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

package controller

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/api"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
)

// See ipallocator/controller/repair.go; this is a copy for ports.
type Repair struct {
	interval      time.Duration
	serviceClient coreclient.ServicesGetter
	portRange     net.PortRange
	alloc         rangeallocation.RangeRegistry
	leaks         map[int]int // counter per leaked port
}

// How many times we need to detect a leak before we clean up.  This is to
// avoid races between allocating a ports and using it.
const numRepairsBeforeLeakCleanup = 3

// NewRepair creates a controller that periodically ensures that all ports are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, serviceClient coreclient.ServicesGetter, portRange net.PortRange, alloc rangeallocation.RangeRegistry) *Repair {
	return &Repair{
		interval:      interval,
		serviceClient: serviceClient,
		portRange:     portRange,
		alloc:         alloc,
		leaks:         map[int]int{},
	}
}

// RunUntil starts the controller until the provided ch is closed.
func (c *Repair) RunUntil(ch chan struct{}) {
	wait.Until(func() {
		if err := c.RunOnce(); err != nil {
			runtime.HandleError(err)
		}
	}, c.interval, ch)
}

// RunOnce verifies the state of the port allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) RunOnce() error {
	return retry.RetryOnConflict(retry.DefaultBackoff, c.runOnce)
}

// runOnce verifies the state of the port allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) runOnce() error {
	// TODO: (per smarterclayton) if Get() or ListServices() is a weak consistency read,
	// or if they are executed against different leaders,
	// the ordering guarantee required to ensure no port is allocated twice is violated.
	// ListServices must return a ResourceVersion higher than the etcd index Get triggers,
	// and the release code must not release services that have had ports allocated but not yet been created
	// See #8295

	// If etcd server is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and etcd at the same time.
	var snapshot *api.RangeAllocation

	err := wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		var err error
		snapshot, err = c.alloc.Get()
		return err == nil, err
	})
	if err != nil {
		return fmt.Errorf("unable to refresh the port allocations: %v", err)
	}
	// If not yet initialized.
	if snapshot.Range == "" {
		snapshot.Range = c.portRange.String()
	}
	// Create an allocator because it is easy to use.
	stored, err := portallocator.NewFromSnapshot(snapshot)
	if err != nil {
		return fmt.Errorf("unable to rebuild allocator from snapshot: %v", err)
	}

	// We explicitly send no resource version, since the resource version
	// of 'snapshot' is from a different collection, it's not comparable to
	// the service collection. The caching layer keeps per-collection RVs,
	// and this is proper, since in theory the collections could be hosted
	// in separate etcd (or even non-etcd) instances.
	list, err := c.serviceClient.Services(metav1.NamespaceAll).List(metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("unable to refresh the port block: %v", err)
	}

	rebuilt := portallocator.NewPortAllocator(c.portRange)
	// Check every Service's ports, and rebuild the state as we think it should be.
	for i := range list.Items {
		svc := &list.Items[i]
		ports := service.CollectServiceNodePorts(svc)
		if len(ports) == 0 {
			continue
		}

		for _, port := range ports {
			switch err := rebuilt.Allocate(port); err {
			case nil:
				if stored.Has(port) {
					// remove it from the old set, so we can find leaks
					stored.Release(port)
				} else {
					// doesn't seem to be allocated
					runtime.HandleError(fmt.Errorf("the node port %d for service %s/%s is not allocated; repairing", port, svc.Name, svc.Namespace))
				}
				delete(c.leaks, port) // it is used, so it can't be leaked
			case portallocator.ErrAllocated:
				// TODO: send event
				// port is duplicate, reallocate
				runtime.HandleError(fmt.Errorf("the node port %d for service %s/%s was assigned to multiple services; please recreate", port, svc.Name, svc.Namespace))
			case err.(*portallocator.ErrNotInRange):
				// TODO: send event
				// port is out of range, reallocate
				runtime.HandleError(fmt.Errorf("the port %d for service %s/%s is not within the port range %v; please recreate", port, svc.Name, svc.Namespace, c.portRange))
			case portallocator.ErrFull:
				// TODO: send event
				// somehow we are out of ports
				return fmt.Errorf("the port range %v is full; you must widen the port range in order to create new services", c.portRange)
			default:
				return fmt.Errorf("unable to allocate port %d for service %s/%s due to an unknown error, exiting: %v", port, svc.Name, svc.Namespace, err)
			}
		}
	}

	// Check for ports that are left in the old set.  They appear to have been leaked.
	stored.ForEach(func(port int) {
		count, found := c.leaks[port]
		switch {
		case !found:
			// flag it to be cleaned up after any races (hopefully) are gone
			runtime.HandleError(fmt.Errorf("the node port %d may have leaked: flagging for later clean up", port))
			count = numRepairsBeforeLeakCleanup - 1
			fallthrough
		case count > 0:
			// pretend it is still in use until count expires
			c.leaks[port] = count - 1
			if err := rebuilt.Allocate(port); err != nil {
				runtime.HandleError(fmt.Errorf("the node port %d may have leaked, but can not be allocated: %v", port, err))
			}
		default:
			// do not add it to the rebuilt set, which means it will be available for reuse
			runtime.HandleError(fmt.Errorf("the node port %d appears to have leaked: cleaning up", port))
		}
	})

	// Blast the rebuilt state into storage.
	if err := rebuilt.Snapshot(snapshot); err != nil {
		return fmt.Errorf("unable to snapshot the updated port allocations: %v", err)
	}

	if err := c.alloc.CreateOrUpdate(snapshot); err != nil {
		if errors.IsConflict(err) {
			return err
		}
		return fmt.Errorf("unable to persist the updated port allocations: %v", err)
	}
	return nil
}
