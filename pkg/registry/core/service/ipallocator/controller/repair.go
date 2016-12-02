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
	"net"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

// Repair is a controller loop that periodically examines all service ClusterIP allocations
// and logs any errors, and then sets the compacted and accurate list of all allocated IPs.
//
// Handles:
// * Duplicate ClusterIP assignments caused by operator action or undetected race conditions
// * ClusterIPs that do not match the currently configured range
// * Allocations to services that were not actually created due to a crash or powerloss
// * Migrates old versions of Kubernetes services into the atomic ipallocator model automatically
//
// Can be run at infrequent intervals, and is best performed on startup of the master.
// Is level driven and idempotent - all valid ClusterIPs will be updated into the ipallocator
// map at the end of a single execution loop if no race is encountered.
//
// TODO: allocate new IPs if necessary
// TODO: perform repair?
type Repair struct {
	interval time.Duration
	registry service.Registry
	network  *net.IPNet
	alloc    rangeallocation.RangeRegistry
	leaks    map[string]int // counter per leaked IP
}

// How many times we need to detect a leak before we clean up.  This is to
// avoid races between allocating an IP and using it.
const numRepairsBeforeLeakCleanup = 3

// NewRepair creates a controller that periodically ensures that all clusterIPs are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, registry service.Registry, network *net.IPNet, alloc rangeallocation.RangeRegistry) *Repair {
	return &Repair{
		interval: interval,
		registry: registry,
		network:  network,
		alloc:    alloc,
		leaks:    map[string]int{},
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

// RunOnce verifies the state of the cluster IP allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) RunOnce() error {
	return retry.RetryOnConflict(retry.DefaultBackoff, c.runOnce)
}

// runOnce verifies the state of the cluster IP allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) runOnce() error {
	// TODO: (per smarterclayton) if Get() or ListServices() is a weak consistency read,
	// or if they are executed against different leaders,
	// the ordering guarantee required to ensure no IP is allocated twice is violated.
	// ListServices must return a ResourceVersion higher than the etcd index Get triggers,
	// and the release code must not release services that have had IPs allocated but not yet been created
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
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}
	// If not yet initialized.
	if snapshot.Range == "" {
		snapshot.Range = c.network.String()
	}
	// Create an allocator because it is easy to use.
	stored, err := ipallocator.NewFromSnapshot(snapshot)
	if err != nil {
		return fmt.Errorf("unable to rebuild allocator from snapshot: %v", err)
	}

	ctx := api.WithNamespace(api.NewDefaultContext(), api.NamespaceAll)
	// We explicitly send no resource version, since the resource version
	// of 'snapshot' is from a different collection, it's not comparable to
	// the service collection. The caching layer keeps per-collection RVs,
	// and this is proper, since in theory the collections could be hosted
	// in separate etcd (or even non-etcd) instances.
	list, err := c.registry.ListServices(ctx, &api.ListOptions{})
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	rebuilt := ipallocator.NewCIDRRange(c.network)
	// Check every Service's ClusterIP, and rebuild the state as we think it should be.
	for _, svc := range list.Items {
		if !api.IsServiceIPSet(&svc) {
			// didn't need a cluster IP
			continue
		}
		ip := net.ParseIP(svc.Spec.ClusterIP)
		if ip == nil {
			// cluster IP is corrupt
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not a valid IP; please recreate", svc.Spec.ClusterIP, svc.Name, svc.Namespace))
			continue
		}
		// mark it as in-use
		switch err := rebuilt.Allocate(ip); err {
		case nil:
			if stored.Has(ip) {
				// remove it from the old set, so we can find leaks
				stored.Release(ip)
			} else {
				// cluster IP doesn't seem to be allocated
				runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not allocated; repairing", svc.Spec.ClusterIP, svc.Name, svc.Namespace))
			}
			delete(c.leaks, ip.String()) // it is used, so it can't be leaked
		case ipallocator.ErrAllocated:
			// TODO: send event
			// cluster IP is duplicate
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s was assigned to multiple services; please recreate", ip, svc.Name, svc.Namespace))
		case ipallocator.ErrNotInRange:
			// TODO: send event
			// cluster IP is out of range
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not within the service CIDR %s; please recreate", ip, svc.Name, svc.Namespace, c.network))
		case ipallocator.ErrFull:
			// TODO: send event
			// somehow we are out of IPs
			return fmt.Errorf("the service CIDR %v is full; you must widen the CIDR in order to create new services", rebuilt)
		default:
			return fmt.Errorf("unable to allocate cluster IP %s for service %s/%s due to an unknown error, exiting: %v", ip, svc.Name, svc.Namespace, err)
		}
	}

	// Check for IPs that are left in the old set.  They appear to have been leaked.
	stored.ForEach(func(ip net.IP) {
		count, found := c.leaks[ip.String()]
		switch {
		case !found:
			// flag it to be cleaned up after any races (hopefully) are gone
			runtime.HandleError(fmt.Errorf("the cluster IP %s may have leaked: flagging for later clean up", ip))
			count = numRepairsBeforeLeakCleanup - 1
			fallthrough
		case count > 0:
			// pretend it is still in use until count expires
			c.leaks[ip.String()] = count - 1
			if err := rebuilt.Allocate(ip); err != nil {
				runtime.HandleError(fmt.Errorf("the cluster IP %s may have leaked, but can not be allocated: %v", ip, err))
			}
		default:
			// do not add it to the rebuilt set, which means it will be available for reuse
			runtime.HandleError(fmt.Errorf("the cluster IP %s appears to have leaked: cleaning up", ip))
		}
	})

	// Blast the rebuilt state into storage.
	if err := rebuilt.Snapshot(snapshot); err != nil {
		return fmt.Errorf("unable to snapshot the updated service IP allocations: %v", err)
	}
	if err := c.alloc.CreateOrUpdate(snapshot); err != nil {
		if errors.IsConflict(err) {
			return err
		}
		return fmt.Errorf("unable to persist the updated service IP allocations: %v", err)
	}
	return nil
}
