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
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
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
	interval      time.Duration
	serviceClient coreclient.ServicesGetter
	network       *net.IPNet
	alloc         rangeallocation.RangeRegistry
}

// NewRepair creates a controller that periodically ensures that all clusterIPs are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, serviceClient coreclient.ServicesGetter, network *net.IPNet, alloc rangeallocation.RangeRegistry) *Repair {
	return &Repair{
		interval:      interval,
		serviceClient: serviceClient,
		network:       network,
		alloc:         alloc,
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
	var latest *api.RangeAllocation
	var err error
	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		latest, err = c.alloc.Get()
		return err == nil, err
	})
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	// We explicitly send no resource version, since the resource version
	// of 'latest' is from a different collection, it's not comparable to
	// the service collection. The caching layer keeps per-collection RVs,
	// and this is proper, since in theory the collections could be hosted
	// in separate etcd (or even non-etcd) instances.
	list, err := c.serviceClient.Services(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	r := ipallocator.NewCIDRRange(c.network)
	for _, svc := range list.Items {
		if !api.IsServiceIPSet(&svc) {
			continue
		}
		ip := net.ParseIP(svc.Spec.ClusterIP)
		if ip == nil {
			// cluster IP is broken, reallocate
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not a valid IP; please recreate", svc.Spec.ClusterIP, svc.Name, svc.Namespace))
			continue
		}
		switch err := r.Allocate(ip); err {
		case nil:
		case ipallocator.ErrAllocated:
			// TODO: send event
			// cluster IP is broken, reallocate
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s was assigned to multiple services; please recreate", ip, svc.Name, svc.Namespace))
		case ipallocator.ErrNotInRange:
			// TODO: send event
			// cluster IP is broken, reallocate
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not within the service CIDR %s; please recreate", ip, svc.Name, svc.Namespace, c.network))
		case ipallocator.ErrFull:
			// TODO: send event
			return fmt.Errorf("the service CIDR %v is full; you must widen the CIDR in order to create new services", r)
		default:
			return fmt.Errorf("unable to allocate cluster IP %s for service %s/%s due to an unknown error, exiting: %v", ip, svc.Name, svc.Namespace, err)
		}
	}

	if err := r.Snapshot(latest); err != nil {
		return fmt.Errorf("unable to snapshot the updated service IP allocations: %v", err)
	}

	if err := c.alloc.CreateOrUpdate(latest); err != nil {
		if errors.IsConflict(err) {
			return err
		}
		return fmt.Errorf("unable to persist the updated service IP allocations: %v", err)
	}
	return nil
}
