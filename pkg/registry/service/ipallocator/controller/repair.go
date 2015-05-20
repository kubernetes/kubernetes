/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/ipallocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Repair is a controller loop that periodically examines all service PortalIP allocations
// and logs any errors, and then sets the compacted and accurate list of all allocated IPs.
//
// Handles:
// * Duplicate PortalIP assignments caused by operator action or undetected race conditions
// * PortalIPs that do not match the current portal network
// * Allocations to services that were not actually created due to a crash or powerloss
// * Migrates old versions of Kubernetes services into the atomic ipallocator model automatically
//
// Can be run at infrequent intervals, and is best performed on startup of the master.
// Is level driven and idempotent - all valid PortalIPs will be updated into the ipallocator
// map at the end of a single execution loop if no race is encountered.
//
// TODO: allocate new IPs if necessary
// TODO: perform repair?
type Repair struct {
	interval time.Duration
	registry service.Registry
	network  *net.IPNet
	alloc    service.RangeRegistry
}

// NewRepair creates a controller that periodically ensures that all portalIPs are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, registry service.Registry, network *net.IPNet, alloc service.RangeRegistry) *Repair {
	return &Repair{
		interval: interval,
		registry: registry,
		network:  network,
		alloc:    alloc,
	}
}

// RunUntil starts the controller until the provided ch is closed.
func (c *Repair) RunUntil(ch chan struct{}) {
	util.Until(func() {
		if err := c.RunOnce(); err != nil {
			util.HandleError(err)
		}
	}, c.interval, ch)
}

// RunOnce verifies the state of the portal IP allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) RunOnce() error {
	// TODO: (per smarterclayton) if Get() or ListServices() is a weak consistency read,
	// or if they are executed against different leaders,
	// the ordering guarantee required to ensure no IP is allocated twice is violated.
	// ListServices must return a ResourceVersion higher than the etcd index Get triggers,
	// and the release code must not release services that have had IPs allocated but not yet been created
	// See #8295

	latest, err := c.alloc.Get()
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	ctx := api.WithNamespace(api.NewDefaultContext(), api.NamespaceAll)
	list, err := c.registry.ListServices(ctx)
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	r := ipallocator.NewCIDRRange(c.network)
	for _, svc := range list.Items {
		if !api.IsServiceIPSet(&svc) {
			continue
		}
		ip := net.ParseIP(svc.Spec.PortalIP)
		if ip == nil {
			// portal IP is broken, reallocate
			util.HandleError(fmt.Errorf("the portal IP %s for service %s/%s is not a valid IP; please recreate", svc.Spec.PortalIP, svc.Name, svc.Namespace))
			continue
		}
		switch err := r.Allocate(ip); err {
		case nil:
		case ipallocator.ErrAllocated:
			// TODO: send event
			// portal IP is broken, reallocate
			util.HandleError(fmt.Errorf("the portal IP %s for service %s/%s was assigned to multiple services; please recreate", ip, svc.Name, svc.Namespace))
		case ipallocator.ErrNotInRange:
			// TODO: send event
			// portal IP is broken, reallocate
			util.HandleError(fmt.Errorf("the portal IP %s for service %s/%s is not within the service CIDR %s; please recreate", ip, svc.Name, svc.Namespace, c.network))
		case ipallocator.ErrFull:
			// TODO: send event
			return fmt.Errorf("the service CIDR %s is full; you must widen the CIDR in order to create new services")
		default:
			return fmt.Errorf("unable to allocate portal IP %s for service %s/%s due to an unknown error, exiting: %v", ip, svc.Name, svc.Namespace, err)
		}
	}

	err = r.Snapshot(latest)
	if err != nil {
		return fmt.Errorf("unable to persist the updated service IP allocations: %v", err)
	}

	if err := c.alloc.CreateOrUpdate(latest); err != nil {
		return fmt.Errorf("unable to persist the updated service IP allocations: %v", err)
	}
	return nil
}
