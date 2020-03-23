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
	"context"
	"fmt"
	"net"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	netutil "k8s.io/utils/net"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
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
	serviceClient corev1client.ServicesGetter

	network          *net.IPNet
	alloc            rangeallocation.RangeRegistry
	secondaryNetwork *net.IPNet
	secondaryAlloc   rangeallocation.RangeRegistry

	leaks    map[string]int // counter per leaked IP
	recorder record.EventRecorder
}

// How many times we need to detect a leak before we clean up.  This is to
// avoid races between allocating an IP and using it.
const numRepairsBeforeLeakCleanup = 3

// NewRepair creates a controller that periodically ensures that all clusterIPs are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, serviceClient corev1client.ServicesGetter, eventClient corev1client.EventsGetter, network *net.IPNet, alloc rangeallocation.RangeRegistry, secondaryNetwork *net.IPNet, secondaryAlloc rangeallocation.RangeRegistry) *Repair {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&corev1client.EventSinkImpl{Interface: eventClient.Events("")})
	recorder := eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: "ipallocator-repair-controller"})

	return &Repair{
		interval:      interval,
		serviceClient: serviceClient,

		network:          network,
		alloc:            alloc,
		secondaryNetwork: secondaryNetwork,
		secondaryAlloc:   secondaryAlloc,

		leaks:    map[string]int{},
		recorder: recorder,
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

// selectAllocForIP returns an allocator for an IP based weather it belongs to the primary or the secondary allocator
func (c *Repair) selectAllocForIP(ip net.IP, primary ipallocator.Interface, secondary ipallocator.Interface) ipallocator.Interface {
	if !c.shouldWorkOnSecondary() {
		return primary
	}

	cidr := secondary.CIDR()
	if netutil.IsIPv6CIDR(&cidr) && netutil.IsIPv6(ip) {
		return secondary
	}

	return primary
}

// shouldWorkOnSecondary returns true if the repairer should perform work for secondary network (dual stack)
func (c *Repair) shouldWorkOnSecondary() bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return false
	}

	return c.secondaryNetwork != nil && c.secondaryNetwork.IP != nil
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
	var secondarySnapshot *api.RangeAllocation

	var stored, secondaryStored ipallocator.Interface
	var err, secondaryErr error

	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		var err error
		snapshot, err = c.alloc.Get()
		if err != nil {
			return false, err
		}

		if c.shouldWorkOnSecondary() {
			secondarySnapshot, err = c.secondaryAlloc.Get()
			if err != nil {
				return false, err
			}
		}

		return true, nil
	})
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}
	// If not yet initialized.
	if snapshot.Range == "" {
		snapshot.Range = c.network.String()
	}

	if c.shouldWorkOnSecondary() && secondarySnapshot.Range == "" {
		secondarySnapshot.Range = c.secondaryNetwork.String()
	}
	// Create an allocator because it is easy to use.

	stored, err = ipallocator.NewFromSnapshot(snapshot)
	if c.shouldWorkOnSecondary() {
		secondaryStored, secondaryErr = ipallocator.NewFromSnapshot(secondarySnapshot)
	}

	if err != nil || secondaryErr != nil {
		return fmt.Errorf("unable to rebuild allocator from snapshots: %v", err)
	}

	// We explicitly send no resource version, since the resource version
	// of 'snapshot' is from a different collection, it's not comparable to
	// the service collection. The caching layer keeps per-collection RVs,
	// and this is proper, since in theory the collections could be hosted
	// in separate etcd (or even non-etcd) instances.
	list, err := c.serviceClient.Services(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block: %v", err)
	}

	var rebuilt, secondaryRebuilt *ipallocator.Range
	rebuilt, err = ipallocator.NewCIDRRange(c.network)
	if err != nil {
		return fmt.Errorf("unable to create CIDR range: %v", err)
	}

	if c.shouldWorkOnSecondary() {
		secondaryRebuilt, err = ipallocator.NewCIDRRange(c.secondaryNetwork)
	}

	if err != nil {
		return fmt.Errorf("unable to create CIDR range: %v", err)
	}

	// Check every Service's ClusterIP, and rebuild the state as we think it should be.
	for _, svc := range list.Items {
		if !helper.IsServiceIPSet(&svc) {
			// didn't need a cluster IP
			continue
		}
		ip := net.ParseIP(svc.Spec.ClusterIP)
		if ip == nil {
			// cluster IP is corrupt
			c.recorder.Eventf(&svc, v1.EventTypeWarning, "ClusterIPNotValid", "Cluster IP %s is not a valid IP; please recreate service", svc.Spec.ClusterIP)
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not a valid IP; please recreate", svc.Spec.ClusterIP, svc.Name, svc.Namespace))
			continue
		}

		// mark it as in-use
		actualAlloc := c.selectAllocForIP(ip, rebuilt, secondaryRebuilt)
		switch err := actualAlloc.Allocate(ip); err {
		case nil:
			actualStored := c.selectAllocForIP(ip, stored, secondaryStored)
			if actualStored.Has(ip) {
				// remove it from the old set, so we can find leaks
				actualStored.Release(ip)
			} else {
				// cluster IP doesn't seem to be allocated
				c.recorder.Eventf(&svc, v1.EventTypeWarning, "ClusterIPNotAllocated", "Cluster IP %s is not allocated; repairing", ip)
				runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not allocated; repairing", ip, svc.Name, svc.Namespace))
			}
			delete(c.leaks, ip.String()) // it is used, so it can't be leaked
		case ipallocator.ErrAllocated:
			// cluster IP is duplicate
			c.recorder.Eventf(&svc, v1.EventTypeWarning, "ClusterIPAlreadyAllocated", "Cluster IP %s was assigned to multiple services; please recreate service", ip)
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s was assigned to multiple services; please recreate", ip, svc.Name, svc.Namespace))
		case err.(*ipallocator.ErrNotInRange):
			// cluster IP is out of range
			c.recorder.Eventf(&svc, v1.EventTypeWarning, "ClusterIPOutOfRange", "Cluster IP %s is not within the service CIDR %s; please recreate service", ip, c.network)
			runtime.HandleError(fmt.Errorf("the cluster IP %s for service %s/%s is not within the service CIDR %s; please recreate", ip, svc.Name, svc.Namespace, c.network))
		case ipallocator.ErrFull:
			// somehow we are out of IPs
			cidr := actualAlloc.CIDR()
			c.recorder.Eventf(&svc, v1.EventTypeWarning, "ServiceCIDRFull", "Service CIDR %v is full; you must widen the CIDR in order to create new services", cidr)
			return fmt.Errorf("the service CIDR %v is full; you must widen the CIDR in order to create new services", cidr)
		default:
			c.recorder.Eventf(&svc, v1.EventTypeWarning, "UnknownError", "Unable to allocate cluster IP %s due to an unknown error", ip)
			return fmt.Errorf("unable to allocate cluster IP %s for service %s/%s due to an unknown error, exiting: %v", ip, svc.Name, svc.Namespace, err)
		}
	}

	c.checkLeaked(stored, rebuilt)
	if c.shouldWorkOnSecondary() {
		c.checkLeaked(secondaryStored, secondaryRebuilt)
	}

	// Blast the rebuilt state into storage.
	err = c.saveSnapShot(rebuilt, c.alloc, snapshot)
	if err != nil {
		return err
	}

	if c.shouldWorkOnSecondary() {
		err := c.saveSnapShot(secondaryRebuilt, c.secondaryAlloc, secondarySnapshot)
		if err != nil {
			return nil
		}
	}
	return nil
}

func (c *Repair) saveSnapShot(rebuilt *ipallocator.Range, alloc rangeallocation.RangeRegistry, snapshot *api.RangeAllocation) error {
	if err := rebuilt.Snapshot(snapshot); err != nil {
		return fmt.Errorf("unable to snapshot the updated service IP allocations: %v", err)
	}
	if err := alloc.CreateOrUpdate(snapshot); err != nil {
		if errors.IsConflict(err) {
			return err
		}
		return fmt.Errorf("unable to persist the updated service IP allocations: %v", err)
	}

	return nil
}

func (c *Repair) checkLeaked(stored ipallocator.Interface, rebuilt *ipallocator.Range) {
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

}
