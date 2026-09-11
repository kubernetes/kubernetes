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
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	eventsv1client "k8s.io/client-go/kubernetes/typed/events/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
)

// See ipallocator/controller/repair.go; this is a copy for ports.
type Repair struct {
	interval       time.Duration
	serviceClient  corev1client.ServicesGetter
	serviceLister  corelisters.ServiceLister
	serviceStore   cache.Store
	servicesSynced cache.InformerSynced
	portRange      net.PortRange
	alloc          rangeallocation.RangeRegistry
	leaks          map[int]int // counter per leaked port

	serviceCacheDeadline time.Time

	broadcaster events.EventBroadcaster
	recorder    events.EventRecorder
}

// How many times we need to detect a leak before we clean up.  This is to
// avoid races between allocating a ports and using it.
const numRepairsBeforeLeakCleanup = 3

// NewRepair creates a controller that periodically ensures that all ports are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
// Services are read from the informer; serviceClient is used to check the
// collection's resource version, or to list Services when the informer store
// does not track its resource version.
func NewRepair(interval time.Duration, serviceClient corev1client.ServicesGetter, serviceInformer coreinformers.ServiceInformer, eventClient eventsv1client.EventsV1Interface, portRange net.PortRange, alloc rangeallocation.RangeRegistry) *Repair {
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: eventClient})
	recorder := eventBroadcaster.NewRecorder(legacyscheme.Scheme, "portallocator-repair-controller")

	registerMetrics()

	return &Repair{
		interval:       interval,
		serviceClient:  serviceClient,
		serviceLister:  serviceInformer.Lister(),
		serviceStore:   serviceInformer.Informer().GetStore(),
		servicesSynced: serviceInformer.Informer().HasSynced,
		portRange:      portRange,
		alloc:          alloc,
		leaks:          map[int]int{},
		broadcaster:    eventBroadcaster,
		recorder:       recorder,
	}
}

// RunUntil starts the controller until stopCh is closed. initialServiceCacheDeadline
// limits freshness checks until the first successful repair, so waiting for the
// informer does not consume the caller's entire startup budget.
func (c *Repair) RunUntil(onFirstSuccess func(), stopCh chan struct{}, initialServiceCacheDeadline time.Time) {
	c.serviceCacheDeadline = initialServiceCacheDeadline
	c.broadcaster.StartRecordingToSink(stopCh)
	defer c.broadcaster.Shutdown()

	if !cache.WaitForNamedCacheSync("portallocator-repair-controller", stopCh, c.servicesSynced) {
		return
	}

	ctx := wait.ContextForChannel(stopCh)
	var once sync.Once
	wait.Until(func() {
		if err := c.runOnce(ctx); err != nil {
			runtime.HandleError(err)
			return
		}
		once.Do(func() {
			c.serviceCacheDeadline = time.Time{}
			onFirstSuccess()
		})
	}, c.interval, stopCh)
}

// runOnce verifies the state of the port allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) runOnce(ctx context.Context) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		err := c.doRunOnce(ctx)
		if err != nil {
			nodePortRepairReconcileErrors.Inc()
		}
		return err
	})
}

// doRunOnce verifies the state of the port allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) doRunOnce(ctx context.Context) error {
	// TODO: (per smarterclayton) if Get() or ListServices() is a weak consistency read,
	// or if they are executed against different leaders,
	// the ordering guarantee required to ensure no port is allocated twice is violated.
	// ListServices must return a ResourceVersion higher than the etcd index Get triggers,
	// and the release code must not release services that have had ports allocated but not yet been created
	// See #8295

	// If etcd server is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and etcd at the same time.
	var snapshot *api.RangeAllocation
	var err error
	err = wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		snapshot, err = c.alloc.Get()
		if err != nil {
			runtime.HandleError(fmt.Errorf("unable to refresh the port allocations: %w", err))
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("unable to refresh the port allocations: %w", err)
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

	// Before releasing unused ports, check that the informer has observed the
	// Services in storage. The allocation snapshot's resource version cannot
	// be used for this check because it belongs to a different collection.
	var list []*corev1.Service
	complete := true
	cacheRV := c.serviceStore.LastStoreSyncResourceVersion()
	if cacheRV == "" {
		// Without AtomicFIFO, the store does not track its resource version,
		// so read Services directly to safely detect unused ports.
		svcList, err := c.serviceClient.Services(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("unable to refresh the port block: %w", err)
		}
		for i := range svcList.Items {
			list = append(list, &svcList.Items[i])
		}
	} else {
		// Bound periodic freshness checks by the repair interval. During startup,
		// also honor the caller's deadline, shared across cache sync and retries,
		// to leave time for persisting allocations and reporting first success.
		deadline := time.Now().Add(c.interval)
		if !c.serviceCacheDeadline.IsZero() && c.serviceCacheDeadline.Before(deadline) {
			deadline = c.serviceCacheDeadline
		}
		readCtx, cancel := context.WithDeadline(ctx, deadline)
		complete = c.serviceCacheIsFresh(readCtx)
		cancel()
		list, err = c.serviceLister.List(labels.Everything())
		if err != nil {
			return fmt.Errorf("unable to refresh the port block: %w", err)
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	rebuilt, err := portallocator.NewInMemory(c.portRange)
	if err != nil {
		return fmt.Errorf("unable to create port allocator: %v", err)
	}
	// Check every Service's ports, and rebuild the state as we think it should be.
	for _, svc := range list {
		ports := collectServiceNodePorts(svc)
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
					nodePortRepairPortErrors.WithLabelValues("repair").Inc()
					c.recorder.Eventf(svc, nil, corev1.EventTypeWarning, "PortNotAllocated", "PortAllocation", "Port %d is not allocated; repairing", port)
					runtime.HandleError(fmt.Errorf("the node port %d for service %s/%s is not allocated; repairing", port, svc.Name, svc.Namespace))
				}
				delete(c.leaks, port) // it is used, so it can't be leaked
			case portallocator.ErrAllocated:
				// port is duplicate, reallocate
				nodePortRepairPortErrors.WithLabelValues("duplicate").Inc()
				c.recorder.Eventf(svc, nil, corev1.EventTypeWarning, "PortAlreadyAllocated", "PortAllocation", "Port %d was assigned to multiple services; please recreate service", port)
				runtime.HandleError(fmt.Errorf("the node port %d for service %s/%s was assigned to multiple services; please recreate", port, svc.Name, svc.Namespace))
			case err.(*portallocator.ErrNotInRange):
				// port is out of range, reallocate
				nodePortRepairPortErrors.WithLabelValues("outOfRange").Inc()
				c.recorder.Eventf(svc, nil, corev1.EventTypeWarning, "PortOutOfRange", "PortAllocation", "Port %d is not within the port range %s; please recreate service", port, c.portRange)
				runtime.HandleError(fmt.Errorf("the port %d for service %s/%s is not within the port range %s; please recreate", port, svc.Name, svc.Namespace, c.portRange))
			case portallocator.ErrFull:
				// somehow we are out of ports
				nodePortRepairPortErrors.WithLabelValues("full").Inc()
				c.recorder.Eventf(svc, nil, corev1.EventTypeWarning, "PortRangeFull", "PortAllocation", "Port range %s is full; you must widen the port range in order to create new services", c.portRange)
				return fmt.Errorf("the port range %s is full; you must widen the port range in order to create new services", c.portRange)
			default:
				nodePortRepairPortErrors.WithLabelValues("unknown").Inc()
				c.recorder.Eventf(svc, nil, corev1.EventTypeWarning, "UnknownError", "PortAllocation", "Unable to allocate port %d due to an unknown error", port)
				return fmt.Errorf("unable to allocate port %d for service %s/%s due to an unknown error, exiting: %v", port, svc.Name, svc.Namespace, err)
			}
		}
	}

	// Check for ports that are left in the old set.  They appear to have been leaked.
	stored.ForEach(func(port int) {
		if !complete {
			// If the informer is stale, these ports may still be in use. Keep
			// them allocated and do not advance their leak cleanup counters.
			if err := rebuilt.Allocate(port); err != nil {
				runtime.HandleError(fmt.Errorf("the node port %d may have leaked, but can not be allocated: %w", port, err))
			}
			return
		}
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
				// do not increment the metric here, if it is a leak it will be detected once the counter gets to 0
				runtime.HandleError(fmt.Errorf("the node port %d may have leaked, but can not be allocated: %v", port, err))
			}
		default:
			nodePortRepairPortErrors.WithLabelValues("leak").Inc()
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

// serviceCacheIsFresh gets the current Service collection resource version from
// a live LIST and waits for the informer store to reach it. It returns false if
// the request or comparison fails, or the context is canceled before the store
// catches up. The caller must read the allocation snapshot before this check
// and list Services from the informer afterwards to avoid releasing ports whose
// Services have not reached the cache yet.
func (c *Repair) serviceCacheIsFresh(ctx context.Context) bool {
	live, err := c.serviceClient.Services(metav1.NamespaceAll).List(ctx, metav1.ListOptions{Limit: 1})
	if err != nil {
		nodePortRepairLeakCleanupDeferred.WithLabelValues("error").Inc()
		runtime.HandleError(fmt.Errorf("unable to read the current services resource version, deferring leak cleanup: %w", err))
		return false
	}
	err = wait.PollUntilContextCancel(ctx, 10*time.Millisecond, true, func(context.Context) (bool, error) {
		cmp, err := resourceversion.CompareResourceVersion(c.serviceStore.LastStoreSyncResourceVersion(), live.ResourceVersion)
		return cmp >= 0, err
	})
	if err != nil {
		reason := "error"
		if ctx.Err() != nil {
			reason = "stale"
		}
		nodePortRepairLeakCleanupDeferred.WithLabelValues(reason).Inc()
		runtime.HandleError(fmt.Errorf("service informer did not reach resource version %s (cache at %s), deferring leak cleanup: %w", live.ResourceVersion, c.serviceStore.LastStoreSyncResourceVersion(), err))
		return false
	}
	return true
}

// collectServiceNodePorts returns nodePorts specified in the Service.
// Please note that:
//  1. same nodePort with *same* protocol will be duplicated as it is
//  2. same nodePort with *different* protocol will be deduplicated
func collectServiceNodePorts(service *corev1.Service) []int {
	var servicePorts []int
	// map from nodePort to set of protocols
	seen := make(map[int]sets.Set[string])
	for _, port := range service.Spec.Ports {
		nodePort := int(port.NodePort)
		if nodePort == 0 {
			continue
		}
		proto := string(port.Protocol)
		s := seen[nodePort]
		if s == nil { // have not seen this nodePort before
			s = sets.New[string](proto)
			servicePorts = append(servicePorts, nodePort)
		} else if s.Has(proto) { // same nodePort with same protocol
			servicePorts = append(servicePorts, nodePort)
		} else { // same nodePort with different protocol
			s.Insert(proto)
		}
		seen[nodePort] = s
	}

	healthPort := int(service.Spec.HealthCheckNodePort)
	if healthPort != 0 {
		s := seen[healthPort]
		// TODO: is it safe to assume the protocol is always TCP?
		if s == nil || s.Has(string(corev1.ProtocolTCP)) {
			servicePorts = append(servicePorts, healthPort)
		}
	}

	return servicePorts
}
