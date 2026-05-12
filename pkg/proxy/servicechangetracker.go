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

package proxy

import (
	"reflect"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
)

// ServiceChangeTracker carries state about uncommitted changes to an arbitrary number of
// Services, keyed by their namespace and name.
type ServiceChangeTracker struct {
	// lock protects items.
	lock sync.Mutex
	// items maps a service to its serviceChange.
	items map[types.NamespacedName]*serviceChange

	// makeServiceInfo allows the proxier to inject customized information when
	// processing services.
	makeServiceInfo makeServicePortFunc
	// processServiceMapChange is invoked by the apply function on every change. This
	// function should not modify the ServicePortMaps, but just use the changes for
	// any Proxier-specific cleanup.
	processServiceMapChange processServiceMapChangeFunc

	ipFamily v1.IPFamily
}

type makeServicePortFunc func(*v1.ServicePort, *v1.Service, *BaseServicePortInfo) ServicePort
type processServiceMapChangeFunc func(previous, current ServicePortMap)

// serviceChange contains all changes to services that happened since proxy rules were synced.  For a single object,
// changes are accumulated, i.e. previous is state from before applying the changes,
// current is state after applying all of the changes.
type serviceChange struct {
	previous ServicePortMap
	current  ServicePortMap
}

// NewServiceChangeTracker initializes a ServiceChangeTracker
func NewServiceChangeTracker(ipFamily v1.IPFamily, makeServiceInfo makeServicePortFunc, processServiceMapChange processServiceMapChangeFunc) *ServiceChangeTracker {
	return &ServiceChangeTracker{
		items:                   make(map[types.NamespacedName]*serviceChange),
		makeServiceInfo:         makeServiceInfo,
		ipFamily:                ipFamily,
		processServiceMapChange: processServiceMapChange,
	}
}

// Update updates the ServiceChangeTracker based on the <previous, current> service pair
// (where either previous or current, but not both, can be nil). It returns true if sct
// contains changes that need to be synced (whether or not those changes were caused by
// this update); note that this is different from the return value of
// EndpointChangeTracker.EndpointSliceUpdate().
func (sct *ServiceChangeTracker) Update(previous, current *v1.Service) bool {
	// This is unexpected, we should return false directly.
	if previous == nil && current == nil {
		return false
	}

	svc := current
	if svc == nil {
		svc = previous
	}
	metrics.ServiceChangesTotal.Inc()
	namespacedName := types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name}

	sct.lock.Lock()
	defer sct.lock.Unlock()

	change, exists := sct.items[namespacedName]
	if !exists {
		change = &serviceChange{}
		change.previous = sct.serviceToServiceMap(previous)
		sct.items[namespacedName] = change
	}
	change.current = sct.serviceToServiceMap(current)
	// if change.previous equal to change.current, it means no change
	if reflect.DeepEqual(change.previous, change.current) {
		delete(sct.items, namespacedName)
	} else {
		klog.V(4).InfoS("Service updated ports", "service", klog.KObj(svc), "portCount", len(change.current))
	}
	metrics.ServiceChangesPending.Set(float64(len(sct.items)))
	return len(sct.items) > 0
}

// ServicePortMap maps a service to its ServicePort.
type ServicePortMap map[ServicePortName]ServicePort

// UpdateServiceMapResult is the updated results after applying service changes.
type UpdateServiceMapResult struct {
	// UpdatedServices lists the names of all services added/updated/deleted since the
	// last Update.
	UpdatedServices sets.Set[types.NamespacedName]
}

// HealthCheckNodePorts returns a map of Service names to HealthCheckNodePort values
// for all Services in sm with non-zero HealthCheckNodePort.
func (sm ServicePortMap) HealthCheckNodePorts() map[types.NamespacedName]uint16 {
	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to svcPortMap.
	ports := make(map[types.NamespacedName]uint16)
	for svcPortName, info := range sm {
		if info.HealthCheckNodePort() != 0 {
			ports[svcPortName.NamespacedName] = uint16(info.HealthCheckNodePort())
		}
	}
	return ports
}

// serviceToServiceMap translates a single Service object to a ServicePortMap.
//
// NOTE: service object should NOT be modified.
func (sct *ServiceChangeTracker) serviceToServiceMap(service *v1.Service) ServicePortMap {
	if service == nil {
		return nil
	}

	if proxyutil.ShouldSkipService(service) {
		return nil
	}

	clusterIP := proxyutil.GetClusterIPByFamily(sct.ipFamily, service)
	if clusterIP == "" {
		return nil
	}

	svcPortMap := make(ServicePortMap)
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := ServicePortName{NamespacedName: svcName, Port: servicePort.Name, Protocol: servicePort.Protocol}
		baseSvcInfo := newBaseServiceInfo(service, sct.ipFamily, servicePort)
		if sct.makeServiceInfo != nil {
			svcPortMap[svcPortName] = sct.makeServiceInfo(servicePort, service, baseSvcInfo)
		} else {
			svcPortMap[svcPortName] = baseSvcInfo
		}
	}
	return svcPortMap
}

// Update updates ServicePortMap base on the given changes, returns information about the
// diff since the last Update, triggers processServiceMapChange on every change, and
// clears the changes map.
func (sm ServicePortMap) Update(sct *ServiceChangeTracker) UpdateServiceMapResult {
	sct.lock.Lock()
	defer sct.lock.Unlock()

	result := UpdateServiceMapResult{
		UpdatedServices: sets.New[types.NamespacedName](),
	}

	for nn, change := range sct.items {
		if sct.processServiceMapChange != nil {
			sct.processServiceMapChange(change.previous, change.current)
		}
		result.UpdatedServices.Insert(nn)

		sm.merge(change.current)
		// filter out the Update event of current changes from previous changes
		// before calling unmerge() so that can skip deleting the Update events.
		change.previous.filter(change.current)
		sm.unmerge(change.previous)
	}
	// clear changes after applying them to ServicePortMap.
	sct.items = make(map[types.NamespacedName]*serviceChange)
	metrics.ServiceChangesPending.Set(0)

	return result
}

// merge adds other ServicePortMap's elements to current ServicePortMap.
// If collision, other ALWAYS win. Otherwise add the other to current.
// In other words, if some elements in current collisions with other, update the current by other.
func (sm *ServicePortMap) merge(other ServicePortMap) {
	for svcPortName, info := range other {
		_, exists := (*sm)[svcPortName]
		if !exists {
			klog.V(4).InfoS("Adding new service port", "portName", svcPortName, "servicePort", info)
		} else {
			klog.V(4).InfoS("Updating existing service port", "portName", svcPortName, "servicePort", info)
		}
		(*sm)[svcPortName] = info
	}
}

// filter filters out elements from ServicePortMap base on given ports string sets.
func (sm *ServicePortMap) filter(other ServicePortMap) {
	for svcPortName := range *sm {
		// skip the delete for Update event.
		if _, ok := other[svcPortName]; ok {
			delete(*sm, svcPortName)
		}
	}
}

// unmerge deletes all other ServicePortMap's elements from current ServicePortMap.
func (sm *ServicePortMap) unmerge(other ServicePortMap) {
	for svcPortName := range other {
		_, exists := (*sm)[svcPortName]
		if exists {
			klog.V(4).InfoS("Removing service port", "portName", svcPortName)
			delete(*sm, svcPortName)
		} else {
			klog.ErrorS(nil, "Service port does not exists", "portName", svcPortName)
		}
	}
}
