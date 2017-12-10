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

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
)

// serviceChange contains all changes to services that happened since proxy rules were synced.  For a single object,
// changes are accumulated, i.e. previous is state from before applying the changes,
// current is state after applying all of the changes.
type serviceChange struct {
	previous ServiceMap
	current  ServiceMap
}

// ServiceChangeTracker carries state about uncommitted changes to an arbitrary number of
// Services, keyed by their namespace and name.
type ServiceChangeTracker struct {
	// lock protects items.
	lock sync.Mutex
	// items maps a service to its serviceChange.
	items map[types.NamespacedName]*serviceChange
}

// NewServiceChangeTracker initializes a ServiceChangeTracker
func NewServiceChangeTracker() *ServiceChangeTracker {
	return &ServiceChangeTracker{
		items: make(map[types.NamespacedName]*serviceChange),
	}
}

// Update updates given service's change map based on the <previous, current> service pair.  It returns true if items changed,
// otherwise return false.  Update can be used to add/update/delete items of ServiceChangeMap.  For example,
// Add item
//   - pass <nil, service> as the <previous, current> pair.
// Update item
//   - pass <oldService, service> as the <previous, current> pair.
// Delete item
//   - pass <service, nil> as the <previous, current> pair.
//
// makeServicePort() return a proxy.ServicePort based on the given Service and its ServicePort.  We inject makeServicePort()
// so that giving caller side a chance to initialize proxy.ServicePort interface.
func (sct *ServiceChangeTracker) Update(previous, current *api.Service, makeServicePort func(servicePort *api.ServicePort, service *api.Service) ServicePort) bool {
	svc := current
	if svc == nil {
		svc = previous
	}
	// previous == nil && current == nil is unexpected, we should return false directly.
	if svc == nil {
		return false
	}
	namespacedName := types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name}

	sct.lock.Lock()
	defer sct.lock.Unlock()

	change, exists := sct.items[namespacedName]
	if !exists {
		change = &serviceChange{}
		change.previous = serviceToServiceMap(previous, makeServicePort)
		sct.items[namespacedName] = change
	}
	change.current = serviceToServiceMap(current, makeServicePort)
	// if change.previous equal to change.current, it means no change
	if reflect.DeepEqual(change.previous, change.current) {
		delete(sct.items, namespacedName)
	}
	return len(sct.items) > 0
}

// UpdateServiceMapResult is the updated results after applying service changes.
type UpdateServiceMapResult struct {
	// HCServiceNodePorts is a map of Service names to node port numbers which indicate the health of that Service on this Node.
	// The value(uint16) of HCServices map is the service health check node port.
	HCServiceNodePorts map[types.NamespacedName]uint16
	// UDPStaleClusterIP holds stale (no longer assigned to a Service) Service IPs that had UDP ports.
	// Callers can use this to abort timeout-waits or clear connection-tracking information.
	UDPStaleClusterIP sets.String
}

// UpdateServiceMap updates ServiceMap based on the given changes.
func UpdateServiceMap(serviceMap ServiceMap, changes *ServiceChangeTracker) (result UpdateServiceMapResult) {
	result.UDPStaleClusterIP = sets.NewString()
	serviceMap.apply(changes, result.UDPStaleClusterIP)

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to serviceMap.
	result.HCServiceNodePorts = make(map[types.NamespacedName]uint16)
	for svcPortName, info := range serviceMap {
		if info.HealthCheckNodePort() != 0 {
			result.HCServiceNodePorts[svcPortName.NamespacedName] = uint16(info.HealthCheckNodePort())
		}
	}

	return result
}

// ServiceMap maps a service to its ServicePort information.
type ServiceMap map[ServicePortName]ServicePort

// serviceToServiceMap translates a single Service object to a ServiceMap.
// makeServicePort() return a proxy.ServicePort based on the given Service and its ServicePort.  We inject makeServicePort()
// so that giving caller side a chance to initialize proxy.ServicePort interface.
//
// NOTE: service object should NOT be modified.
func serviceToServiceMap(service *api.Service, makeServicePort func(servicePort *api.ServicePort, service *api.Service) ServicePort) ServiceMap {
	if service == nil {
		return nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxyutil.ShouldSkipService(svcName, service) {
		return nil
	}

	serviceMap := make(ServiceMap)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		serviceMap[svcPortName] = makeServicePort(servicePort, service)
	}
	return serviceMap
}

// apply the changes to ServiceMap and update the stale udp cluster IP set. The UDPStaleClusterIP argument is passed in to store the
// udp protocol service cluster ip when service is deleted from the ServiceMap.
func (serviceMap *ServiceMap) apply(changes *ServiceChangeTracker, UDPStaleClusterIP sets.String) {
	changes.lock.Lock()
	defer changes.lock.Unlock()
	for _, change := range changes.items {
		serviceMap.merge(change.current)
		// filter out the Update event of current changes from previous changes before calling unmerge() so that can
		// skip deleting the Update events.
		change.previous.filter(change.current)
		serviceMap.unmerge(change.previous, UDPStaleClusterIP)
	}
	// clear changes after applying them to ServiceMap.
	changes.items = make(map[types.NamespacedName]*serviceChange)
	return
}

// merge adds other ServiceMap's elements to current ServiceMap.
// If collision, other ALWAYS win. Otherwise add the other to current.
// In other words, if some elements in current collisions with other, update the current by other.
// It returns a string type set which stores all the newly merged services' identifier, ServicePortName.String(), to help users
// tell if a service is deleted or updated.
// The returned value is one of the arguments of ServiceMap.unmerge().
// ServiceMap A Merge ServiceMap B will do following 2 things:
//   * update ServiceMap A.
//   * produce a string set which stores all other ServiceMap's ServicePortName.String().
// For example,
//   - A{}
//   - B{{"ns", "cluster-ip", "http"}: {"172.16.55.10", 1234, "TCP"}}
//     - A updated to be {{"ns", "cluster-ip", "http"}: {"172.16.55.10", 1234, "TCP"}}
//     - produce string set {"ns/cluster-ip:http"}
//   - A{{"ns", "cluster-ip", "http"}: {"172.16.55.10", 345, "UDP"}}
//   - B{{"ns", "cluster-ip", "http"}: {"172.16.55.10", 1234, "TCP"}}
//     - A updated to be {{"ns", "cluster-ip", "http"}: {"172.16.55.10", 1234, "TCP"}}
//     - produce string set {"ns/cluster-ip:http"}
func (sm *ServiceMap) merge(other ServiceMap) sets.String {
	// existingPorts is going to store all identifiers of all services in `other` ServiceMap.
	existingPorts := sets.NewString()
	for svcPortName, info := range other {
		// Take ServicePortName.String() as the newly merged service's identifier and put it into existingPorts.
		existingPorts.Insert(svcPortName.String())
		_, exists := (*sm)[svcPortName]
		if !exists {
			glog.V(1).Infof("Adding new service port %q at %s", svcPortName, info.String())
		} else {
			glog.V(1).Infof("Updating existing service port %q at %s", svcPortName, info.String())
		}
		(*sm)[svcPortName] = info
	}
	return existingPorts
}

// filter filters out elements from ServiceMap base on given ports string sets.
func (sm *ServiceMap) filter(other ServiceMap) {
	for svcPortName := range *sm {
		// skip the delete for Update event.
		if _, ok := other[svcPortName]; ok {
			delete(*sm, svcPortName)
		}
	}
}

// unmerge deletes all other ServiceMap's elements from current ServiceMap.  We pass in the UDPStaleClusterIP strings sets
// for storing the stale udp service cluster IPs. We will clear stale udp connection base on UDPStaleClusterIP later
func (sm *ServiceMap) unmerge(other ServiceMap, UDPStaleClusterIP sets.String) {
	for svcPortName := range other {
		info, exists := (*sm)[svcPortName]
		if exists {
			glog.V(1).Infof("Removing service port %q", svcPortName)
			if info.Protocol() == api.ProtocolUDP {
				UDPStaleClusterIP.Insert(info.ClusterIP())
			}
			delete(*sm, svcPortName)
		} else {
			glog.Errorf("Service port %q doesn't exists", svcPortName)
		}
	}
}
