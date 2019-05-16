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
	"fmt"
	"net"
	"reflect"
	"strings"
	"sync"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	apiservice "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

// BaseServiceInfo contains base information that defines a service.
// This could be used directly by proxier while processing services,
// or can be used for constructing a more specific ServiceInfo struct
// defined by the proxier if needed.
type BaseServiceInfo struct {
	ClusterIP                net.IP
	Port                     int
	Protocol                 v1.Protocol
	NodePort                 int
	LoadBalancerStatus       v1.LoadBalancerStatus
	SessionAffinityType      v1.ServiceAffinity
	StickyMaxAgeSeconds      int
	ExternalIPs              []string
	LoadBalancerSourceRanges []string
	HealthCheckNodePort      int
	OnlyNodeLocalEndpoints   bool
}

var _ ServicePort = &BaseServiceInfo{}

// String is part of ServicePort interface.
func (info *BaseServiceInfo) String() string {
	return fmt.Sprintf("%s:%d/%s", info.ClusterIP, info.Port, info.Protocol)
}

// ClusterIPString is part of ServicePort interface.
func (info *BaseServiceInfo) ClusterIPString() string {
	return info.ClusterIP.String()
}

// GetProtocol is part of ServicePort interface.
func (info *BaseServiceInfo) GetProtocol() v1.Protocol {
	return info.Protocol
}

// GetHealthCheckNodePort is part of ServicePort interface.
func (info *BaseServiceInfo) GetHealthCheckNodePort() int {
	return info.HealthCheckNodePort
}

// GetNodePort is part of the ServicePort interface.
func (info *BaseServiceInfo) GetNodePort() int {
	return info.NodePort
}

// ExternalIPStrings is part of ServicePort interface.
func (info *BaseServiceInfo) ExternalIPStrings() []string {
	return info.ExternalIPs
}

// LoadBalancerIPStrings is part of ServicePort interface.
func (info *BaseServiceInfo) LoadBalancerIPStrings() []string {
	var ips []string
	for _, ing := range info.LoadBalancerStatus.Ingress {
		ips = append(ips, ing.IP)
	}
	return ips
}

func (sct *ServiceChangeTracker) newBaseServiceInfo(port *v1.ServicePort, service *v1.Service) *BaseServiceInfo {
	onlyNodeLocalEndpoints := false
	if apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}
	var stickyMaxAgeSeconds int
	if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
		// Kube-apiserver side guarantees SessionAffinityConfig won't be nil when session affinity type is ClientIP
		stickyMaxAgeSeconds = int(*service.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	}
	info := &BaseServiceInfo{
		ClusterIP: net.ParseIP(service.Spec.ClusterIP),
		Port:      int(port.Port),
		Protocol:  port.Protocol,
		NodePort:  int(port.NodePort),
		// Deep-copy in case the service instance changes
		LoadBalancerStatus:     *service.Status.LoadBalancer.DeepCopy(),
		SessionAffinityType:    service.Spec.SessionAffinity,
		StickyMaxAgeSeconds:    stickyMaxAgeSeconds,
		OnlyNodeLocalEndpoints: onlyNodeLocalEndpoints,
	}

	if sct.isIPv6Mode == nil {
		info.ExternalIPs = make([]string, len(service.Spec.ExternalIPs))
		info.LoadBalancerSourceRanges = make([]string, len(service.Spec.LoadBalancerSourceRanges))
		copy(info.LoadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
		copy(info.ExternalIPs, service.Spec.ExternalIPs)
	} else {
		// Filter out the incorrect IP version case.
		// If ExternalIPs and LoadBalancerSourceRanges on service contains incorrect IP versions,
		// only filter out the incorrect ones.
		var incorrectIPs []string
		info.ExternalIPs, incorrectIPs = utilproxy.FilterIncorrectIPVersion(service.Spec.ExternalIPs, *sct.isIPv6Mode)
		if len(incorrectIPs) > 0 {
			utilproxy.LogAndEmitIncorrectIPVersionEvent(sct.recorder, "externalIPs", strings.Join(incorrectIPs, ","), service.Namespace, service.Name, service.UID)
		}
		info.LoadBalancerSourceRanges, incorrectIPs = utilproxy.FilterIncorrectCIDRVersion(service.Spec.LoadBalancerSourceRanges, *sct.isIPv6Mode)
		if len(incorrectIPs) > 0 {
			utilproxy.LogAndEmitIncorrectIPVersionEvent(sct.recorder, "loadBalancerSourceRanges", strings.Join(incorrectIPs, ","), service.Namespace, service.Name, service.UID)
		}
	}

	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p == 0 {
			klog.Errorf("Service %s/%s has no healthcheck nodeport", service.Namespace, service.Name)
		} else {
			info.HealthCheckNodePort = int(p)
		}
	}

	return info
}

type makeServicePortFunc func(*v1.ServicePort, *v1.Service, *BaseServiceInfo) ServicePort

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
	// makeServiceInfo allows proxier to inject customized information when processing service.
	makeServiceInfo makeServicePortFunc
	// isIPv6Mode indicates if change tracker is under IPv6/IPv4 mode. Nil means not applicable.
	isIPv6Mode *bool
	recorder   record.EventRecorder
}

// NewServiceChangeTracker initializes a ServiceChangeTracker
func NewServiceChangeTracker(makeServiceInfo makeServicePortFunc, isIPv6Mode *bool, recorder record.EventRecorder) *ServiceChangeTracker {
	return &ServiceChangeTracker{
		items:           make(map[types.NamespacedName]*serviceChange),
		makeServiceInfo: makeServiceInfo,
		isIPv6Mode:      isIPv6Mode,
		recorder:        recorder,
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
func (sct *ServiceChangeTracker) Update(previous, current *v1.Service) bool {
	svc := current
	if svc == nil {
		svc = previous
	}
	// previous == nil && current == nil is unexpected, we should return false directly.
	if svc == nil {
		return false
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
	}
	metrics.ServiceChangesPending.Set(float64(len(sct.items)))
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
		if info.GetHealthCheckNodePort() != 0 {
			result.HCServiceNodePorts[svcPortName.NamespacedName] = uint16(info.GetHealthCheckNodePort())
		}
	}

	return result
}

// ServiceMap maps a service to its ServicePort.
type ServiceMap map[ServicePortName]ServicePort

// serviceToServiceMap translates a single Service object to a ServiceMap.
//
// NOTE: service object should NOT be modified.
func (sct *ServiceChangeTracker) serviceToServiceMap(service *v1.Service) ServiceMap {
	if service == nil {
		return nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if utilproxy.ShouldSkipService(svcName, service) {
		return nil
	}

	if len(service.Spec.ClusterIP) != 0 {
		// Filter out the incorrect IP version case.
		// If ClusterIP on service has incorrect IP version, service itself will be ignored.
		if sct.isIPv6Mode != nil && utilnet.IsIPv6String(service.Spec.ClusterIP) != *sct.isIPv6Mode {
			utilproxy.LogAndEmitIncorrectIPVersionEvent(sct.recorder, "clusterIP", service.Spec.ClusterIP, service.Namespace, service.Name, service.UID)
			return nil
		}
	}

	serviceMap := make(ServiceMap)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		baseSvcInfo := sct.newBaseServiceInfo(servicePort, service)
		if sct.makeServiceInfo != nil {
			serviceMap[svcPortName] = sct.makeServiceInfo(servicePort, service, baseSvcInfo)
		} else {
			serviceMap[svcPortName] = baseSvcInfo
		}
	}
	return serviceMap
}

// apply the changes to ServiceMap and update the stale udp cluster IP set. The UDPStaleClusterIP argument is passed in to store the
// udp protocol service cluster ip when service is deleted from the ServiceMap.
func (sm *ServiceMap) apply(changes *ServiceChangeTracker, UDPStaleClusterIP sets.String) {
	changes.lock.Lock()
	defer changes.lock.Unlock()
	for _, change := range changes.items {
		sm.merge(change.current)
		// filter out the Update event of current changes from previous changes before calling unmerge() so that can
		// skip deleting the Update events.
		change.previous.filter(change.current)
		sm.unmerge(change.previous, UDPStaleClusterIP)
	}
	// clear changes after applying them to ServiceMap.
	changes.items = make(map[types.NamespacedName]*serviceChange)
	metrics.ServiceChangesPending.Set(0)
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
			klog.V(1).Infof("Adding new service port %q at %s", svcPortName, info.String())
		} else {
			klog.V(1).Infof("Updating existing service port %q at %s", svcPortName, info.String())
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
			klog.V(1).Infof("Removing service port %q", svcPortName)
			if info.GetProtocol() == v1.ProtocolUDP {
				UDPStaleClusterIP.Insert(info.ClusterIPString())
			}
			delete(*sm, svcPortName)
		} else {
			klog.Errorf("Service port %q doesn't exists", svcPortName)
		}
	}
}
