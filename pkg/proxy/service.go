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

	v1 "k8s.io/api/core/v1"
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
	clusterIP                net.IP
	port                     int
	protocol                 v1.Protocol
	nodePort                 int
	loadBalancerStatus       v1.LoadBalancerStatus
	sessionAffinityType      v1.ServiceAffinity
	stickyMaxAgeSeconds      int
	externalIPs              []string
	loadBalancerSourceRanges []string
	healthCheckNodePort      int
	onlyNodeLocalEndpoints   bool
	topologyKeys             []string
}

var _ ServicePort = &BaseServiceInfo{}

// String is part of ServicePort interface.
func (info *BaseServiceInfo) String() string {
	return fmt.Sprintf("%s:%d/%s", info.clusterIP, info.port, info.protocol)
}

// ClusterIP is part of ServicePort interface.
func (info *BaseServiceInfo) ClusterIP() net.IP {
	return info.clusterIP
}

// Port is part of ServicePort interface.
func (info *BaseServiceInfo) Port() int {
	return info.port
}

// SessionAffinityType is part of the ServicePort interface.
func (info *BaseServiceInfo) SessionAffinityType() v1.ServiceAffinity {
	return info.sessionAffinityType
}

// StickyMaxAgeSeconds is part of the ServicePort interface
func (info *BaseServiceInfo) StickyMaxAgeSeconds() int {
	return info.stickyMaxAgeSeconds
}

// Protocol is part of ServicePort interface.
func (info *BaseServiceInfo) Protocol() v1.Protocol {
	return info.protocol
}

// LoadBalancerSourceRanges is part of ServicePort interface
func (info *BaseServiceInfo) LoadBalancerSourceRanges() []string {
	return info.loadBalancerSourceRanges
}

// HealthCheckNodePort is part of ServicePort interface.
func (info *BaseServiceInfo) HealthCheckNodePort() int {
	return info.healthCheckNodePort
}

// NodePort is part of the ServicePort interface.
func (info *BaseServiceInfo) NodePort() int {
	return info.nodePort
}

// ExternalIPStrings is part of ServicePort interface.
func (info *BaseServiceInfo) ExternalIPStrings() []string {
	return info.externalIPs
}

// LoadBalancerIPStrings is part of ServicePort interface.
func (info *BaseServiceInfo) LoadBalancerIPStrings() []string {
	var ips []string
	for _, ing := range info.loadBalancerStatus.Ingress {
		ips = append(ips, ing.IP)
	}
	return ips
}

// OnlyNodeLocalEndpoints is part of ServicePort interface.
func (info *BaseServiceInfo) OnlyNodeLocalEndpoints() bool {
	return info.onlyNodeLocalEndpoints
}

// TopologyKeys is part of ServicePort interface.
func (info *BaseServiceInfo) TopologyKeys() []string {
	return info.topologyKeys
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
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(port.Port),
		protocol:  port.Protocol,
		nodePort:  int(port.NodePort),
		// Deep-copy in case the service instance changes
		loadBalancerStatus:     *service.Status.LoadBalancer.DeepCopy(),
		sessionAffinityType:    service.Spec.SessionAffinity,
		stickyMaxAgeSeconds:    stickyMaxAgeSeconds,
		onlyNodeLocalEndpoints: onlyNodeLocalEndpoints,
		topologyKeys:           service.Spec.TopologyKeys,
	}

	if sct.isIPv6Mode == nil {
		info.externalIPs = make([]string, len(service.Spec.ExternalIPs))
		info.loadBalancerSourceRanges = make([]string, len(service.Spec.LoadBalancerSourceRanges))
		copy(info.loadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
		copy(info.externalIPs, service.Spec.ExternalIPs)
	} else {
		// Filter out the incorrect IP version case.
		// If ExternalIPs and LoadBalancerSourceRanges on service contains incorrect IP versions,
		// only filter out the incorrect ones.
		var incorrectIPs []string
		info.externalIPs, incorrectIPs = utilproxy.FilterIncorrectIPVersion(service.Spec.ExternalIPs, *sct.isIPv6Mode)
		if len(incorrectIPs) > 0 {
			utilproxy.LogAndEmitIncorrectIPVersionEvent(sct.recorder, "externalIPs", strings.Join(incorrectIPs, ","), service.Namespace, service.Name, service.UID)
		}
		info.loadBalancerSourceRanges, incorrectIPs = utilproxy.FilterIncorrectCIDRVersion(service.Spec.LoadBalancerSourceRanges, *sct.isIPv6Mode)
		if len(incorrectIPs) > 0 {
			utilproxy.LogAndEmitIncorrectIPVersionEvent(sct.recorder, "loadBalancerSourceRanges", strings.Join(incorrectIPs, ","), service.Namespace, service.Name, service.UID)
		}
	}

	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p == 0 {
			klog.Errorf("Service %s/%s has no healthcheck nodeport", service.Namespace, service.Name)
		} else {
			info.healthCheckNodePort = int(p)
		}
	}

	return info
}

type makeServicePortFunc func(*v1.ServicePort, *v1.Service, *BaseServiceInfo) ServicePort

// ServiceChangeTracker keeps track of the last-synced state of the world for
// Services as well as buffering future uncommitted (to, say, iptables) changes.
type ServiceChangeTracker struct {
	lock sync.Mutex

	// lastState is the last committed state, with no pending changes applied
	// it is a map of Service to ServiceMap, so we can quickly look up all
	// ServicePorts that belong to a given service
	lastState map[types.NamespacedName]PartialServiceMap

	// pendingChanges is the set of Services (really ServicePorts) that have changed
	// since the last Commit(). Again, this is per-Service, so it's a PartialServiceMap
	pendingChanges map[types.NamespacedName]PartialServiceMap

	// makeServiceInfo allows proxier to inject customized information when processing service.
	makeServiceInfo makeServicePortFunc

	// isIPv6Mode indicates if change tracker is under IPv6/IPv4 mode. Nil means not applicable.
	isIPv6Mode *bool
	recorder   record.EventRecorder
}

// NewServiceChangeTracker initializes a ServiceChangeTracker
func NewServiceChangeTracker(makeServiceInfo makeServicePortFunc, isIPv6Mode *bool, recorder record.EventRecorder) *ServiceChangeTracker {
	return &ServiceChangeTracker{
		pendingChanges:  make(map[types.NamespacedName]PartialServiceMap),
		lastState:       make(map[types.NamespacedName]PartialServiceMap),
		makeServiceInfo: makeServiceInfo,
		isIPv6Mode:      isIPv6Mode,
		recorder:        recorder,
	}
}

// Update updates given service's change map based on the <previous, current> service pair.  It returns true if the Tracker has any pending changes.
// otherwise return false.  Update can be used to add/update/delete items of ServiceChangeMap.  For example,
// Add item
//   - pass <nil, service> as the <previous, current> pair.
// Update item
//   - pass <oldService, service> as the <previous, current> pair. oldService is ignored
// Delete item
//  - pass <oldService, nil> as the <previous, current> pair.
func (sct *ServiceChangeTracker) Update(previous, current *v1.Service) bool {
	var nsn types.NamespacedName
	if current != nil {
		nsn = types.NamespacedName{Namespace: current.Namespace, Name: current.Name}
	} else if previous != nil {
		nsn = types.NamespacedName{Namespace: previous.Namespace, Name: previous.Name}
	} else {
		return false // prev, current nil
	}

	sct.lock.Lock()
	defer sct.lock.Unlock()

	metrics.ServiceChangesTotal.Inc()

	oldState := sct.lastState[nsn]               // may be nil
	newState := sct.serviceToServiceMap(current) // may be nil

	// if newState == currentState, it means either no change or ABA, so we
	// can delete this from the list of pending changes
	if reflect.DeepEqual(oldState, newState) {
		delete(sct.pendingChanges, nsn)
	} else {
		sct.pendingChanges[nsn] = newState
	}

	metrics.ServiceChangesPending.Set(float64(len(sct.pendingChanges)))
	return len(sct.pendingChanges) > 0
}

// UpdateServiceMapResult is the updated results after applying service changes.
type UpdateServiceMapResult struct {
	ServiceMap ServiceMap

	StaleServicePorts []ServicePort
}

// UDPStaleClusterIPs holds stale (no longer assigned to a Service) Service IPs that had UDP ports.
// Callers can use this to abort timeout-waits or clear connection-tracking information.
func (smr *UpdateServiceMapResult) UDPStaleClusterIPs() sets.String {
	result := sets.NewString()

	for _, staleServicePort := range smr.StaleServicePorts {
		if staleServicePort.Protocol() == v1.ProtocolUDP {
			result.Insert(staleServicePort.ClusterIP().String())
		}
	}
	return result
}

// Commit applies the pending changes to the state. It returns the newly
// committed state as well as the list of deleted ServicePorts
func (sct *ServiceChangeTracker) Commit() UpdateServiceMapResult {
	sct.lock.Lock()
	defer sct.lock.Unlock()

	result := UpdateServiceMapResult{
		ServiceMap:        make(ServiceMap, len(sct.lastState)),
		StaleServicePorts: []ServicePort{},
	}

	// merge pending changes, compute deleted
	for serviceNamespacedName, newServicePorts := range sct.pendingChanges {
		oldServicePorts := sct.lastState[serviceNamespacedName]
		if len(newServicePorts) == 0 {
			delete(sct.lastState, serviceNamespacedName)
		} else {
			sct.lastState[serviceNamespacedName] = newServicePorts
		}

		// delete serviceports from old that are also in new, any left over
		// are stale.
		for servicePortName := range newServicePorts {
			delete(oldServicePorts, servicePortName)
		}
		for _, staleServicePort := range oldServicePorts {
			result.StaleServicePorts = append(result.StaleServicePorts, staleServicePort)
		}
	}

	// flatten list of PartialServiceMaps to the final result
	for _, sm := range sct.lastState {
		for k, v := range sm {
			result.ServiceMap[k] = v
		}
	}

	// clear pending changes
	sct.pendingChanges = make(map[types.NamespacedName]PartialServiceMap)
	metrics.ServiceChangesPending.Set(0)

	return result
}

// ServiceMap is a map of ServicePorts by their names
type ServiceMap map[ServicePortName]ServicePort

// PartialServiceMap is the same as a ServiceMap, but with a different type
// to detect coding errors.
type PartialServiceMap map[ServicePortName]ServicePort

// HCServiceNodePorts returns a map of Service to healthcheck port
func (sm ServiceMap) HCServiceNodePorts() map[types.NamespacedName]uint16 {
	result := make(map[types.NamespacedName]uint16)
	for svcPortName, info := range sm {
		if hcp := info.HealthCheckNodePort(); hcp != 0 {
			result[svcPortName.NamespacedName] = uint16(hcp)
		}
	}
	return result
}

// serviceToServiceMap translates a single Service object to a PartialServiceMap.
// The PartialServiceMap is the set of ServicePorts created by this Service.
//
// NOTE: service object should NOT be modified.
func (sct *ServiceChangeTracker) serviceToServiceMap(service *v1.Service) PartialServiceMap {
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

	serviceMap := make(PartialServiceMap)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := ServicePortName{NamespacedName: svcName, Port: servicePort.Name, Protocol: servicePort.Protocol}
		baseSvcInfo := sct.newBaseServiceInfo(servicePort, service)
		if sct.makeServiceInfo != nil {
			serviceMap[svcPortName] = sct.makeServiceInfo(servicePort, service, baseSvcInfo)
		} else {
			serviceMap[svcPortName] = baseSvcInfo
		}
	}
	return serviceMap
}
