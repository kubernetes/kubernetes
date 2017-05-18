package util

import (
	"net"
	"reflect"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	"k8s.io/kubernetes/pkg/proxy"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/features"

	"github.com/golang/glog"
)

type ProxyServiceMap map[proxy.ServicePortName]*ServiceInfo

// internal struct for string service information
type ServiceInfo struct {
	ClusterIP                net.IP
	Port                     int
	Protocol                 api.Protocol
	NodePort                 int
	LoadBalancerStatus       api.LoadBalancerStatus
	SessionAffinityType      api.ServiceAffinity
	StickyMaxAgeMinutes      int
	ExternalIPs              []string
	LoadBalancerSourceRanges []string
	OnlyNodeLocalEndpoints   bool
	HealthCheckNodePort      int
}

// <serviceMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func UpdateServiceMap(
	serviceMap ProxyServiceMap,
	changes *ServiceChangeMap) (syncRequired bool, hcServices map[types.NamespacedName]uint16, staleServices sets.String) {
	syncRequired = false
	staleServices = sets.NewString()

	for _, change := range changes.items {
		mergeSyncRequired, existingPorts := serviceMap.mergeService(change.current)
		unmergeSyncRequired := serviceMap.unmergeService(change.previous, existingPorts, staleServices)
		syncRequired = syncRequired || mergeSyncRequired || unmergeSyncRequired
	}
	changes.items = make(map[types.NamespacedName]*ServiceChange)

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to serviceMap.
	hcServices = make(map[types.NamespacedName]uint16)
	for svcPort, info := range serviceMap {
		if info.HealthCheckNodePort != 0 {
			hcServices[svcPort.NamespacedName] = uint16(info.HealthCheckNodePort)
		}
	}

	return syncRequired, hcServices, staleServices
}

// returns a new serviceInfo struct
func newServiceInfo(serviceName proxy.ServicePortName, port *api.ServicePort, service *api.Service) *ServiceInfo {
	onlyNodeLocalEndpoints := false
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) &&
		apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}
	info := &ServiceInfo{
		ClusterIP: net.ParseIP(service.Spec.ClusterIP),
		Port:      int(port.Port),
		Protocol:  port.Protocol,
		NodePort:  int(port.NodePort),
		// Deep-copy in case the service instance changes
		LoadBalancerStatus:       *helper.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer),
		SessionAffinityType:      service.Spec.SessionAffinity,
		StickyMaxAgeMinutes:      180, // TODO: paramaterize this in the API.
		ExternalIPs:              make([]string, len(service.Spec.ExternalIPs)),
		LoadBalancerSourceRanges: make([]string, len(service.Spec.LoadBalancerSourceRanges)),
		OnlyNodeLocalEndpoints:   onlyNodeLocalEndpoints,
	}
	copy(info.LoadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
	copy(info.ExternalIPs, service.Spec.ExternalIPs)

	if apiservice.NeedsHealthCheck(service) {
		p := apiservice.GetServiceHealthCheckNodePort(service)
		if p == 0 {
			glog.Errorf("Service %q has no healthcheck nodeport", serviceName)
		} else {
			info.HealthCheckNodePort = int(p)
		}
	}

	return info
}

func (sm *ProxyServiceMap) mergeService(service *api.Service) (bool, sets.String) {
	if service == nil {
		return false, nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if ShouldSkipService(svcName, service) {
		return false, nil
	}
	syncRequired := false
	existingPorts := sets.NewString()
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		serviceName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		existingPorts.Insert(servicePort.Name)
		info := newServiceInfo(serviceName, servicePort, service)
		oldInfo, exists := (*sm)[serviceName]
		equal := reflect.DeepEqual(info, oldInfo)
		if exists {
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, info.ClusterIP, servicePort.Port, servicePort.Protocol)
		} else if !equal {
			glog.V(1).Infof("Updating existing service %q at %s:%d/%s", serviceName, info.ClusterIP, servicePort.Port, servicePort.Protocol)
		}
		if !equal {
			(*sm)[serviceName] = info
			syncRequired = true
		}
	}
	return syncRequired, existingPorts
}

// <staleServices> are modified by this function with detected stale services.
func (sm *ProxyServiceMap) unmergeService(service *api.Service, existingPorts, staleServices sets.String) bool {
	if service == nil {
		return false
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if ShouldSkipService(svcName, service) {
		return false
	}
	syncRequired := false
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if existingPorts.Has(servicePort.Name) {
			continue
		}
		serviceName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		info, exists := (*sm)[serviceName]
		if exists {
			glog.V(1).Infof("Removing service %q", serviceName)
			if info.Protocol == api.ProtocolUDP {
				staleServices.Insert(info.ClusterIP.String())
			}
			delete(*sm, serviceName)
			syncRequired = true
		} else {
			glog.Errorf("Service %q removed, but doesn't exists", serviceName)
		}
	}
	return syncRequired
}

type ServiceChangeMap struct {
	sync.Mutex
	items map[types.NamespacedName]*ServiceChange
}

type ServiceChange struct {
	previous *api.Service
	current  *api.Service
}

func NewServiceChangeMap() ServiceChangeMap {
	return ServiceChangeMap{
		items: make(map[types.NamespacedName]*ServiceChange),
	}
}

func (scm *ServiceChangeMap) Update(namespacedName *types.NamespacedName, previous, current *api.Service) {
	scm.Lock()
	defer scm.Unlock()

	change, exists := scm.items[*namespacedName]
	if !exists {
		change = &ServiceChange{}
		change.previous = previous
		scm.items[*namespacedName] = change
	}
	change.current = current
}
