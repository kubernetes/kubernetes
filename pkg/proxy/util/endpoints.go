package util

import (
	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"net"
	"reflect"
	"strconv"
	"sync"

	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// internal struct for endpoints information
type EndpointsInfo struct {
	Endpoint string // TODO: should be an endpointString type
	IsLocal  bool
}

func (e *EndpointsInfo) String() string {
	return fmt.Sprintf("%v", *e)
}

type EndpointServicePair struct {
	Endpoint        string
	ServicePortName proxy.ServicePortName
}

type ProxyEndpointsMap map[proxy.ServicePortName][]*EndpointsInfo

type endpointsChange struct {
	previous *api.Endpoints
	current  *api.Endpoints
}

type EndpointsChangeMap struct {
	sync.Mutex
	items map[types.NamespacedName]*endpointsChange
}

// <staleEndpoints> are modified by this function with detected stale
// connections.
func detectStaleConnections(oldEndpointsMap, newEndpointsMap ProxyEndpointsMap, staleEndpoints map[EndpointServicePair]bool) {
	for svcPort, epList := range oldEndpointsMap {
		for _, ep := range epList {
			stale := true
			for i := range newEndpointsMap[svcPort] {
				if *newEndpointsMap[svcPort][i] == *ep {
					stale = false
					break
				}
			}
			if stale {
				glog.V(4).Infof("Stale endpoint %v -> %v", svcPort, ep.Endpoint)
				staleEndpoints[EndpointServicePair{Endpoint: ep.Endpoint, ServicePortName: svcPort}] = true
			}
		}
	}
}

// <endpointsMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func UpdateEndpointsMap(
	endpointsMap ProxyEndpointsMap,
	changes *EndpointsChangeMap,
	hostname string) (syncRequired bool, hcEndpoints map[types.NamespacedName]int, staleSet map[EndpointServicePair]bool) {
	syncRequired = false
	staleSet = make(map[EndpointServicePair]bool)
	for _, change := range changes.items {
		oldEndpointsMap := endpointsToEndpointsMap(change.previous, hostname)
		newEndpointsMap := endpointsToEndpointsMap(change.current, hostname)
		if !reflect.DeepEqual(oldEndpointsMap, newEndpointsMap) {
			endpointsMap.Unmerge(oldEndpointsMap)
			endpointsMap.Merge(newEndpointsMap)
			detectStaleConnections(oldEndpointsMap, newEndpointsMap, staleSet)
			syncRequired = true
		}
	}
	changes.items = make(map[types.NamespacedName]*endpointsChange)

	if !utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) {
		return
	}

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.
	hcEndpoints = make(map[types.NamespacedName]int)
	localIPs := getLocalIPs(endpointsMap)
	for nsn, ips := range localIPs {
		hcEndpoints[nsn] = len(ips)
	}

	return syncRequired, hcEndpoints, staleSet
}

// Translates single Endpoints object to proxyEndpointsMap.
// This function is used for incremental updated of endpointsMap.
//
// NOTE: endpoints object should NOT be modified.
func endpointsToEndpointsMap(endpoints *api.Endpoints, hostname string) ProxyEndpointsMap {
	if endpoints == nil {
		return nil
	}

	endpointsMap := make(ProxyEndpointsMap)
	// We need to build a map of portname -> all ip:ports for that
	// portname.  Explode Endpoints.Subsets[*] into this structure.
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if port.Port == 0 {
				glog.Warningf("ignoring invalid endpoint port %s", port.Name)
				continue
			}
			svcPort := proxy.ServicePortName{
				NamespacedName: types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name},
				Port:           port.Name,
			}
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if addr.IP == "" {
					glog.Warningf("ignoring invalid endpoint port %s with empty host", port.Name)
					continue
				}
				epInfo := &EndpointsInfo{
					Endpoint: net.JoinHostPort(addr.IP, strconv.Itoa(int(port.Port))),
					IsLocal:  addr.NodeName != nil && *addr.NodeName == hostname,
				}
				endpointsMap[svcPort] = append(endpointsMap[svcPort], epInfo)
			}
			if glog.V(3) {
				newEPList := []string{}
				for _, ep := range endpointsMap[svcPort] {
					newEPList = append(newEPList, ep.Endpoint)
				}
				glog.Infof("Setting endpoints for %q to %+v", svcPort, newEPList)
			}
		}
	}
	return endpointsMap
}

func NewEndpointsChangeMap() EndpointsChangeMap {
	return EndpointsChangeMap{
		items: make(map[types.NamespacedName]*endpointsChange),
	}
}

func (ecm *EndpointsChangeMap) Update(namespacedName *types.NamespacedName, previous, current *api.Endpoints) {
	ecm.Lock()
	defer ecm.Unlock()

	change, exists := ecm.items[*namespacedName]
	if !exists {
		change = &endpointsChange{}
		change.previous = previous
		ecm.items[*namespacedName] = change
	}
	change.current = current
}

func (em ProxyEndpointsMap) Merge(other ProxyEndpointsMap) {
	for svcPort := range other {
		em[svcPort] = other[svcPort]
	}
}

func (em ProxyEndpointsMap) Unmerge(other ProxyEndpointsMap) {
	for svcPort := range other {
		delete(em, svcPort)
	}
}
