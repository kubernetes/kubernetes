/*
Copyright 2019 The Kubernetes Authors.

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

package metaproxier

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/util/async"

	"k8s.io/apimachinery/pkg/util/sets"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilnet "k8s.io/utils/net"
)

type localPortWithFamily struct {
	localPort  utilproxy.LocalPort
	family     v1.IPFamily
	serviceKey string
}

type metaProxier struct {
	mu sync.Mutex

	hostname string
	// used to publish events in case of failures to hold
	// alternative family port open
	recorder record.EventRecorder

	// periods for syncing open ports
	syncPeriod    time.Duration
	minSyncPeriod time.Duration

	// port syncing runner
	syncRunner *async.BoundedFrequencyRunner

	// cidrs that node port operates on
	nodePortAddresses []string

	// current ports held (alternative families)
	portsMap map[utilproxy.LocalPort]utilproxy.Closeable

	// interface for net libs to operate on
	networkInterfacer utilproxy.NetworkInterfacer

	// node address
	nodeAddresses sets.String
	// actual, wrapped
	ipv4Proxier proxy.Provider
	// actual, wrapped
	ipv6Proxier proxy.Provider
	// service that metaproxier operates on
	services map[string]*v1.Service
	// TODO(imroc): implement node handler for meta proxier.
	config.NoopNodeHandler
}

// NewMetaProxier returns a dual-stack "meta-proxier". Proxier API
// calls will be dispatched to the ProxyProvider instances depending
// on address family.
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider, hostname string, nodePortAddresses []string, syncPeriod time.Duration, minSyncPeriod time.Duration, recorder record.EventRecorder) proxy.Provider {
	metaProxy := &metaProxier{
		hostname:          hostname,
		recorder:          recorder,
		syncPeriod:        syncPeriod,
		minSyncPeriod:     minSyncPeriod,
		nodePortAddresses: nodePortAddresses,
		networkInterfacer: utilproxy.RealNetwork{}, // must use an alternative during tests

		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
		services:    make(map[string]*v1.Service),
	}
	burstSyncs := 2
	metaProxy.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", metaProxy.syncPorts, minSyncPeriod, syncPeriod, burstSyncs)

	provider := proxy.Provider(metaProxy)
	return provider
}

func (proxier *metaProxier) addService(svc *v1.Service) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.services[getServiceKey(svc)] = svc
}

func (proxier *metaProxier) updateService(svc *v1.Service) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.services[getServiceKey(svc)] = svc
}

func (proxier *metaProxier) deleteService(svc *v1.Service) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	delete(proxier.services, getServiceKey(svc))
}

// getProxierByIPFamily returns the proxy selected for a specific ipfamily
func (proxier *metaProxier) getProxierByIPFamily(ipFamily v1.IPFamily) proxy.Provider {
	if ipFamily == v1.IPv4Protocol {
		return proxier.ipv4Proxier
	}

	return proxier.ipv6Proxier
}

//getProxierByClusterIP gets proxy by using identifying the ipFamily of ClusterIP
func (proxier *metaProxier) getProxierByClusterIP(service *v1.Service) proxy.Provider {
	ipFamily := v1.IPv4Protocol
	if utilnet.IsIPv6String(service.Spec.ClusterIP) {
		ipFamily = v1.IPv6Protocol
	}

	return proxier.getProxierByIPFamily(ipFamily)
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules.
func (proxier *metaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

// SyncLoop runs periodic work.  This is expected to run as a
// goroutine or as the main loop of the app.  It does not return.
func (proxier *metaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop() // Use go-routine here!
	proxier.ipv4Proxier.SyncLoop()    // never returns
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}

	// we need to force a pre port sync before we call actuals. if actual raced
	// to lock a port that we hold (for a service that was recently deleted)
	// it will fail since we are holding it. so we sync our ports then call
	// let actual know about the service update
	// note: we do release port in a lazy way (no forced sync on delete)
	proxier.addService(service)
	proxier.syncPorts()

	// this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(service)
		actual.OnServiceAdd(service)
		return
	}

	for _, ipFamily := range service.Spec.IPFamilies {
		actual := proxier.getProxierByIPFamily(ipFamily)
		actual.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}

	// we need to force a pre port sync before we call actuals. if actual raced
	// to lock a port that we hold (service was single stack. then upgraded)
	// it will fail since we are holding it. so we sync our ports then call
	// let actual know about the service update
	proxier.updateService(service)
	proxier.syncPorts()

	// case zero: this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(oldService)
		actual.OnServiceUpdate(oldService, service)
		return
	}

	// case one: something has changed, but not families
	// call update on all families the service carries.
	if len(oldService.Spec.IPFamilies) == len(service.Spec.IPFamilies) {
		for _, ipFamily := range service.Spec.IPFamilies {
			actual := proxier.getProxierByIPFamily(ipFamily)
			actual.OnServiceUpdate(oldService, service)
		}

		klog.V(4).Infof("service %s/%s has been updated but no ip family change detected", service.Namespace, service.Name)
		return
	}
	// while apiserver does not allow changing primary ipfamily
	// we use the below approach to stay on the safe side.
	// note: in all cases, we check all families just
	// in case the service moved from ExternalName => ClusterIP

	// case two: service was upgraded (+1 ipFamily)
	// call add for new family
	// call update for existing family
	// note: Service might have been upgraded and
	// had port/toplogy keys etc  changed.
	if len(service.Spec.IPFamilies) > len(oldService.Spec.IPFamilies) {
		found := false
		for _, newSvcIPFamily := range service.Spec.IPFamilies {
			for _, existingSvcIPFamily := range oldService.Spec.IPFamilies {
				if newSvcIPFamily == existingSvcIPFamily {
					found = true
					break
				}
			}

			actual := proxier.getProxierByIPFamily(newSvcIPFamily)
			if found {
				actual.OnServiceUpdate(oldService, service)
			} else {
				klog.V(4).Infof("service %s/%s has been updated and ipfamily %v was added", service.Namespace, service.Name, newSvcIPFamily)
				actual.OnServiceAdd(service)
			}
		}

		return
	}

	// case three: service was downgraded
	// call delete for removed family
	// call update for existing family
	if len(service.Spec.IPFamilies) < len(oldService.Spec.IPFamilies) {
		found := false
		for _, existingSvcIPFamily := range oldService.Spec.IPFamilies {
			for _, newSvcIPFamily := range service.Spec.IPFamilies {
				if newSvcIPFamily == existingSvcIPFamily {
					found = true
					break
				}
			}

			actual := proxier.getProxierByIPFamily(existingSvcIPFamily)
			if found {
				actual.OnServiceUpdate(oldService, service)
			} else {
				klog.V(4).Infof("service %s/%s has been updated and ipfamily %v was was removed", service.Namespace, service.Name, existingSvcIPFamily)
				actual.OnServiceDelete(service)
			}
		}

		return
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	if utilproxy.ShouldSkipService(service) {
		return
	}

	// we don't need to force a port sync here
	// the assumption is, if ports was reused
	// then it will be either visibile in onAdd*
	// or regular sync.
	proxier.deleteService(service)

	// this allows skew between new proxy and old apiserver
	if len(service.Spec.IPFamilies) == 0 {
		actual := proxier.getProxierByClusterIP(service)
		actual.OnServiceDelete(service)
		return
	}

	for _, ipFamily := range service.Spec.IPFamilies {
		actual := proxier.getProxierByIPFamily(ipFamily)
		actual.OnServiceDelete(service)
	}
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *metaProxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to add endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}
	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsAdd(endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsAdd(endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *metaProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to update endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}

	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
}

// OnEndpointsDelete is called whenever deletion of an existing
// endpoints object is observed.
func (proxier *metaProxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	ipFamily, err := endpointsIPFamily(endpoints)
	if err != nil {
		klog.V(4).Infof("failed to delete endpoints %s/%s with error %v", endpoints.ObjectMeta.Namespace, endpoints.ObjectMeta.Name, err)
		return
	}

	if *ipFamily == v1.IPv4Protocol {
		proxier.ipv4Proxier.OnEndpointsDelete(endpoints)
		return
	}
	proxier.ipv6Proxier.OnEndpointsDelete(endpoints)
}

// OnEndpointsSynced is called once all the initial event handlers
// were called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnEndpointsSynced() {
	proxier.ipv4Proxier.OnEndpointsSynced()
	proxier.ipv6Proxier.OnEndpointsSynced()
}

// TODO: (khenidak) implement EndpointSlice handling

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
	default:
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", endpointSlice.AddressType)
	}
}

// OnEndpointSliceUpdate is called whenever modification of an existing endpoint
// slice object is observed.
func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	default:
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", newEndpointSlice.AddressType)
	}
}

// OnEndpointSliceDelete is called whenever deletion of an existing endpoint slice
// object is observed.
func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
	default:
		klog.V(4).Infof("EndpointSlice address type not supported by kube-proxy: %s", endpointSlice.AddressType)
	}
}

// OnEndpointSlicesSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *metaProxier) OnEndpointSlicesSynced() {
	proxier.ipv4Proxier.OnEndpointSlicesSynced()
	proxier.ipv6Proxier.OnEndpointSlicesSynced()
}

// endpointsIPFamily that returns IPFamily of endpoints or error if
// failed to identify the IP family.
func endpointsIPFamily(endpoints *v1.Endpoints) (*v1.IPFamily, error) {
	if len(endpoints.Subsets) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (no subsets)")
	}

	// we only need to work with subset [0],endpoint controller
	// ensures that endpoints selected are of the same family.
	subset := endpoints.Subsets[0]
	if len(subset.Addresses) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (no addresses)")
	}
	// same apply on addresses
	address := subset.Addresses[0]
	if len(address.IP) == 0 {
		return nil, fmt.Errorf("failed to identify ipfamily for endpoints (address has no ip)")
	}

	ipv4 := v1.IPv4Protocol
	ipv6 := v1.IPv6Protocol
	if utilnet.IsIPv6String(address.IP) {
		return &ipv6, nil
	}

	return &ipv4, nil
}

// returns a consistent key for service
func getServiceKey(svc *v1.Service) string {
	key := "%s/%s"
	return fmt.Sprintf(key, svc.Namespace, svc.Name)
}

func getServiceAlterantiveIPFamily(svc *v1.Service) []v1.IPFamily {
	allIPFamilies := map[v1.IPFamily]int{v1.IPv4Protocol: 0, v1.IPv6Protocol: 0}
	svcIPFamilies := make([]v1.IPFamily, 0)
	// get families for service

	if len(svc.Spec.IPFamilies) == 0 {
		if utilnet.IsIPv6String(svc.Spec.ClusterIP) {
			svcIPFamilies = append(svcIPFamilies, v1.IPv6Protocol)
		} else {
			svcIPFamilies = append(svcIPFamilies, v1.IPv4Protocol)
		}
	} else {
		svcIPFamilies = svc.Spec.IPFamilies
	}
	// filter them out
	for _, family := range svcIPFamilies {
		delete(allIPFamilies, family)
	}

	filtered := make([]v1.IPFamily, 0)
	for family := range allIPFamilies {
		filtered = append(filtered, family)
	}

	return filtered
}

// getLocalPortsForServices return ports to open for services
func (proxier *metaProxier) getLocalPortsForServices() []localPortWithFamily {
	localPorts := make([]localPortWithFamily, 0)

	if proxier.nodeAddresses == nil {
		nodeAddresses, err := utilproxy.GetNodeAddresses(proxier.nodePortAddresses, proxier.networkInterfacer)
		if err != nil {
			klog.Infof("Failed to get node ip address matching nodeport cidrs %v, services with nodeport may not work as intended: %v", proxier.nodePortAddresses, err)
			proxier.nodeAddresses = sets.NewString() // empty
		} else {
			proxier.nodeAddresses = nodeAddresses
		}
	}
	if len(proxier.nodeAddresses) == 0 {
		return nil // if we don't have node address to operate on, then we can not do node ports
	}

	// now that we have node address, we need to collect node ports for each service
	for svcKey, svc := range proxier.services {

		svcAltFamilies := getServiceAlterantiveIPFamily(svc)
		if len(svcAltFamilies) == 0 {
			continue // no need to process this service, it has two families
		}

		// for each port on service
		for _, svcPort := range svc.Spec.Ports {

			// if and only if it has a node port
			if svcPort.NodePort == 0 {
				continue
			}

			// if and only if it is not sctp
			if svcPort.Protocol == v1.ProtocolSCTP {
				continue
			}
			// for each family we need to lock port for
			for _, family := range svcAltFamilies {
				// open a port for each of the local node address
				for nodeAddress := range proxier.nodeAddresses {

					// if and only if node address match the family
					if (family == v1.IPv6Protocol) != utilnet.IsIPv6String(nodeAddress) {
						continue
					}

					lp := localPortWithFamily{
						localPort: utilproxy.LocalPort{
							Description: fmt.Sprintf("node port[%v] for %s on %v", svcPort.NodePort, svcKey, nodeAddress),
							IP:          nodeAddress,
							Port:        int(svcPort.NodePort),
							Protocol:    strings.ToLower(string(svcPort.Protocol)),
						},

						family:     family,
						serviceKey: getServiceKey(svc),
					}
					localPorts = append(localPorts, lp)
				}
			}
		}
	}

	return localPorts
}

//syncPorts ensures that ports are held across ip families
// even when services specifically didn't ask for. This ensure
// that services that can upgrade to dual stack knowing that the port
// will always be ready for them
func (proxier *metaProxier) syncPorts() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	newPortsList := make(map[utilproxy.LocalPort]utilproxy.Closeable)
	portsForServices := proxier.getLocalPortsForServices()
	for _, portToCheck := range portsForServices {
		// if port was already open by this service || other services
		if _, ok := proxier.portsMap[portToCheck.localPort]; !ok {
			closable, err := openLocalPort(&portToCheck.localPort, portToCheck.family == v1.IPv6Protocol)
			if err != nil {
				msg := fmt.Sprintf("meta proxier failed to open port on alternative family (%v) for service (%v) port:%v", portToCheck.family, portToCheck.serviceKey, portToCheck.localPort.String())
				// log to node
				klog.V(2).Infof(msg)

				// log to node object
				proxier.recorder.Eventf(
					&v1.ObjectReference{
						Kind:      "Node",
						Name:      proxier.hostname,
						UID:       types.UID(proxier.hostname),
						Namespace: "",
					}, v1.EventTypeWarning, err.Error(), msg)
				continue
			}
			newPortsList[portToCheck.localPort] = closable

		} else {
			// copy closable around
			newPortsList[portToCheck.localPort] = proxier.portsMap[portToCheck.localPort]
		}
	}

	// check exist ports, if they are not in new ports, then they need
	// to be closed
	for openPort, closable := range proxier.portsMap {
		if _, ok := newPortsList[openPort]; !ok {
			// close it
			closable.Close()
		}
	}

	// reset to current list
	proxier.portsMap = newPortsList
}

// copy from iptables proxier
func openLocalPort(lp *utilproxy.LocalPort, isIPv6 bool) (utilproxy.Closeable, error) {
	// For ports on node IPs, open the actual port and hold it, even though we
	// use iptables to redirect traffic.
	// This ensures a) that it's safe to use that port and b) that (a) stays
	// true.  The risk is that some process on the node (e.g. sshd or kubelet)
	// is using a port and we give that same port out to a Service.  That would
	// be bad because iptables would silently claim the traffic but the process
	// would never know.
	// NOTE: We should not need to have a real listen()ing socket - bind()
	// should be enough, but I can't figure out a way to e2e test without
	// it.  Tools like 'ss' and 'netstat' do not show sockets that are
	// bind()ed but not listen()ed, and at least the default debian netcat
	// has no way to avoid about 10 seconds of retries.
	var socket utilproxy.Closeable
	switch lp.Protocol {
	case "tcp":
		network := "tcp4"
		if isIPv6 {
			network = "tcp6"
		}
		listener, err := net.Listen(network, net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port)))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		network := "udp4"
		if isIPv6 {
			network = "udp6"
		}
		addr, err := net.ResolveUDPAddr(network, net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP(network, addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", lp.Protocol)
	}
	klog.V(2).Infof("Opened local port %s", lp.String())
	return socket, nil
}
