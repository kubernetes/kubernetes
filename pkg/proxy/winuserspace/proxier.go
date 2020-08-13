/*
Copyright 2016 The Kubernetes Authors.

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

package winuserspace

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/util/netsh"
)

const allAvailableInterfaces string = ""

type portal struct {
	ip         string
	port       int
	isExternal bool
}

type serviceInfo struct {
	isAliveAtomic       int32 // Only access this with atomic ops
	portal              portal
	protocol            v1.Protocol
	socket              proxySocket
	timeout             time.Duration
	activeClients       *clientCache
	dnsClients          *dnsClientCache
	sessionAffinityType v1.ServiceAffinity
}

func (info *serviceInfo) setAlive(b bool) {
	var i int32
	if b {
		i = 1
	}
	atomic.StoreInt32(&info.isAliveAtomic, i)
}

func (info *serviceInfo) isAlive() bool {
	return atomic.LoadInt32(&info.isAliveAtomic) != 0
}

func logTimeout(err error) bool {
	if e, ok := err.(net.Error); ok {
		if e.Timeout() {
			klog.V(3).Infof("connection to endpoint closed due to inactivity")
			return true
		}
	}
	return false
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	// EndpointSlice support has not been added for this proxier yet.
	config.NoopEndpointSliceHandler
	// TODO(imroc): implement node handler for winuserspace proxier.
	config.NoopNodeHandler

	loadBalancer   LoadBalancer
	mu             sync.Mutex // protects serviceMap
	serviceMap     map[ServicePortPortalName]*serviceInfo
	syncPeriod     time.Duration
	udpIdleTimeout time.Duration
	numProxyLoops  int32 // use atomic ops to access this; mostly for testing
	netsh          netsh.Interface
	hostIP         net.IP
}

// assert Proxier is a proxy.Provider
var _ proxy.Provider = &Proxier{}

var (
	// ErrProxyOnLocalhost is returned by NewProxier if the user requests a proxier on
	// the loopback address. May be checked for by callers of NewProxier to know whether
	// the caller provided invalid input.
	ErrProxyOnLocalhost = fmt.Errorf("cannot proxy on localhost")
)

// Used below.
var localhostIPv4 = net.ParseIP("127.0.0.1")
var localhostIPv6 = net.ParseIP("::1")

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen. It is assumed that there is only a single Proxier active
// on a machine. An error will be returned if the proxier cannot be started
// due to an invalid ListenIP (loopback)
func NewProxier(loadBalancer LoadBalancer, listenIP net.IP, netsh netsh.Interface, pr utilnet.PortRange, syncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	if listenIP.Equal(localhostIPv4) || listenIP.Equal(localhostIPv6) {
		return nil, ErrProxyOnLocalhost
	}

	hostIP, err := utilnet.ChooseHostInterface()
	if err != nil {
		return nil, fmt.Errorf("failed to select a host interface: %v", err)
	}

	klog.V(2).Infof("Setting proxy IP to %v", hostIP)
	return createProxier(loadBalancer, listenIP, netsh, hostIP, syncPeriod, udpIdleTimeout)
}

func createProxier(loadBalancer LoadBalancer, listenIP net.IP, netsh netsh.Interface, hostIP net.IP, syncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	return &Proxier{
		loadBalancer:   loadBalancer,
		serviceMap:     make(map[ServicePortPortalName]*serviceInfo),
		syncPeriod:     syncPeriod,
		udpIdleTimeout: udpIdleTimeout,
		netsh:          netsh,
		hostIP:         hostIP,
	}, nil
}

// Sync is called to immediately synchronize the proxier state
func (proxier *Proxier) Sync() {
	proxier.cleanupStaleStickySessions()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	for {
		<-t.C
		klog.V(6).Infof("Periodic sync")
		proxier.Sync()
	}
}

// cleanupStaleStickySessions cleans up any stale sticky session records in the hash map.
func (proxier *Proxier) cleanupStaleStickySessions() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	servicePortNameMap := make(map[proxy.ServicePortName]bool)
	for name := range proxier.serviceMap {
		servicePortName := proxy.ServicePortName{
			NamespacedName: types.NamespacedName{
				Namespace: name.Namespace,
				Name:      name.Name,
			},
			Port: name.Port,
		}
		if !servicePortNameMap[servicePortName] {
			// ensure cleanup sticky sessions only gets called once per serviceportname
			servicePortNameMap[servicePortName] = true
			proxier.loadBalancer.CleanupStaleStickySessions(servicePortName)
		}
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service ServicePortPortalName, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service ServicePortPortalName, info *serviceInfo) error {
	delete(proxier.serviceMap, service)
	info.setAlive(false)
	err := info.socket.Close()
	return err
}

func (proxier *Proxier) getServiceInfo(service ServicePortPortalName) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service ServicePortPortalName, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServicePortPortal starts listening for a new service, returning the serviceInfo.
// The timeout only applies to UDP connections, for now.
func (proxier *Proxier) addServicePortPortal(servicePortPortalName ServicePortPortalName, protocol v1.Protocol, listenIP string, port int, timeout time.Duration) (*serviceInfo, error) {
	var serviceIP net.IP
	if listenIP != allAvailableInterfaces {
		if serviceIP = net.ParseIP(listenIP); serviceIP == nil {
			return nil, fmt.Errorf("could not parse ip '%q'", listenIP)
		}
		// add the IP address.  Node port binds to all interfaces.
		args := proxier.netshIPv4AddressAddArgs(serviceIP)
		if existed, err := proxier.netsh.EnsureIPAddress(args, serviceIP); err != nil {
			return nil, err
		} else if !existed {
			klog.V(3).Infof("Added ip address to fowarder interface for service %q at %s/%s", servicePortPortalName, net.JoinHostPort(listenIP, strconv.Itoa(port)), protocol)
		}
	}

	// add the listener, proxy
	sock, err := newProxySocket(protocol, serviceIP, port)
	if err != nil {
		return nil, err
	}
	si := &serviceInfo{
		isAliveAtomic: 1,
		portal: portal{
			ip:         listenIP,
			port:       port,
			isExternal: false,
		},
		protocol:            protocol,
		socket:              sock,
		timeout:             timeout,
		activeClients:       newClientCache(),
		dnsClients:          newDNSClientCache(),
		sessionAffinityType: v1.ServiceAffinityNone, // default
	}
	proxier.setServiceInfo(servicePortPortalName, si)

	klog.V(2).Infof("Proxying for service %q at %s/%s", servicePortPortalName, net.JoinHostPort(listenIP, strconv.Itoa(port)), protocol)
	go func(service ServicePortPortalName, proxier *Proxier) {
		defer runtime.HandleCrash()
		atomic.AddInt32(&proxier.numProxyLoops, 1)
		sock.ProxyLoop(service, si, proxier)
		atomic.AddInt32(&proxier.numProxyLoops, -1)
	}(servicePortPortalName, proxier)

	return si, nil
}

func (proxier *Proxier) closeServicePortPortal(servicePortPortalName ServicePortPortalName, info *serviceInfo) error {
	// turn off the proxy
	if err := proxier.stopProxy(servicePortPortalName, info); err != nil {
		return err
	}

	// close the PortalProxy by deleting the service IP address
	if info.portal.ip != allAvailableInterfaces {
		serviceIP := net.ParseIP(info.portal.ip)
		args := proxier.netshIPv4AddressDeleteArgs(serviceIP)
		if err := proxier.netsh.DeleteIPAddress(args); err != nil {
			return err
		}
	}
	return nil
}

// getListenIPPortMap returns a slice of all listen IPs for a service.
func getListenIPPortMap(service *v1.Service, listenPort int, nodePort int) map[string]int {
	listenIPPortMap := make(map[string]int)
	listenIPPortMap[service.Spec.ClusterIP] = listenPort

	for _, ip := range service.Spec.ExternalIPs {
		listenIPPortMap[ip] = listenPort
	}

	for _, ingress := range service.Status.LoadBalancer.Ingress {
		listenIPPortMap[ingress.IP] = listenPort
	}

	if nodePort != 0 {
		listenIPPortMap[allAvailableInterfaces] = nodePort
	}

	return listenIPPortMap
}

func (proxier *Proxier) mergeService(service *v1.Service) map[ServicePortPortalName]bool {
	if service == nil {
		return nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if !helper.IsServiceIPSet(service) {
		klog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
		return nil
	}
	existingPortPortals := make(map[ServicePortPortalName]bool)

	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		// create a slice of all the source IPs to use for service port portals
		listenIPPortMap := getListenIPPortMap(service, int(servicePort.Port), int(servicePort.NodePort))
		protocol := servicePort.Protocol

		for listenIP, listenPort := range listenIPPortMap {
			servicePortPortalName := ServicePortPortalName{
				NamespacedName: svcName,
				Port:           servicePort.Name,
				PortalIPName:   listenIP,
			}
			existingPortPortals[servicePortPortalName] = true
			info, exists := proxier.getServiceInfo(servicePortPortalName)
			if exists && sameConfig(info, service, protocol, listenPort) {
				// Nothing changed.
				continue
			}
			if exists {
				klog.V(4).Infof("Something changed for service %q: stopping it", servicePortPortalName)
				if err := proxier.closeServicePortPortal(servicePortPortalName, info); err != nil {
					klog.Errorf("Failed to close service port portal %q: %v", servicePortPortalName, err)
				}
			}
			klog.V(1).Infof("Adding new service %q at %s/%s", servicePortPortalName, net.JoinHostPort(listenIP, strconv.Itoa(listenPort)), protocol)
			info, err := proxier.addServicePortPortal(servicePortPortalName, protocol, listenIP, listenPort, proxier.udpIdleTimeout)
			if err != nil {
				klog.Errorf("Failed to start proxy for %q: %v", servicePortPortalName, err)
				continue
			}
			info.sessionAffinityType = service.Spec.SessionAffinity
			klog.V(10).Infof("info: %#v", info)
		}
		if len(listenIPPortMap) > 0 {
			// only one loadbalancer per service port portal
			servicePortName := proxy.ServicePortName{
				NamespacedName: types.NamespacedName{
					Namespace: service.Namespace,
					Name:      service.Name,
				},
				Port: servicePort.Name,
			}
			timeoutSeconds := 0
			if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
				timeoutSeconds = int(*service.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
			}
			proxier.loadBalancer.NewService(servicePortName, service.Spec.SessionAffinity, timeoutSeconds)
		}
	}

	return existingPortPortals
}

func (proxier *Proxier) unmergeService(service *v1.Service, existingPortPortals map[ServicePortPortalName]bool) {
	if service == nil {
		return
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if !helper.IsServiceIPSet(service) {
		klog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
		return
	}

	servicePortNameMap := make(map[proxy.ServicePortName]bool)
	for name := range existingPortPortals {
		servicePortName := proxy.ServicePortName{
			NamespacedName: types.NamespacedName{
				Namespace: name.Namespace,
				Name:      name.Name,
			},
			Port: name.Port,
		}
		servicePortNameMap[servicePortName] = true
	}

	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		serviceName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		// create a slice of all the source IPs to use for service port portals
		listenIPPortMap := getListenIPPortMap(service, int(servicePort.Port), int(servicePort.NodePort))

		for listenIP := range listenIPPortMap {
			servicePortPortalName := ServicePortPortalName{
				NamespacedName: svcName,
				Port:           servicePort.Name,
				PortalIPName:   listenIP,
			}
			if existingPortPortals[servicePortPortalName] {
				continue
			}

			klog.V(1).Infof("Stopping service %q", servicePortPortalName)
			info, exists := proxier.getServiceInfo(servicePortPortalName)
			if !exists {
				klog.Errorf("Service %q is being removed but doesn't exist", servicePortPortalName)
				continue
			}

			if err := proxier.closeServicePortPortal(servicePortPortalName, info); err != nil {
				klog.Errorf("Failed to close service port portal %q: %v", servicePortPortalName, err)
			}
		}

		// Only delete load balancer if all listen ips per name/port show inactive.
		if !servicePortNameMap[serviceName] {
			proxier.loadBalancer.DeleteService(serviceName)
		}
	}
}

// OnServiceAdd is called whenever creation of new service object
// is observed.
func (proxier *Proxier) OnServiceAdd(service *v1.Service) {
	_ = proxier.mergeService(service)
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *Proxier) OnServiceUpdate(oldService, service *v1.Service) {
	existingPortPortals := proxier.mergeService(service)
	proxier.unmergeService(oldService, existingPortPortals)
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *Proxier) OnServiceDelete(service *v1.Service) {
	proxier.unmergeService(service, map[ServicePortPortalName]bool{})
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnServiceSynced() {
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *Proxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	proxier.loadBalancer.OnEndpointsAdd(endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	proxier.loadBalancer.OnEndpointsUpdate(oldEndpoints, endpoints)
}

// OnEndpointsDelete is called whenever deletion of an existing endpoints
// object is observed.
func (proxier *Proxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	proxier.loadBalancer.OnEndpointsDelete(endpoints)
}

// OnEndpointsSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointsSynced() {
	proxier.loadBalancer.OnEndpointsSynced()
}

func sameConfig(info *serviceInfo, service *v1.Service, protocol v1.Protocol, listenPort int) bool {
	return info.protocol == protocol && info.portal.port == listenPort && info.sessionAffinityType == service.Spec.SessionAffinity
}

func isTooManyFDsError(err error) bool {
	return strings.Contains(err.Error(), "too many open files")
}

func isClosedError(err error) bool {
	// A brief discussion about handling closed error here:
	// https://code.google.com/p/go/issues/detail?id=4373#c14
	// TODO: maybe create a stoppable TCP listener that returns a StoppedError
	return strings.HasSuffix(err.Error(), "use of closed network connection")
}

func (proxier *Proxier) netshIPv4AddressAddArgs(destIP net.IP) []string {
	intName := proxier.netsh.GetInterfaceToAddIP()
	args := []string{
		"interface", "ipv4", "add", "address",
		"name=" + intName,
		"address=" + destIP.String(),
	}

	return args
}

func (proxier *Proxier) netshIPv4AddressDeleteArgs(destIP net.IP) []string {
	intName := proxier.netsh.GetInterfaceToAddIP()
	args := []string{
		"interface", "ipv4", "delete", "address",
		"name=" + intName,
		"address=" + destIP.String(),
	}

	return args
}
