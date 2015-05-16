/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"
	"github.com/golang/glog"
)

type serviceInfo struct {
	portalIP            net.IP
	portalPort          int
	protocol            api.Protocol
	proxyPort           int
	socket              proxySocket
	timeout             time.Duration
	publicIPs           []string // TODO: make this net.IP
	sessionAffinityType api.AffinityType
	stickyMaxAgeMinutes int
}

func logTimeout(err error) bool {
	if e, ok := err.(net.Error); ok {
		if e.Timeout() {
			glog.V(3).Infof("connection to endpoint closed due to inactivity")
			return true
		}
	}
	return false
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer  LoadBalancer
	mu            sync.Mutex // protects serviceMap
	serviceMap    map[ServicePortName]*serviceInfo
	numProxyLoops int32 // use atomic ops to access this; mostly for testing
	portalManager PortalManager
	*ProxierIPs
}

type ProxierIPs struct {
	listenIP net.IP
	hostIP   net.IP
}

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen.  Because of the iptables logic, It is assumed that there
// is only a single Proxier active on a machine.
func NewProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface) *Proxier {
	if listenIP.Equal(localhostIPv4) || listenIP.Equal(localhostIPv6) {
		glog.Errorf("Can't proxy only on localhost - iptables can't do it")
		return nil
	}

	hostIP, err := util.ChooseHostInterface()
	if err != nil {
		glog.Errorf("Failed to select a host interface: %v", err)
		return nil
	}
	glog.Infof("Setting Proxy IP to %v", hostIP)
	return createProxier(loadBalancer, listenIP, iptables, hostIP)
}

func createProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface, hostIP net.IP) *Proxier {
	glog.Infof("Initializing iptables")
	portalManager := NewIptablesPortalManager(iptables)
	// Clean up old messes.  Ignore erors.
	portalManager.DeleteOld()
	// Set up the iptables foundations we need.
	if err := portalManager.Init(); err != nil {
		glog.Errorf("Failed to initialize iptables: %v", err)
		return nil
	}
	// Flush old iptables rules (since the bound ports will be invalid after a restart).
	// When OnUpdate() is first called, the rules will be recreated.
	if err := portalManager.Flush(); err != nil {
		glog.Errorf("Failed to flush iptables: %v", err)
		return nil
	}
	return &Proxier{
		loadBalancer:  loadBalancer,
		serviceMap:    make(map[ServicePortName]*serviceInfo),
		portalManager: portalManager,
		ProxierIPs: &ProxierIPs{
			listenIP: listenIP,
			hostIP:   hostIP,
		},
	}
}

// The periodic interval for checking the state of things.
const syncInterval = 5 * time.Second

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(syncInterval)
	defer t.Stop()
	for {
		<-t.C
		glog.V(3).Infof("Periodic sync")
		if err := proxier.portalManager.Init(); err != nil {
			glog.Errorf("Failed to ensure iptables: %v", err)
		}
		proxier.ensurePortals()
		proxier.cleanupStaleStickySessions()
	}
}

// Ensure that portals exist for all services.
func (proxier *Proxier) ensurePortals() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	// NB: This does not remove rules that should not be present.
	for name, info := range proxier.serviceMap {
		err := proxier.portalManager.OpenPortal(proxier.ProxierIPs, name, info)
		if err != nil {
			glog.Errorf("Failed to ensure portal for %q: %v", name, err)
		}
	}
}

// clean up any stale sticky session records in the hash map.
func (proxier *Proxier) cleanupStaleStickySessions() {
	for name := range proxier.serviceMap {
		proxier.loadBalancer.CleanupStaleStickySessions(name)
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service ServicePortName, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service ServicePortName, info *serviceInfo) error {
	delete(proxier.serviceMap, service)
	return info.socket.Close()
}

func (proxier *Proxier) getServiceInfo(service ServicePortName) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service ServicePortName, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServiceOnPort starts listening for a new service, returning the serviceInfo.
// Pass proxyPort=0 to allocate a random port. The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnPort(service ServicePortName, protocol api.Protocol, proxyPort int, timeout time.Duration) (*serviceInfo, error) {
	sock, err := newProxySocket(protocol, proxier.listenIP, proxyPort)
	if err != nil {
		return nil, err
	}
	_, portStr, err := net.SplitHostPort(sock.Addr().String())
	if err != nil {
		sock.Close()
		return nil, err
	}
	portNum, err := strconv.Atoi(portStr)
	if err != nil {
		sock.Close()
		return nil, err
	}
	si := &serviceInfo{
		proxyPort:           portNum,
		protocol:            protocol,
		socket:              sock,
		timeout:             timeout,
		sessionAffinityType: api.AffinityTypeNone, // default
		stickyMaxAgeMinutes: 180,                  // TODO: paramaterize this in the API.
	}
	proxier.setServiceInfo(service, si)

	glog.V(1).Infof("Proxying for service %q on %s port %d", service, protocol, portNum)
	go func(service ServicePortName, proxier *Proxier) {
		defer util.HandleCrash()
		atomic.AddInt32(&proxier.numProxyLoops, 1)
		sock.ProxyLoop(service, si, proxier)
		atomic.AddInt32(&proxier.numProxyLoops, -1)
	}(service, proxier)

	return si, nil
}

// How long we leave idle UDP connections open.
const udpIdleTimeout = 10 * time.Second

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.V(4).Infof("Received update notice: %+v", services)
	activeServices := make(map[ServicePortName]bool) // use a map as a set
	for i := range services {
		service := &services[i]

		// if PortalIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to portal IP = %q", types.NamespacedName{service.Namespace, service.Name}, service.Spec.PortalIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			serviceName := ServicePortName{types.NamespacedName{service.Namespace, service.Name}, servicePort.Name}
			activeServices[serviceName] = true
			serviceIP := net.ParseIP(service.Spec.PortalIP)
			info, exists := proxier.getServiceInfo(serviceName)
			// TODO: check health of the socket?  What if ProxyLoop exited?
			if exists && sameConfig(info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				glog.V(4).Infof("Something changed for service %q: stopping it", serviceName)
				err := proxier.portalManager.ClosePortal(proxier.ProxierIPs, serviceName, info)
				if err != nil {
					glog.Errorf("Failed to close portal for %q: %v", serviceName, err)
				}
				err = proxier.stopProxy(serviceName, info)
				if err != nil {
					glog.Errorf("Failed to stop service %q: %v", serviceName, err)
				}
			}
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)
			info, err := proxier.addServiceOnPort(serviceName, servicePort.Protocol, 0, udpIdleTimeout)
			if err != nil {
				glog.Errorf("Failed to start proxy for %q: %v", serviceName, err)
				continue
			}
			info.portalIP = serviceIP
			info.portalPort = servicePort.Port
			info.publicIPs = service.Spec.PublicIPs
			info.sessionAffinityType = service.Spec.SessionAffinity
			glog.V(4).Infof("info: %+v", info)

			err = proxier.portalManager.OpenPortal(proxier.ProxierIPs, serviceName, info)
			if err != nil {
				glog.Errorf("Failed to open portal for %q: %v", serviceName, err)
			}
			proxier.loadBalancer.NewService(serviceName, info.sessionAffinityType, info.stickyMaxAgeMinutes)
		}
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Stopping service %q", name)
			err := proxier.portalManager.ClosePortal(proxier.ProxierIPs, name, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %v", name, err)
			}
			err = proxier.stopProxyInternal(name, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %v", name, err)
			}
		}
	}
}

func sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.portalPort != port.Port {
		return false
	}
	if !info.portalIP.Equal(net.ParseIP(service.Spec.PortalIP)) {
		return false
	}
	if !ipsEqual(info.publicIPs, service.Spec.PublicIPs) {
		return false
	}
	if info.sessionAffinityType != service.Spec.SessionAffinity {
		return false
	}
	return true
}

func ipsEqual(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if lhs[i] != rhs[i] {
			return false
		}
	}
	return true
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
