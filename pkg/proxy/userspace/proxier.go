/*
Copyright 2014 The Kubernetes Authors.

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

package userspace

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/types"
	utilnet "k8s.io/kubernetes/pkg/util/net"

	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/runtime"
)

type portal struct {
	ip         net.IP
	port       int
	isExternal bool
}

type serviceInfo struct {
	isAliveAtomic       int32 // Only access this with atomic ops
	portal              portal
	protocol            api.Protocol
	proxyPort           int
	socket              proxySocket
	timeout             time.Duration
	activeClients       *clientCache
	nodePort            int
	loadBalancerStatus  api.LoadBalancerStatus
	sessionAffinityType api.ServiceAffinity
	stickyMaxAgeMinutes int
	// Deprecated, but required for back-compat (including e2e)
	externalIPs []string
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
			glog.V(3).Infof("connection to endpoint closed due to inactivity")
			return true
		}
	}
	return false
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer   LoadBalancer
	mu             sync.Mutex // protects serviceMap
	serviceMap     map[proxy.ServicePortName]*serviceInfo
	syncPeriod     time.Duration
	minSyncPeriod  time.Duration // unused atm, but plumbed through
	udpIdleTimeout time.Duration
	portMapMutex   sync.Mutex
	portMap        map[portMapKey]*portMapValue
	numProxyLoops  int32 // use atomic ops to access this; mostly for testing
	listenIP       net.IP
	iptables       iptables.Interface
	hostIP         net.IP
	proxyPorts     PortAllocator
}

// assert Proxier is a ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// A key for the portMap.  The ip has to be a string because slices can't be map
// keys.
type portMapKey struct {
	ip       string
	port     int
	protocol api.Protocol
}

func (k *portMapKey) String() string {
	return fmt.Sprintf("%s:%d/%s", k.ip, k.port, k.protocol)
}

// A value for the portMap
type portMapValue struct {
	owner  proxy.ServicePortName
	socket interface {
		Close() error
	}
}

var (
	// ErrProxyOnLocalhost is returned by NewProxier if the user requests a proxier on
	// the loopback address. May be checked for by callers of NewProxier to know whether
	// the caller provided invalid input.
	ErrProxyOnLocalhost = fmt.Errorf("cannot proxy on localhost")
)

// IsProxyLocked returns true if the proxy could not acquire the lock on iptables.
func IsProxyLocked(err error) bool {
	return strings.Contains(err.Error(), "holding the xtables lock")
}

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen.  Because of the iptables logic, It is assumed that there
// is only a single Proxier active on a machine. An error will be returned if
// the proxier cannot be started due to an invalid ListenIP (loopback) or
// if iptables fails to update or acquire the initial lock. Once a proxier is
// created, it will keep iptables up to date in the background and will not
// terminate if a particular iptables call fails.
func NewProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface, pr utilnet.PortRange, syncPeriod, minSyncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	if listenIP.Equal(localhostIPv4) || listenIP.Equal(localhostIPv6) {
		return nil, ErrProxyOnLocalhost
	}

	hostIP, err := utilnet.ChooseHostInterface()
	if err != nil {
		return nil, fmt.Errorf("failed to select a host interface: %v", err)
	}

	err = setRLimit(64 * 1000)
	if err != nil {
		return nil, fmt.Errorf("failed to set open file handler limit: %v", err)
	}

	proxyPorts := newPortAllocator(pr)

	glog.V(2).Infof("Setting proxy IP to %v and initializing iptables", hostIP)
	return createProxier(loadBalancer, listenIP, iptables, hostIP, proxyPorts, syncPeriod, minSyncPeriod, udpIdleTimeout)
}

func createProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface, hostIP net.IP, proxyPorts PortAllocator, syncPeriod, minSyncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	// convenient to pass nil for tests..
	if proxyPorts == nil {
		proxyPorts = newPortAllocator(utilnet.PortRange{})
	}
	// Set up the iptables foundations we need.
	if err := iptablesInit(iptables); err != nil {
		return nil, fmt.Errorf("failed to initialize iptables: %v", err)
	}
	// Flush old iptables rules (since the bound ports will be invalid after a restart).
	// When OnUpdate() is first called, the rules will be recreated.
	if err := iptablesFlush(iptables); err != nil {
		return nil, fmt.Errorf("failed to flush iptables: %v", err)
	}
	return &Proxier{
		loadBalancer: loadBalancer,
		serviceMap:   make(map[proxy.ServicePortName]*serviceInfo),
		portMap:      make(map[portMapKey]*portMapValue),
		syncPeriod:   syncPeriod,
		// plumbed through if needed, not used atm.
		minSyncPeriod:  minSyncPeriod,
		udpIdleTimeout: udpIdleTimeout,
		listenIP:       listenIP,
		iptables:       iptables,
		hostIP:         hostIP,
		proxyPorts:     proxyPorts,
	}, nil
}

// CleanupLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ipt iptables.Interface) (encounteredError bool) {
	// NOTE: Warning, this needs to be kept in sync with the userspace Proxier,
	// we want to ensure we remove all of the iptables rules it creates.
	// Currently they are all in iptablesInit()
	// Delete Rules first, then Flush and Delete Chains
	args := []string{"-m", "comment", "--comment", "handle ClusterIPs; NOTE: this must be before the NodePort rules"}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostPortalChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerPortalChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	args = []string{"-m", "addrtype", "--dst-type", "LOCAL"}
	args = append(args, "-m", "comment", "--comment", "handle service NodePorts; NOTE: this must be the last rule in the chain")
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	args = []string{"-m", "comment", "--comment", "Ensure that non-local NodePort traffic can flow"}
	if err := ipt.DeleteRule(iptables.TableFilter, iptables.ChainInput, append(args, "-j", string(iptablesNonLocalNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}

	// flush and delete chains.
	tableChains := map[iptables.Table][]iptables.Chain{
		iptables.TableNAT:    {iptablesContainerPortalChain, iptablesHostPortalChain, iptablesHostNodePortChain, iptablesContainerNodePortChain},
		iptables.TableFilter: {iptablesNonLocalNodePortChain},
	}
	for table, chains := range tableChains {
		for _, c := range chains {
			// flush chain, then if successful delete, delete will fail if flush fails.
			if err := ipt.FlushChain(table, c); err != nil {
				if !iptables.IsNotFoundError(err) {
					glog.Errorf("Error flushing userspace chain: %v", err)
					encounteredError = true
				}
			} else {
				if err = ipt.DeleteChain(table, c); err != nil {
					if !iptables.IsNotFoundError(err) {
						glog.Errorf("Error deleting userspace chain: %v", err)
						encounteredError = true
					}
				}
			}
		}
	}
	return encounteredError
}

// Sync is called to immediately synchronize the proxier state to iptables
func (proxier *Proxier) Sync() {
	if err := iptablesInit(proxier.iptables); err != nil {
		glog.Errorf("Failed to ensure iptables: %v", err)
	}
	proxier.ensurePortals()
	proxier.cleanupStaleStickySessions()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		proxier.Sync()
	}
}

// Ensure that portals exist for all services.
func (proxier *Proxier) ensurePortals() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	// NB: This does not remove rules that should not be present.
	for name, info := range proxier.serviceMap {
		err := proxier.openPortal(name, info)
		if err != nil {
			glog.Errorf("Failed to ensure portal for %q: %v", name, err)
		}
	}
}

// clean up any stale sticky session records in the hash map.
func (proxier *Proxier) cleanupStaleStickySessions() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name := range proxier.serviceMap {
		proxier.loadBalancer.CleanupStaleStickySessions(name)
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service proxy.ServicePortName, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service proxy.ServicePortName, info *serviceInfo) error {
	delete(proxier.serviceMap, service)
	info.setAlive(false)
	err := info.socket.Close()
	port := info.socket.ListenPort()
	proxier.proxyPorts.Release(port)
	return err
}

func (proxier *Proxier) getServiceInfo(service proxy.ServicePortName) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service proxy.ServicePortName, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServiceOnPort starts listening for a new service, returning the serviceInfo.
// Pass proxyPort=0 to allocate a random port. The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnPort(service proxy.ServicePortName, protocol api.Protocol, proxyPort int, timeout time.Duration) (*serviceInfo, error) {
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
		isAliveAtomic:       1,
		proxyPort:           portNum,
		protocol:            protocol,
		socket:              sock,
		timeout:             timeout,
		activeClients:       newClientCache(),
		sessionAffinityType: api.ServiceAffinityNone, // default
		stickyMaxAgeMinutes: 180,                     // TODO: parameterize this in the API.
	}
	proxier.setServiceInfo(service, si)

	glog.V(2).Infof("Proxying for service %q on %s port %d", service, protocol, portNum)
	go func(service proxy.ServicePortName, proxier *Proxier) {
		defer runtime.HandleCrash()
		atomic.AddInt32(&proxier.numProxyLoops, 1)
		sock.ProxyLoop(service, si, proxier)
		atomic.AddInt32(&proxier.numProxyLoops, -1)
	}(service, proxier)

	return si, nil
}

// OnServiceUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnServiceUpdate(services []api.Service) {
	glog.V(4).Infof("Received update notice: %+v", services)
	activeServices := make(map[proxy.ServicePortName]bool) // use a map as a set
	for i := range services {
		service := &services[i]

		// if ClusterIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to clusterIP = %q", types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, service.Spec.ClusterIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]
			serviceName := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: servicePort.Name}
			activeServices[serviceName] = true
			serviceIP := net.ParseIP(service.Spec.ClusterIP)
			info, exists := proxier.getServiceInfo(serviceName)
			// TODO: check health of the socket?  What if ProxyLoop exited?
			if exists && sameConfig(info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				glog.V(4).Infof("Something changed for service %q: stopping it", serviceName)
				err := proxier.closePortal(serviceName, info)
				if err != nil {
					glog.Errorf("Failed to close portal for %q: %v", serviceName, err)
				}
				err = proxier.stopProxy(serviceName, info)
				if err != nil {
					glog.Errorf("Failed to stop service %q: %v", serviceName, err)
				}
			}

			proxyPort, err := proxier.proxyPorts.AllocateNext()
			if err != nil {
				glog.Errorf("failed to allocate proxy port for service %q: %v", serviceName, err)
				continue
			}

			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)
			info, err = proxier.addServiceOnPort(serviceName, servicePort.Protocol, proxyPort, proxier.udpIdleTimeout)
			if err != nil {
				glog.Errorf("Failed to start proxy for %q: %v", serviceName, err)
				continue
			}
			info.portal.ip = serviceIP
			info.portal.port = int(servicePort.Port)
			info.externalIPs = service.Spec.ExternalIPs
			// Deep-copy in case the service instance changes
			info.loadBalancerStatus = *api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
			info.nodePort = int(servicePort.NodePort)
			info.sessionAffinityType = service.Spec.SessionAffinity
			glog.V(4).Infof("info: %#v", info)

			err = proxier.openPortal(serviceName, info)
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
			err := proxier.closePortal(name, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %v", name, err)
			}
			err = proxier.stopProxyInternal(name, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %v", name, err)
			}
			proxier.loadBalancer.DeleteService(name)
		}
	}
}

func sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.portal.port != int(port.Port) || info.nodePort != int(port.NodePort) {
		return false
	}
	if !info.portal.ip.Equal(net.ParseIP(service.Spec.ClusterIP)) {
		return false
	}
	if !ipsEqual(info.externalIPs, service.Spec.ExternalIPs) {
		return false
	}
	if !api.LoadBalancerStatusEqual(&info.loadBalancerStatus, &service.Status.LoadBalancer) {
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

func (proxier *Proxier) openPortal(service proxy.ServicePortName, info *serviceInfo) error {
	err := proxier.openOnePortal(info.portal, info.protocol, proxier.listenIP, info.proxyPort, service)
	if err != nil {
		return err
	}
	for _, publicIP := range info.externalIPs {
		err = proxier.openOnePortal(portal{net.ParseIP(publicIP), info.portal.port, true}, info.protocol, proxier.listenIP, info.proxyPort, service)
		if err != nil {
			return err
		}
	}
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			err = proxier.openOnePortal(portal{net.ParseIP(ingress.IP), info.portal.port, false}, info.protocol, proxier.listenIP, info.proxyPort, service)
			if err != nil {
				return err
			}
		}
	}
	if info.nodePort != 0 {
		err = proxier.openNodePort(info.nodePort, info.protocol, proxier.listenIP, info.proxyPort, service)
		if err != nil {
			return err
		}
	}
	return nil
}

func (proxier *Proxier) openOnePortal(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	if local, err := isLocalIP(portal.ip); err != nil {
		return fmt.Errorf("can't determine if IP %s is local, assuming not: %v", portal.ip, err)
	} else if local {
		err := proxier.claimNodePort(portal.ip, portal.port, protocol, name)
		if err != nil {
			return err
		}
	}

	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portal.ip, portal.isExternal, false, portal.port, protocol, proxyIP, proxyPort, name)
	existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q, args:%v", iptablesContainerPortalChain, name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-containers portal for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}
	if portal.isExternal {
		args := proxier.iptablesContainerPortalArgs(portal.ip, false, true, portal.port, protocol, proxyIP, proxyPort, name)
		existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerPortalChain, args...)
		if err != nil {
			glog.Errorf("Failed to install iptables %s rule that opens service %q for local traffic, args:%v", iptablesContainerPortalChain, name, args)
			return err
		}
		if !existed {
			glog.V(3).Infof("Opened iptables from-containers portal for service %q on %s %s:%d for local traffic", name, protocol, portal.ip, portal.port)
		}

		args = proxier.iptablesHostPortalArgs(portal.ip, true, portal.port, protocol, proxyIP, proxyPort, name)
		existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostPortalChain, args...)
		if err != nil {
			glog.Errorf("Failed to install iptables %s rule for service %q for dst-local traffic", iptablesHostPortalChain, name)
			return err
		}
		if !existed {
			glog.V(3).Infof("Opened iptables from-host portal for service %q on %s %s:%d for dst-local traffic", name, protocol, portal.ip, portal.port)
		}
		return nil
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostPortalArgs(portal.ip, false, portal.port, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesHostPortalChain, name)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-host portal for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}
	return nil
}

// Marks a port as being owned by a particular service, or returns error if already claimed.
// Idempotent: reclaiming with the same owner is not an error
func (proxier *Proxier) claimNodePort(ip net.IP, port int, protocol api.Protocol, owner proxy.ServicePortName) error {
	proxier.portMapMutex.Lock()
	defer proxier.portMapMutex.Unlock()

	// TODO: We could pre-populate some reserved ports into portMap and/or blacklist some well-known ports

	key := portMapKey{ip: ip.String(), port: port, protocol: protocol}
	existing, found := proxier.portMap[key]
	if !found {
		// Hold the actual port open, even though we use iptables to redirect
		// it.  This ensures that a) it's safe to take and b) that stays true.
		// NOTE: We should not need to have a real listen()ing socket - bind()
		// should be enough, but I can't figure out a way to e2e test without
		// it.  Tools like 'ss' and 'netstat' do not show sockets that are
		// bind()ed but not listen()ed, and at least the default debian netcat
		// has no way to avoid about 10 seconds of retries.
		socket, err := newProxySocket(protocol, ip, port)
		if err != nil {
			return fmt.Errorf("can't open node port for %s: %v", key.String(), err)
		}
		proxier.portMap[key] = &portMapValue{owner: owner, socket: socket}
		glog.V(2).Infof("Claimed local port %s", key.String())
		return nil
	}
	if existing.owner == owner {
		// We are idempotent
		return nil
	}
	return fmt.Errorf("Port conflict detected on port %s.  %v vs %v", key.String(), owner, existing)
}

// Release a claim on a port.  Returns an error if the owner does not match the claim.
// Tolerates release on an unclaimed port, to simplify .
func (proxier *Proxier) releaseNodePort(ip net.IP, port int, protocol api.Protocol, owner proxy.ServicePortName) error {
	proxier.portMapMutex.Lock()
	defer proxier.portMapMutex.Unlock()

	key := portMapKey{ip: ip.String(), port: port, protocol: protocol}
	existing, found := proxier.portMap[key]
	if !found {
		// We tolerate this, it happens if we are cleaning up a failed allocation
		glog.Infof("Ignoring release on unowned port: %v", key)
		return nil
	}
	if existing.owner != owner {
		return fmt.Errorf("Port conflict detected on port %v (unowned unlock).  %v vs %v", key, owner, existing)
	}
	delete(proxier.portMap, key)
	existing.socket.Close()
	return nil
}

func (proxier *Proxier) openNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	// TODO: Do we want to allow containers to access public services?  Probably yes.
	// TODO: We could refactor this to be the same code as portal, but with IP == nil

	err := proxier.claimNodePort(nil, nodePort, protocol, name)
	if err != nil {
		return err
	}

	// Handle traffic from containers.
	args := proxier.iptablesContainerNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesContainerNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-containers public port for service %q on %s port %d", name, protocol, nodePort)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesHostNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-host public port for service %q on %s port %d", name, protocol, nodePort)
	}

	args = proxier.iptablesNonLocalNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableFilter, iptablesNonLocalNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesNonLocalNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-non-local public port for service %q on %s port %d", name, protocol, nodePort)
	}

	return nil
}

func (proxier *Proxier) closePortal(service proxy.ServicePortName, info *serviceInfo) error {
	// Collect errors and report them all at the end.
	el := proxier.closeOnePortal(info.portal, info.protocol, proxier.listenIP, info.proxyPort, service)
	for _, publicIP := range info.externalIPs {
		el = append(el, proxier.closeOnePortal(portal{net.ParseIP(publicIP), info.portal.port, true}, info.protocol, proxier.listenIP, info.proxyPort, service)...)
	}
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			el = append(el, proxier.closeOnePortal(portal{net.ParseIP(ingress.IP), info.portal.port, false}, info.protocol, proxier.listenIP, info.proxyPort, service)...)
		}
	}
	if info.nodePort != 0 {
		el = append(el, proxier.closeNodePort(info.nodePort, info.protocol, proxier.listenIP, info.proxyPort, service)...)
	}
	if len(el) == 0 {
		glog.V(3).Infof("Closed iptables portals for service %q", service)
	} else {
		glog.Errorf("Some errors closing iptables portals for service %q", service)
	}
	return utilerrors.NewAggregate(el)
}

func (proxier *Proxier) closeOnePortal(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	if local, err := isLocalIP(portal.ip); err != nil {
		el = append(el, fmt.Errorf("can't determine if IP %s is local, assuming not: %v", portal.ip, err))
	} else if local {
		if err := proxier.releaseNodePort(portal.ip, portal.port, protocol, name); err != nil {
			el = append(el, err)
		}
	}

	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portal.ip, portal.isExternal, false, portal.port, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerPortalChain, name)
		el = append(el, err)
	}

	if portal.isExternal {
		args := proxier.iptablesContainerPortalArgs(portal.ip, false, true, portal.port, protocol, proxyIP, proxyPort, name)
		if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerPortalChain, args...); err != nil {
			glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerPortalChain, name)
			el = append(el, err)
		}

		args = proxier.iptablesHostPortalArgs(portal.ip, true, portal.port, protocol, proxyIP, proxyPort, name)
		if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostPortalChain, args...); err != nil {
			glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostPortalChain, name)
			el = append(el, err)
		}
		return el
	}

	// Handle traffic from the host (portalIP is not external).
	args = proxier.iptablesHostPortalArgs(portal.ip, false, portal.port, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostPortalChain, name)
		el = append(el, err)
	}

	return el
}

func (proxier *Proxier) closeNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	// Handle traffic from containers.
	args := proxier.iptablesContainerNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerNodePortChain, name)
		el = append(el, err)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostNodePortChain, name)
		el = append(el, err)
	}

	// Handle traffic not local to the host
	args = proxier.iptablesNonLocalNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableFilter, iptablesNonLocalNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesNonLocalNodePortChain, name)
		el = append(el, err)
	}

	if err := proxier.releaseNodePort(nil, nodePort, protocol, name); err != nil {
		el = append(el, err)
	}

	return el
}

func isLocalIP(ip net.IP) (bool, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false, err
	}
	for i := range addrs {
		intf, _, err := net.ParseCIDR(addrs[i].String())
		if err != nil {
			return false, err
		}
		if ip.Equal(intf) {
			return true, nil
		}
	}
	return false, nil
}

// See comments in the *PortalArgs() functions for some details about why we
// use two chains for portals.
var iptablesContainerPortalChain iptables.Chain = "KUBE-PORTALS-CONTAINER"
var iptablesHostPortalChain iptables.Chain = "KUBE-PORTALS-HOST"

// Chains for NodePort services
var iptablesContainerNodePortChain iptables.Chain = "KUBE-NODEPORT-CONTAINER"
var iptablesHostNodePortChain iptables.Chain = "KUBE-NODEPORT-HOST"
var iptablesNonLocalNodePortChain iptables.Chain = "KUBE-NODEPORT-NON-LOCAL"

// Ensure that the iptables infrastructure we use is set up.  This can safely be called periodically.
func iptablesInit(ipt iptables.Interface) error {
	// TODO: There is almost certainly room for optimization here.  E.g. If
	// we knew the service-cluster-ip-range CIDR we could fast-track outbound packets not
	// destined for a service. There's probably more, help wanted.

	// Danger - order of these rules matters here:
	//
	// We match portal rules first, then NodePort rules.  For NodePort rules, we filter primarily on --dst-type LOCAL,
	// because we want to listen on all local addresses, but don't match internet traffic with the same dst port number.
	//
	// There is one complication (per thockin):
	// -m addrtype --dst-type LOCAL is what we want except that it is broken (by intent without foresight to our usecase)
	// on at least GCE. Specifically, GCE machines have a daemon which learns what external IPs are forwarded to that
	// machine, and configure a local route for that IP, making a match for --dst-type LOCAL when we don't want it to.
	// Removing the route gives correct behavior until the daemon recreates it.
	// Killing the daemon is an option, but means that any non-kubernetes use of the machine with external IP will be broken.
	//
	// This applies to IPs on GCE that are actually from a load-balancer; they will be categorized as LOCAL.
	// _If_ the chains were in the wrong order, and the LB traffic had dst-port == a NodePort on some other service,
	// the NodePort would take priority (incorrectly).
	// This is unlikely (and would only affect outgoing traffic from the cluster to the load balancer, which seems
	// doubly-unlikely), but we need to be careful to keep the rules in the right order.
	args := []string{ /* service-cluster-ip-range matching could go here */ }
	args = append(args, "-m", "comment", "--comment", "handle ClusterIPs; NOTE: this must be before the NodePort rules")
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerPortalChain))...); err != nil {
		return err
	}
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostPortalChain))...); err != nil {
		return err
	}

	// This set of rules matches broadly (addrtype & destination port), and therefore must come after the portal rules
	args = []string{"-m", "addrtype", "--dst-type", "LOCAL"}
	args = append(args, "-m", "comment", "--comment", "handle service NodePorts; NOTE: this must be the last rule in the chain")
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesContainerNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Append, iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerNodePortChain))...); err != nil {
		return err
	}
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesHostNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Append, iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostNodePortChain))...); err != nil {
		return err
	}

	// Create a chain intended to explicitly allow non-local NodePort
	// traffic to work around default-deny iptables configurations
	// that would otherwise reject such traffic.
	args = []string{"-m", "comment", "--comment", "Ensure that non-local NodePort traffic can flow"}
	if _, err := ipt.EnsureChain(iptables.TableFilter, iptablesNonLocalNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableFilter, iptables.ChainInput, append(args, "-j", string(iptablesNonLocalNodePortChain))...); err != nil {
		return err
	}

	// TODO: Verify order of rules.
	return nil
}

// Flush all of our custom iptables rules.
func iptablesFlush(ipt iptables.Interface) error {
	el := []error{}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesContainerNodePortChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesHostNodePortChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableFilter, iptablesNonLocalNodePortChain); err != nil {
		el = append(el, err)
	}
	if len(el) != 0 {
		glog.Errorf("Some errors flushing old iptables portals: %v", el)
	}
	return utilerrors.NewAggregate(el)
}

// Used below.
var zeroIPv4 = net.ParseIP("0.0.0.0")
var localhostIPv4 = net.ParseIP("127.0.0.1")

var zeroIPv6 = net.ParseIP("::0")
var localhostIPv6 = net.ParseIP("::1")

// Build a slice of iptables args that are common to from-container and from-host portal rules.
func iptablesCommonPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, service proxy.ServicePortName) []string {
	// This list needs to include all fields as they are eventually spit out
	// by iptables-save.  This is because some systems do not support the
	// 'iptables -C' arg, and so fall back on parsing iptables-save output.
	// If this does not match, it will not pass the check.  For example:
	// adding the /32 on the destination IP arg is not strictly required,
	// but causes this list to not match the final iptables-save output.
	// This is fragile and I hope one day we can stop supporting such old
	// iptables versions.
	args := []string{
		"-m", "comment",
		"--comment", service.String(),
		"-p", strings.ToLower(string(protocol)),
		"-m", strings.ToLower(string(protocol)),
		"--dport", fmt.Sprintf("%d", destPort),
	}

	if destIP != nil {
		args = append(args, "-d", fmt.Sprintf("%s/32", destIP.String()))
	}

	if addPhysicalInterfaceMatch {
		args = append(args, "-m", "physdev", "!", "--physdev-is-in")
	}

	if addDstLocalMatch {
		args = append(args, "-m", "addrtype", "--dst-type", "LOCAL")
	}

	return args
}

// Build a slice of iptables args for a from-container portal rule.
func (proxier *Proxier) iptablesContainerPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, addPhysicalInterfaceMatch, addDstLocalMatch, destPort, protocol, service)

	// This is tricky.
	//
	// If the proxy is bound (see Proxier.listenIP) to 0.0.0.0 ("any
	// interface") we want to use REDIRECT, which sends traffic to the
	// "primary address of the incoming interface" which means the container
	// bridge, if there is one.  When the response comes, it comes from that
	// same interface, so the NAT matches and the response packet is
	// correct.  This matters for UDP, since there is no per-connection port
	// number.
	//
	// The alternative would be to use DNAT, except that it doesn't work
	// (empirically):
	//   * DNAT to 127.0.0.1 = Packets just disappear - this seems to be a
	//     well-known limitation of iptables.
	//   * DNAT to eth0's IP = Response packets come from the bridge, which
	//     breaks the NAT, and makes things like DNS not accept them.  If
	//     this could be resolved, it would simplify all of this code.
	//
	// If the proxy is bound to a specific IP, then we have to use DNAT to
	// that IP.  Unlike the previous case, this works because the proxy is
	// ONLY listening on that IP, not the bridge.
	//
	// Why would anyone bind to an address that is not inclusive of
	// localhost?  Apparently some cloud environments have their public IP
	// exposed as a real network interface AND do not have firewalling.  We
	// don't want to expose everything out to the world.
	//
	// Unfortunately, I don't know of any way to listen on some (N > 1)
	// interfaces but not ALL interfaces, short of doing it manually, and
	// this is simpler than that.
	//
	// If the proxy is bound to localhost only, all of this is broken.  Not
	// allowed.
	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		// TODO: Can we REDIRECT with IPv6?
		args = append(args, "-j", "REDIRECT", "--to-ports", fmt.Sprintf("%d", proxyPort))
	} else {
		// TODO: Can we DNAT with IPv6?
		args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	}
	return args
}

// Build a slice of iptables args for a from-host portal rule.
func (proxier *Proxier) iptablesHostPortalArgs(destIP net.IP, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, false, addDstLocalMatch, destPort, protocol, service)

	// This is tricky.
	//
	// If the proxy is bound (see Proxier.listenIP) to 0.0.0.0 ("any
	// interface") we want to do the same as from-container traffic and use
	// REDIRECT.  Except that it doesn't work (empirically).  REDIRECT on
	// local packets sends the traffic to localhost (special case, but it is
	// documented) but the response comes from the eth0 IP (not sure why,
	// truthfully), which makes DNS unhappy.
	//
	// So we have to use DNAT.  DNAT to 127.0.0.1 can't work for the same
	// reason.
	//
	// So we do our best to find an interface that is not a loopback and
	// DNAT to that.  This works (again, empirically).
	//
	// If the proxy is bound to a specific IP, then we have to use DNAT to
	// that IP.  Unlike the previous case, this works because the proxy is
	// ONLY listening on that IP, not the bridge.
	//
	// If the proxy is bound to localhost only, this should work, but we
	// don't allow it for now.
	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		proxyIP = proxier.hostIP
	}
	// TODO: Can we DNAT with IPv6?
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

// Build a slice of iptables args for a from-container public-port rule.
// See iptablesContainerPortalArgs
// TODO: Should we just reuse iptablesContainerPortalArgs?
func (proxier *Proxier) iptablesContainerNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, nodePort, protocol, service)

	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		// TODO: Can we REDIRECT with IPv6?
		args = append(args, "-j", "REDIRECT", "--to-ports", fmt.Sprintf("%d", proxyPort))
	} else {
		// TODO: Can we DNAT with IPv6?
		args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	}

	return args
}

// Build a slice of iptables args for a from-host public-port rule.
// See iptablesHostPortalArgs
// TODO: Should we just reuse iptablesHostPortalArgs?
func (proxier *Proxier) iptablesHostNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, nodePort, protocol, service)

	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		proxyIP = proxier.hostIP
	}
	// TODO: Can we DNAT with IPv6?
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

// Build a slice of iptables args for an from-non-local public-port rule.
func (proxier *Proxier) iptablesNonLocalNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, proxyPort, protocol, service)
	args = append(args, "-m", "comment", "--comment", service.String(), "-m", "state", "--state", "NEW", "-j", "ACCEPT")
	return args
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
