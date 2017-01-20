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

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/util/netsh"
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
	udpIdleTimeout time.Duration
	portMapMutex   sync.Mutex
	portMap        map[portMapKey]*portMapValue
	numProxyLoops  int32 // use atomic ops to access this; mostly for testing
	listenIP       net.IP
	netsh          netsh.Interface
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

// Used below.
var localhostIPv4 = net.ParseIP("127.0.0.1")
var localhostIPv6 = net.ParseIP("::1")

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen.  Because of the iptables logic, It is assumed that there
// is only a single Proxier active on a machine. An error will be returned if
// the proxier cannot be started due to an invalid ListenIP (loopback) or
// if iptables fails to update or acquire the initial lock. Once a proxier is
// created, it will keep iptables up to date in the background and will not
// terminate if a particular iptables call fails.
func NewProxier(loadBalancer LoadBalancer, listenIP net.IP, netsh netsh.Interface, pr utilnet.PortRange, syncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	if listenIP.Equal(localhostIPv4) || listenIP.Equal(localhostIPv6) {
		return nil, ErrProxyOnLocalhost
	}

	hostIP, err := utilnet.ChooseHostInterface()
	if err != nil {
		return nil, fmt.Errorf("failed to select a host interface: %v", err)
	}

	proxyPorts := newPortAllocator(pr)

	glog.V(2).Infof("Setting proxy IP to %v and initializing iptables", hostIP)
	return createProxier(loadBalancer, listenIP, netsh, hostIP, proxyPorts, syncPeriod, udpIdleTimeout)
}

func createProxier(loadBalancer LoadBalancer, listenIP net.IP, netsh netsh.Interface, hostIP net.IP, proxyPorts PortAllocator, syncPeriod, udpIdleTimeout time.Duration) (*Proxier, error) {
	// convenient to pass nil for tests..
	if proxyPorts == nil {
		proxyPorts = newPortAllocator(utilnet.PortRange{})
	}
	return &Proxier{
		loadBalancer:   loadBalancer,
		serviceMap:     make(map[proxy.ServicePortName]*serviceInfo),
		portMap:        make(map[portMapKey]*portMapValue),
		syncPeriod:     syncPeriod,
		udpIdleTimeout: udpIdleTimeout,
		listenIP:       listenIP,
		netsh:          netsh,
		hostIP:         hostIP,
		proxyPorts:     proxyPorts,
	}, nil
}

// Sync is called to immediately synchronize the proxier state to iptables
func (proxier *Proxier) Sync() {
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

// cleanupStaleStickySessions cleans up any stale sticky session records in the hash map.
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
	if protocol == api.ProtocolUDP {
		glog.Warningf("Not adding rule for %q on %s:%d as UDP protocol is not supported by netsh portproxy", name, portal.ip, portal.port)
		return nil
	}

	// Add IP address to "vEthernet (HNSTransparent)" so that portproxy could be used to redirect the traffic
	args := proxier.netshIpv4AddressAddArgs(portal.ip)
	existed, err := proxier.netsh.EnsureIPAddress(args, portal.ip)

	if err != nil {
		glog.Errorf("Failed to add ip address for service %q, args:%v", name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Added ip address to HNSTransparent interface for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}

	args = proxier.netshPortProxyAddArgs(portal.ip, portal.port, proxyIP, proxyPort, name)
	existed, err = proxier.netsh.EnsurePortProxyRule(args)

	if err != nil {
		glog.Errorf("Failed to run portproxy rule for service %q, args:%v", name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Added portproxy rule for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}

	return nil
}

// claimNodePort marks a port as being owned by a particular service, or returns error if already claimed.
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

// releaseNodePort releases a claim on a port.  Returns an error if the owner does not match the claim.
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
	if protocol == api.ProtocolUDP {
		glog.Warningf("Not adding node port rule for %q on port %d as UDP protocol is not supported by netsh portproxy", name, nodePort)
		return nil
	}

	err := proxier.claimNodePort(nil, nodePort, protocol, name)
	if err != nil {
		return err
	}

	args := proxier.netshPortProxyAddArgs(nil, nodePort, proxyIP, proxyPort, name)
	existed, err := proxier.netsh.EnsurePortProxyRule(args)

	if err != nil {
		glog.Errorf("Failed to run portproxy rule for service %q", name)
		return err
	}
	if !existed {
		glog.Infof("Added portproxy rule for service %q on %s port %d", name, protocol, nodePort)
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
		el = append(el, fmt.Errorf("can't determine if IP is local, assuming not: %v", err))
	} else if local {
		if err := proxier.releaseNodePort(portal.ip, portal.port, protocol, name); err != nil {
			el = append(el, err)
		}
	}

	args := proxier.netshIpv4AddressDeleteArgs(portal.ip)
	if err := proxier.netsh.DeleteIPAddress(args); err != nil {
		glog.Errorf("Failed to delete IP address for service %q", name)
		el = append(el, err)
	}

	args = proxier.netshPortProxyDeleteArgs(portal.ip, portal.port, proxyIP, proxyPort, name)
	if err := proxier.netsh.DeletePortProxyRule(args); err != nil {
		glog.Errorf("Failed to delete portproxy rule for service %q", name)
		el = append(el, err)
	}

	return el
}

func (proxier *Proxier) closeNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	args := proxier.netshPortProxyDeleteArgs(nil, nodePort, proxyIP, proxyPort, name)
	if err := proxier.netsh.DeletePortProxyRule(args); err != nil {
		glog.Errorf("Failed to delete portproxy rule for service %q", name)
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

func isTooManyFDsError(err error) bool {
	return strings.Contains(err.Error(), "too many open files")
}

func isClosedError(err error) bool {
	// A brief discussion about handling closed error here:
	// https://code.google.com/p/go/issues/detail?id=4373#c14
	// TODO: maybe create a stoppable TCP listener that returns a StoppedError
	return strings.HasSuffix(err.Error(), "use of closed network connection")
}

func (proxier *Proxier) netshPortProxyAddArgs(destIP net.IP, destPort int, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := []string{
		"interface", "portproxy", "set", "v4tov4",
		"listenPort=" + strconv.Itoa(destPort),
		"connectaddress=" + proxyIP.String(),
		"connectPort=" + strconv.Itoa(proxyPort),
	}
	if destIP != nil {
		args = append(args, "listenaddress="+destIP.String())
	}

	return args
}

func (proxier *Proxier) netshIpv4AddressAddArgs(destIP net.IP) []string {
	intName := proxier.netsh.GetInterfaceToAddIP()
	args := []string{
		"interface", "ipv4", "add", "address",
		"name=" + intName,
		"address=" + destIP.String(),
	}

	return args
}

func (proxier *Proxier) netshPortProxyDeleteArgs(destIP net.IP, destPort int, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := []string{
		"interface", "portproxy", "delete", "v4tov4",
		"listenPort=" + strconv.Itoa(destPort),
	}
	if destIP != nil {
		args = append(args, "listenaddress="+destIP.String())
	}

	return args
}

func (proxier *Proxier) netshIpv4AddressDeleteArgs(destIP net.IP) []string {
	intName := proxier.netsh.GetInterfaceToAddIP()
	args := []string{
		"interface", "ipv4", "delete", "address",
		"name=" + intName,
		"address=" + destIP.String(),
	}

	return args
}
