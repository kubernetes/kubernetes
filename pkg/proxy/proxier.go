/*
Copyright 2014 Google Inc. All rights reserved.

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
	"io"
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
	portalIP   net.IP
	portalPort int
	protocol   api.Protocol
	proxyPort  int
	socket     proxySocket
	timeout    time.Duration
	// TODO: make this an net.IP address
	publicIP            []string
	sessionAffinityType api.AffinityType
	stickyMaxAgeMinutes int
}

// How long we wait for a connection to a backend in seconds
var endpointDialTimeout = []time.Duration{1, 2, 4, 8}

// Abstraction over TCP/UDP sockets which are proxied.
type proxySocket interface {
	// Addr gets the net.Addr for a proxySocket.
	Addr() net.Addr
	// Close stops the proxySocket from accepting incoming connections.
	// Each implementation should comment on the impact of calling Close
	// while sessions are active.
	Close() error
	// ProxyLoop proxies incoming connections for the specified service to the service endpoints.
	ProxyLoop(service types.NamespacedName, info *serviceInfo, proxier *Proxier)
}

// tcpProxySocket implements proxySocket.  Close() is implemented by net.Listener.  When Close() is called,
// no new connections are allowed but existing connections are left untouched.
type tcpProxySocket struct {
	net.Listener
}

func tryConnect(service types.NamespacedName, srcAddr net.Addr, protocol string, proxier *Proxier) (out net.Conn, err error) {
	for _, retryTimeout := range endpointDialTimeout {
		endpoint, err := proxier.loadBalancer.NextEndpoint(service, srcAddr)
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s: %v", service, err)
			return nil, err
		}
		glog.V(3).Infof("Mapped service %q to endpoint %s", service, endpoint)
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		outConn, err := net.DialTimeout(protocol, endpoint, retryTimeout*time.Second)
		if err != nil {
			glog.Errorf("Dial failed: %v", err)
			continue
		}
		return outConn, nil
	}
	return nil, fmt.Errorf("failed to connect to an endpoint.")
}

func (tcp *tcpProxySocket) ProxyLoop(service types.NamespacedName, myInfo *serviceInfo, proxier *Proxier) {
	for {
		if info, exists := proxier.getServiceInfo(service); !exists || info != myInfo {
			// The service port was closed or replaced.
			break
		}

		// Block until a connection is made.
		inConn, err := tcp.Accept()
		if err != nil {
			if info, exists := proxier.getServiceInfo(service); !exists || info != myInfo {
				// Then the service port was just closed so the accept failure is to be expected.
				break
			}
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.V(2).Infof("Accepted TCP connection from %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		outConn, err := tryConnect(service, inConn.(*net.TCPConn).RemoteAddr(), "tcp", proxier)
		if err != nil {
			glog.Errorf("Failed to connect to balancer: %v", err)
			inConn.Close()
			continue
		}
		// Spin up an async copy loop.
		go proxyTCP(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// proxyTCP proxies data bi-directionally between in and out.
func proxyTCP(in, out *net.TCPConn) {
	var wg sync.WaitGroup
	wg.Add(2)
	glog.V(4).Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes("from backend", in, out, &wg)
	go copyBytes("to backend", out, in, &wg)
	wg.Wait()
	in.Close()
	out.Close()
}

func copyBytes(direction string, dest, src *net.TCPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	glog.V(4).Infof("Copying %s: %s -> %s", direction, src.RemoteAddr(), dest.RemoteAddr())
	n, err := io.Copy(dest, src)
	if err != nil {
		glog.Errorf("I/O error: %v", err)
	}
	glog.V(4).Infof("Copied %d bytes %s: %s -> %s", n, direction, src.RemoteAddr(), dest.RemoteAddr())
	dest.CloseWrite()
	src.CloseRead()
}

// udpProxySocket implements proxySocket.  Close() is implemented by net.UDPConn.  When Close() is called,
// no new connections are allowed and existing connections are broken.
// TODO: We could lame-duck this ourselves, if it becomes important.
type udpProxySocket struct {
	*net.UDPConn
}

func (udp *udpProxySocket) Addr() net.Addr {
	return udp.LocalAddr()
}

// Holds all the known UDP clients that have not timed out.
type clientCache struct {
	mu      sync.Mutex
	clients map[string]net.Conn // addr string -> connection
}

func newClientCache() *clientCache {
	return &clientCache{clients: map[string]net.Conn{}}
}

func (udp *udpProxySocket) ProxyLoop(service types.NamespacedName, myInfo *serviceInfo, proxier *Proxier) {
	activeClients := newClientCache()
	var buffer [4096]byte // 4KiB should be enough for most whole-packets
	for {
		if info, exists := proxier.getServiceInfo(service); !exists || info != myInfo {
			// The service port was closed or replaced.
			break
		}

		// Block until data arrives.
		// TODO: Accumulate a histogram of n or something, to fine tune the buffer size.
		n, cliAddr, err := udp.ReadFrom(buffer[0:])
		if err != nil {
			if e, ok := err.(net.Error); ok {
				if e.Temporary() {
					glog.V(1).Infof("ReadFrom had a temporary failure: %v", err)
					continue
				}
			}
			glog.Errorf("ReadFrom failed, exiting ProxyLoop: %v", err)
			break
		}
		// If this is a client we know already, reuse the connection and goroutine.
		svrConn, err := udp.getBackendConn(activeClients, cliAddr, proxier, service, myInfo.timeout)
		if err != nil {
			continue
		}
		// TODO: It would be nice to let the goroutine handle this write, but we don't
		// really want to copy the buffer.  We could do a pool of buffers or something.
		_, err = svrConn.Write(buffer[0:n])
		if err != nil {
			if !logTimeout(err) {
				glog.Errorf("Write failed: %v", err)
				// TODO: Maybe tear down the goroutine for this client/server pair?
			}
			continue
		}
		err = svrConn.SetDeadline(time.Now().Add(myInfo.timeout))
		if err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			continue
		}
	}
}

func (udp *udpProxySocket) getBackendConn(activeClients *clientCache, cliAddr net.Addr, proxier *Proxier, service types.NamespacedName, timeout time.Duration) (net.Conn, error) {
	activeClients.mu.Lock()
	defer activeClients.mu.Unlock()

	svrConn, found := activeClients.clients[cliAddr.String()]
	if !found {
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		glog.V(2).Infof("New UDP connection from %s", cliAddr)
		var err error
		svrConn, err = tryConnect(service, cliAddr, "udp", proxier)
		if err != nil {
			return nil, err
		}
		if err = svrConn.SetDeadline(time.Now().Add(timeout)); err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			return nil, err
		}
		activeClients.clients[cliAddr.String()] = svrConn
		go func(cliAddr net.Addr, svrConn net.Conn, activeClients *clientCache, timeout time.Duration) {
			defer util.HandleCrash()
			udp.proxyClient(cliAddr, svrConn, activeClients, timeout)
		}(cliAddr, svrConn, activeClients, timeout)
	}
	return svrConn, nil
}

// This function is expected to be called as a goroutine.
// TODO: Track and log bytes copied, like TCP
func (udp *udpProxySocket) proxyClient(cliAddr net.Addr, svrConn net.Conn, activeClients *clientCache, timeout time.Duration) {
	defer svrConn.Close()
	var buffer [4096]byte
	for {
		n, err := svrConn.Read(buffer[0:])
		if err != nil {
			if !logTimeout(err) {
				glog.Errorf("Read failed: %v", err)
			}
			break
		}
		err = svrConn.SetDeadline(time.Now().Add(timeout))
		if err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			break
		}
		n, err = udp.WriteTo(buffer[0:n], cliAddr)
		if err != nil {
			if !logTimeout(err) {
				glog.Errorf("WriteTo failed: %v", err)
			}
			break
		}
	}
	activeClients.mu.Lock()
	delete(activeClients.clients, cliAddr.String())
	activeClients.mu.Unlock()
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

func newProxySocket(protocol api.Protocol, ip net.IP, port int) (proxySocket, error) {
	host := ip.String()
	switch strings.ToUpper(string(protocol)) {
	case "TCP":
		listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		return &tcpProxySocket{listener}, nil
	case "UDP":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		return &udpProxySocket{conn}, nil
	}
	return nil, fmt.Errorf("unknown protocol %q", protocol)
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer  LoadBalancer
	mu            sync.Mutex // protects serviceMap
	serviceMap    map[types.NamespacedName]*serviceInfo
	numProxyLoops int32 // use atomic ops to access this; mostly for testing
	listenIP      net.IP
	portalManager PortalManager
	hostIP        net.IP
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
	return CreateProxier(loadBalancer, listenIP, iptables, hostIP)
}

func CreateProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface, hostIP net.IP) *Proxier {
	portalManager := NewIptablesPortalManager(iptables)
	glog.Infof("Initializing iptables")
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
		serviceMap:    make(map[types.NamespacedName]*serviceInfo),
		listenIP:      listenIP,
		portalManager: portalManager,
		hostIP:        hostIP,
	}
}

// The periodic interval for checking the state of things.
const syncInterval = 5 * time.Second

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	for {
		select {
		case <-time.After(syncInterval):
			glog.V(2).Infof("Periodic sync")
			if err := proxier.portalManager.Init(); err != nil {
				glog.Errorf("Failed to ensure iptables: %v", err)
			}
			proxier.ensurePortals()
			proxier.cleanupStaleStickySessions()
		}
	}
}

// Ensure that portals exist for all services.
func (proxier *Proxier) ensurePortals() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	// NB: This does not remove rules that should not be present.
	for name, info := range proxier.serviceMap {
		err := proxier.portalManager.OpenPortal(proxier, name, info)
		if err != nil {
			glog.Errorf("Failed to ensure portal for %q: %v", name, err)
		}
	}
}

// clean up any stale sticky session records in the hash map.
func (proxier *Proxier) cleanupStaleStickySessions() {
	for name, info := range proxier.serviceMap {
		if info.sessionAffinityType != api.AffinityTypeNone {
			proxier.loadBalancer.CleanupStaleStickySessions(name)
		}
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service types.NamespacedName, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service types.NamespacedName, info *serviceInfo) error {
	delete(proxier.serviceMap, service)
	return info.socket.Close()
}

func (proxier *Proxier) getServiceInfo(service types.NamespacedName) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service types.NamespacedName, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServiceOnPort starts listening for a new service, returning the serviceInfo.
// Pass proxyPort=0 to allocate a random port. The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnPort(service types.NamespacedName, protocol api.Protocol, proxyPort int, timeout time.Duration) (*serviceInfo, error) {
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
		sessionAffinityType: api.AffinityTypeNone,
		stickyMaxAgeMinutes: 180,
	}
	proxier.setServiceInfo(service, si)

	glog.V(1).Infof("Proxying for service %q on %s port %d", service, protocol, portNum)
	go func(service types.NamespacedName, proxier *Proxier) {
		defer util.HandleCrash()
		atomic.AddInt32(&proxier.numProxyLoops, 1)
		sock.ProxyLoop(service, si, proxier)
		atomic.AddInt32(&proxier.numProxyLoops, -1)
	}(service, proxier)

	return si, nil
}

// How long we leave idle UDP connections open.
const udpIdleTimeout = 1 * time.Minute

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.V(4).Infof("Received update notice: %+v", services)
	activeServices := make(map[types.NamespacedName]bool) // use a map as a set
	for _, service := range services {
		// if PortalIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(&service) {
			continue
		}
		serviceName := types.NamespacedName{service.Namespace, service.Name}
		activeServices[serviceName] = true
		info, exists := proxier.getServiceInfo(serviceName)
		serviceIP := net.ParseIP(service.Spec.PortalIP)
		// TODO: check health of the socket?  What if ProxyLoop exited?
		if exists && info.portalPort == service.Spec.Port && info.portalIP.Equal(serviceIP) {
			continue
		}
		if exists && (info.portalPort != service.Spec.Port || !info.portalIP.Equal(serviceIP) || !ipsEqual(service.Spec.PublicIPs, info.publicIP)) {
			glog.V(4).Infof("Something changed for service %q: stopping it", serviceName.String())
			err := proxier.portalManager.ClosePortal(proxier, serviceName, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %v", serviceName, err)
			}
			err = proxier.stopProxy(serviceName, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %v", serviceName, err)
			}
		}
		glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, service.Spec.Port, service.Spec.Protocol)
		info, err := proxier.addServiceOnPort(serviceName, service.Spec.Protocol, 0, udpIdleTimeout)
		if err != nil {
			glog.Errorf("Failed to start proxy for %q: %v", serviceName, err)
			continue
		}
		info.portalIP = serviceIP
		info.portalPort = service.Spec.Port
		info.publicIP = service.Spec.PublicIPs
		info.sessionAffinityType = service.Spec.SessionAffinity
		// TODO: paramaterize this in the types api file as an attribute of sticky session.   For now it's hardcoded to 3 hours.
		info.stickyMaxAgeMinutes = 180
		glog.V(4).Infof("info: %+v", info)

		err = proxier.portalManager.OpenPortal(proxier, serviceName, info)
		if err != nil {
			glog.Errorf("Failed to open portal for %q: %v", serviceName, err)
		}
		proxier.loadBalancer.NewService(serviceName, info.sessionAffinityType, info.stickyMaxAgeMinutes)
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Stopping service %q", name)
			err := proxier.portalManager.ClosePortal(proxier, name, info)
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
