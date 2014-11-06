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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
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
	mu         sync.Mutex // protects active
	active     bool
}

func (si *serviceInfo) isActive() bool {
	si.mu.Lock()
	defer si.mu.Unlock()
	return si.active
}

func (si *serviceInfo) setActive(val bool) bool {
	si.mu.Lock()
	defer si.mu.Unlock()
	tmp := si.active
	si.active = val
	return tmp
}

// How long we wait for a connection to a backend.
const endpointDialTimeout = 5 * time.Second

// Abstraction over TCP/UDP sockets which are proxied.
type proxySocket interface {
	// Addr gets the net.Addr for a proxySocket.
	Addr() net.Addr
	// Close stops the proxySocket from accepting incoming connections.  Each implementation should comment
	// on the impact of calling Close while sessions are active.
	Close() error
	// ProxyLoop proxies incoming connections for the specified service to the service endpoints.
	ProxyLoop(service string, info *serviceInfo, proxier *Proxier)
}

// tcpProxySocket implements proxySocket.  Close() is implemented by net.Listener.  When Close() is called,
// no new connections are allowed but existing connections are left untouched.
type tcpProxySocket struct {
	net.Listener
}

func (tcp *tcpProxySocket) ProxyLoop(service string, info *serviceInfo, proxier *Proxier) {
	for {
		if !info.isActive() {
			break
		}

		// Block until a connection is made.
		inConn, err := tcp.Accept()
		if err != nil {
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.V(2).Infof("Accepted TCP connection from %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		endpoint, err := proxier.loadBalancer.NextEndpoint(service, inConn.RemoteAddr())
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s %v", service, err)
			inConn.Close()
			continue
		}
		glog.V(3).Infof("Mapped service %q to endpoint %s", service, endpoint)
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		outConn, err := net.DialTimeout("tcp", endpoint, endpointDialTimeout)
		if err != nil {
			// TODO: Try another endpoint?
			glog.Errorf("Dial failed: %v", err)
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

func (udp *udpProxySocket) ProxyLoop(service string, info *serviceInfo, proxier *Proxier) {
	activeClients := newClientCache()
	var buffer [4096]byte // 4KiB should be enough for most whole-packets
	for {
		if !info.isActive() {
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
		svrConn, err := udp.getBackendConn(activeClients, cliAddr, proxier, service, info.timeout)
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
		svrConn.SetDeadline(time.Now().Add(info.timeout))
		if err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			continue
		}
	}
}

func (udp *udpProxySocket) getBackendConn(activeClients *clientCache, cliAddr net.Addr, proxier *Proxier, service string, timeout time.Duration) (net.Conn, error) {
	activeClients.mu.Lock()
	defer activeClients.mu.Unlock()

	svrConn, found := activeClients.clients[cliAddr.String()]
	if !found {
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		glog.V(2).Infof("New UDP connection from %s", cliAddr)
		endpoint, err := proxier.loadBalancer.NextEndpoint(service, cliAddr)
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s %v", service, err)
			return nil, err
		}
		glog.V(3).Infof("Mapped service %q to endpoint %s", service, endpoint)
		svrConn, err = net.DialTimeout("udp", endpoint, endpointDialTimeout)
		if err != nil {
			// TODO: Try another endpoint?
			glog.Errorf("Dial failed: %v", err)
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
		svrConn.SetDeadline(time.Now().Add(timeout))
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
			glog.V(1).Infof("connection to endpoint closed due to inactivity")
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
	return nil, fmt.Errorf("Unknown protocol %q", protocol)
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer  LoadBalancer
	mu            sync.Mutex // protects serviceMap
	serviceMap    map[string]*serviceInfo
	listenAddress net.IP
	iptables      iptables.Interface
}

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen.  Because of the iptables logic, It is assumed that there
// is only a single Proxier active on a machine.
func NewProxier(loadBalancer LoadBalancer, listenAddress net.IP, iptables iptables.Interface) *Proxier {
	glog.Infof("Initializing iptables")
	// Set up the iptables foundations we need.
	if err := iptablesInit(iptables); err != nil {
		glog.Errorf("Failed to initialize iptables: %s", err)
		return nil
	}
	// Flush old iptables rules (since the bound ports will be invalid after a restart).
	// When OnUpdate() is first called, the rules will be recreated.
	if err := iptablesFlush(iptables); err != nil {
		glog.Errorf("Failed to flush iptables: %s", err)
		return nil
	}
	return &Proxier{
		loadBalancer:  loadBalancer,
		serviceMap:    make(map[string]*serviceInfo),
		listenAddress: listenAddress,
		iptables:      iptables,
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
			if err := iptablesInit(proxier.iptables); err != nil {
				glog.Errorf("Failed to ensure iptables: %s", err)
			}
			proxier.ensurePortals()
		}
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
			glog.Errorf("Failed to ensure portal for %q: %s", name, err)
		}
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service string, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service string, info *serviceInfo) error {
	if !info.setActive(false) {
		return nil
	}
	delete(proxier.serviceMap, service)
	return info.socket.Close()
}

func (proxier *Proxier) getServiceInfo(service string) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service string, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServiceOnPort starts listening for a new service, returning the serviceInfo.
// Pass proxyPort=0 to allocate a random port. The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnPort(service string, protocol api.Protocol, proxyPort int, timeout time.Duration) (*serviceInfo, error) {
	sock, err := newProxySocket(protocol, proxier.listenAddress, proxyPort)
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
		proxyPort: portNum,
		protocol:  protocol,
		active:    true,
		socket:    sock,
		timeout:   timeout,
	}
	proxier.setServiceInfo(service, si)

	glog.V(1).Infof("Proxying for service %q on %s port %d", service, protocol, portNum)
	go func(service string, info *serviceInfo, proxier *Proxier) {
		defer util.HandleCrash()
		sock.ProxyLoop(service, info, proxier)
	}(service, si, proxier)

	return si, nil
}

// How long we leave idle UDP connections open.
const udpIdleTimeout = 1 * time.Minute

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.V(4).Infof("Received update notice: %+v", services)
	activeServices := util.StringSet{}
	for _, service := range services {
		activeServices.Insert(service.Name)
		info, exists := proxier.getServiceInfo(service.Name)
		serviceIP := net.ParseIP(service.Spec.PortalIP)
		// TODO: check health of the socket?  What if ProxyLoop exited?
		if exists && info.isActive() && info.portalPort == service.Spec.Port && info.portalIP.Equal(serviceIP) {
			continue
		}
		if exists && (info.portalPort != service.Spec.Port || !info.portalIP.Equal(serviceIP)) {
			glog.V(4).Infof("Something changed for service %q: stopping it", service.Name)
			err := proxier.closePortal(service.Name, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %s", service.Name, err)
			}
			err = proxier.stopProxy(service.Name, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %s", service.Name, err)
			}
		}
		glog.V(1).Infof("Adding new service %q at %s:%d/%s (local :%d)", service.Name, serviceIP, service.Spec.Port, service.Spec.Protocol, service.Spec.ProxyPort)
		info, err := proxier.addServiceOnPort(service.Name, service.Spec.Protocol, service.Spec.ProxyPort, udpIdleTimeout)
		if err != nil {
			glog.Errorf("Failed to start proxy for %q: %+v", service.Name, err)
			continue
		}
		info.portalIP = serviceIP
		info.portalPort = service.Spec.Port
		err = proxier.openPortal(service.Name, info)
		if err != nil {
			glog.Errorf("Failed to open portal for %q: %s", service.Name, err)
		}
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices.Has(name) {
			glog.V(1).Infof("Stopping service %q", name)
			err := proxier.closePortal(name, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %s", name, err)
			}
			err = proxier.stopProxyInternal(name, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %s", name, err)
			}
		}
	}
}

func (proxier *Proxier) openPortal(service string, info *serviceInfo) error {
	args := iptablesPortalArgs(info.portalIP, info.portalPort, info.protocol, proxier.listenAddress, info.proxyPort, service)
	existed, err := proxier.iptables.EnsureRule(iptables.TableNAT, iptablesProxyChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesProxyChain, service)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables portal for service %q on %s:%d", service, info.portalIP, info.portalPort)
	}
	return nil
}

func (proxier *Proxier) closePortal(service string, info *serviceInfo) error {
	args := iptablesPortalArgs(info.portalIP, info.portalPort, info.protocol, proxier.listenAddress, info.proxyPort, service)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesProxyChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesProxyChain, service)
		return err
	}
	glog.Infof("Closed iptables portal for service %q", service)
	return nil
}

var iptablesProxyChain iptables.Chain = "KUBE-PROXY"

// Ensure that the iptables infrastructure we use is set up.  This can safely be called periodically.
func iptablesInit(ipt iptables.Interface) error {
	// TODO: There is almost certainly room for optimization here.  E.g. If
	// we knew the portal_net CIDR we could fast-track outbound packets not
	// destined for a service. There's probably more, help wanted.
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesProxyChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.TableNAT, iptables.ChainPrerouting, "-j", string(iptablesProxyChain)); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.TableNAT, iptables.ChainOutput, "-j", string(iptablesProxyChain)); err != nil {
		return err
	}
	return nil
}

// Flush all of our custom iptables rules.
func iptablesFlush(ipt iptables.Interface) error {
	return ipt.FlushChain(iptables.TableNAT, iptablesProxyChain)
}

// Used below.
var zeroIP = net.ParseIP("0.0.0.0")
var localhostIP = net.ParseIP("127.0.0.1")

// Build a slice of iptables args for a portal rule.
func iptablesPortalArgs(destIP net.IP, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service string) []string {
	args := []string{
		"-m", "comment",
		"--comment", service,
		"-p", strings.ToLower(string(protocol)),
		"-d", destIP.String(),
		"--dport", fmt.Sprintf("%d", destPort),
	}
	// This is tricky.  If the proxy is bound (see Proxier.listenAddress)
	// to 0.0.0.0 ("any interface") or 127.0.0.1, we can use REDIRECT,
	// which will bring packets back to the host's loopback interface.  If
	// the proxy is bound to any other interface, then it is not listening
	// on the hosts's loopback, so we have to use DNAT to that specific
	// IP.  We can not simply use DNAT to 127.0.0.1 in the first case
	// because from within a container, 127.0.0.1 is the container's
	// loopback interface, not the host's.
	//
	// Why would anyone bind to an address that is not inclusive of
	// localhost?  Apparently some cloud environments have their public IP
	// exposed as a real network interface AND do not have firewalling.  We
	// don't want to expose everything out to the world.
	//
	// Unfortunately, I don't know of any way to listen on some (N > 1)
	// interfaces but not ALL interfaces, short of doing it manually, and
	// this is simpler than that.
	if proxyIP.Equal(zeroIP) || proxyIP.Equal(localhostIP) {
		args = append(args, "-j", "REDIRECT", "--to-ports", fmt.Sprintf("%d", proxyPort))
	} else {
		args = append(args, "-j", "DNAT", "--to-destination", fmt.Sprintf("%s:%d", proxyIP.String(), proxyPort))
	}
	return args
}
