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
	"github.com/golang/glog"
)

type serviceInfo struct {
	port     int
	protocol api.Protocol
	socket   proxySocket
	timeout  time.Duration
	mu       sync.Mutex // protects active
	active   bool
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
	ProxyLoop(service string, proxier *Proxier)
}

// tcpProxySocket implements proxySocket.  Close() is implemented by net.Listener.  When Close() is called,
// no new connections are allowed but existing connections are left untouched.
type tcpProxySocket struct {
	net.Listener
}

func (tcp *tcpProxySocket) ProxyLoop(service string, proxier *Proxier) {
	info, found := proxier.getServiceInfo(service)
	if !found {
		glog.Errorf("Failed to find service: %s", service)
		return
	}
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
		glog.V(3).Infof("Mapped service %s to endpoint %s", service, endpoint)
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
	go copyBytes(in, out, &wg)
	go copyBytes(out, in, &wg)
	wg.Wait()
	in.Close()
	out.Close()
}

func copyBytes(in, out *net.TCPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	glog.V(4).Infof("Copying from %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	if _, err := io.Copy(in, out); err != nil {
		glog.Errorf("I/O error: %v", err)
	}
	in.CloseRead()
	out.CloseWrite()
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

func (udp *udpProxySocket) ProxyLoop(service string, proxier *Proxier) {
	info, found := proxier.getServiceInfo(service)
	if !found {
		glog.Errorf("Failed to find service: %s", service)
		return
	}
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
		glog.V(4).Infof("Mapped service %s to endpoint %s", service, endpoint)
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
	loadBalancer LoadBalancer
	mu           sync.Mutex // protects serviceMap
	serviceMap   map[string]*serviceInfo
	address      net.IP
}

// NewProxier returns a new Proxier given a LoadBalancer and an
// address on which to listen
func NewProxier(loadBalancer LoadBalancer, address net.IP) *Proxier {
	return &Proxier{
		loadBalancer: loadBalancer,
		serviceMap:   make(map[string]*serviceInfo),
		address:      address,
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
	glog.V(3).Infof("Removing service: %s", service)
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

// addServiceOnUnusedPort starts listening for a new service, returning the
// port it's using.  For testing on a system with unknown ports used.  The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnUnusedPort(service string, protocol api.Protocol, timeout time.Duration) (string, error) {
	sock, err := newProxySocket(protocol, proxier.address, 0)
	if err != nil {
		return "", err
	}
	_, port, err := net.SplitHostPort(sock.Addr().String())
	if err != nil {
		return "", err
	}
	portNum, err := strconv.Atoi(port)
	if err != nil {
		return "", err
	}
	proxier.setServiceInfo(service, &serviceInfo{
		port:     portNum,
		protocol: protocol,
		active:   true,
		socket:   sock,
		timeout:  timeout,
	})
	proxier.startAccepting(service, sock)
	return port, nil
}

func (proxier *Proxier) startAccepting(service string, sock proxySocket) {
	glog.V(1).Infof("Listening for %s on %s:%s", service, sock.Addr().Network(), sock.Addr().String())
	go func(service string, proxier *Proxier) {
		defer util.HandleCrash()
		sock.ProxyLoop(service, proxier)
	}(service, proxier)
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
		activeServices.Insert(service.ID)
		info, exists := proxier.getServiceInfo(service.ID)
		// TODO: check health of the socket?  What if ProxyLoop exited?
		if exists && info.isActive() && info.port == service.Port {
			continue
		}
		if exists && info.port != service.Port {
			err := proxier.stopProxy(service.ID, info)
			if err != nil {
				glog.Errorf("error stopping %s: %v", service.ID, err)
			}
		}
		glog.V(3).Infof("Adding a new service %s on %s port %d", service.ID, service.Protocol, service.Port)
		sock, err := newProxySocket(service.Protocol, proxier.address, service.Port)
		if err != nil {
			glog.Errorf("Failed to get a socket for %s: %+v", service.ID, err)
			continue
		}
		proxier.setServiceInfo(service.ID, &serviceInfo{
			port:     service.Port,
			protocol: service.Protocol,
			active:   true,
			socket:   sock,
			timeout:  udpIdleTimeout,
		})
		proxier.startAccepting(service.ID, sock)
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices.Has(name) {
			err := proxier.stopProxyInternal(name, info)
			if err != nil {
				glog.Errorf("error stopping %s: %v", name, err)
			}
		}
	}
}
