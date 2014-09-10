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
	name     string
	port     int
	protocol string
	socket   proxySocket
	mu       sync.Mutex // protects active
	active   bool
}

// Abstraction over TCP/UDP sockets which are proxied.
type proxySocket interface {
	// Addr gets the net.Addr for a proxySocket.
	Addr() net.Addr
	// Close stops the proxySocket from accepting incoming connections.
	Close() error
	// ProxyLoop proxies incoming connections for the specified service to the service endpoints.
	ProxyLoop(service string, proxier *Proxier)
}

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
		info.mu.Lock()
		if !info.active {
			info.mu.Unlock()
			break
		}
		info.mu.Unlock()

		// Block until a connection is made.
		inConn, err := tcp.Accept()
		if err != nil {
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.Infof("Accepted connection from %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		endpoint, err := proxier.loadBalancer.NextEndpoint(service, inConn.RemoteAddr())
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s %v", service, err)
			inConn.Close()
			continue
		}
		glog.Infof("Mapped service %s to endpoint %s", service, endpoint)
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		outConn, err := net.DialTimeout("tcp", endpoint, time.Duration(5)*time.Second)
		if err != nil {
			// TODO: Try another endpoint?
			glog.Errorf("Dial failed: %v", err)
			inConn.Close()
			continue
		}
		// Spin up an async copy loop.
		proxyTCP(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// proxyTCP proxies data bi-directionally between in and out.
func proxyTCP(in, out *net.TCPConn) {
	glog.Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes(in, out)
	go copyBytes(out, in)
}

func newProxySocket(protocol string, addr string, port int) (proxySocket, error) {
	switch strings.ToUpper(protocol) {
	case "TCP":
		listener, err := net.Listen("tcp", net.JoinHostPort(addr, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		return &tcpProxySocket{listener}, nil
		//TODO: add UDP support
	}
	return nil, fmt.Errorf("Unknown protocol %q", protocol)
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer LoadBalancer
	mu           sync.Mutex // protects serviceMap
	serviceMap   map[string]*serviceInfo
	address      string
}

// NewProxier returns a new Proxier given a LoadBalancer and an
// address on which to listen
func NewProxier(loadBalancer LoadBalancer, address string) *Proxier {
	return &Proxier{
		loadBalancer: loadBalancer,
		serviceMap:   make(map[string]*serviceInfo),
		address:      address,
	}
}

func copyBytes(in, out *net.TCPConn) {
	glog.Infof("Copying from %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	if _, err := io.Copy(in, out); err != nil {
		glog.Errorf("I/O error: %v", err)
	}
	in.CloseRead()
	out.CloseWrite()
}

// StopProxy stops the proxy for the named service.
func (proxier *Proxier) StopProxy(service string) error {
	// TODO: delete from map here?
	info, found := proxier.getServiceInfo(service)
	if !found {
		return fmt.Errorf("unknown service: %s", service)
	}
	return proxier.stopProxyInternal(info)
}

func (proxier *Proxier) stopProxyInternal(info *serviceInfo) error {
	info.mu.Lock()
	defer info.mu.Unlock()
	if !info.active {
		return nil
	}
	glog.Infof("Removing service: %s", info.name)
	info.active = false
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
	info.name = service
	proxier.serviceMap[service] = info
}

// used to globally lock around unused ports. Only used in testing.
var unusedPortLock sync.Mutex

// addServiceOnUnusedPort starts listening for a new service, returning the
// port it's using.  For testing on a system with unknown ports used.
func (proxier *Proxier) addServiceOnUnusedPort(service, protocol string) (string, error) {
	unusedPortLock.Lock()
	defer unusedPortLock.Unlock()
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
	})
	proxier.startAccepting(service, sock)
	return port, nil
}

func (proxier *Proxier) startAccepting(service string, sock proxySocket) {
	glog.Infof("Listening for %s on %s", service, sock.Addr().String())
	go sock.ProxyLoop(service, proxier)
}

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.Infof("Received update notice: %+v", services)
	activeServices := util.StringSet{}
	for _, service := range services {
		activeServices.Insert(service.ID)
		info, exists := proxier.getServiceInfo(service.ID)
		if exists && info.active && info.port == service.Port {
			continue
		}
		if exists && info.port != service.Port {
			proxier.StopProxy(service.ID)
		}
		glog.Infof("Adding a new service %s on %s port %d", service.ID, service.Protocol, service.Port)
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
		})
		proxier.startAccepting(service.ID, sock)
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices.Has(name) {
			proxier.stopProxyInternal(info)
		}
	}
}
