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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type serviceInfo struct {
	name     string
	port     int
	listener net.Listener
	mu       sync.Mutex // protects active
	active   bool
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer LoadBalancer
	mu           sync.Mutex // protects serviceMap
	serviceMap   map[string]*serviceInfo
}

// NewProxier returns a new Proxier given a LoadBalancer.
func NewProxier(loadBalancer LoadBalancer) *Proxier {
	return &Proxier{
		loadBalancer: loadBalancer,
		serviceMap:   make(map[string]*serviceInfo),
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

// proxyConnection proxies data bidirectionally between in and out.
func proxyConnection(in, out *net.TCPConn) {
	glog.Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes(in, out)
	go copyBytes(out, in)
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
	return info.listener.Close()
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

// AcceptHandler proxies incoming connections for the specified service
// to the load-balanced service endpoints.
func (proxier *Proxier) AcceptHandler(service string, listener net.Listener) {
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
		inConn, err := listener.Accept()
		if err != nil {
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.Infof("Accepted connection from: %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		endpoint, err := proxier.loadBalancer.NextEndpoint(service, inConn.RemoteAddr())
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s %v", service, err)
			inConn.Close()
			continue
		}
		glog.Infof("Mapped service %s to endpoint %s", service, endpoint)
		outConn, err := net.DialTimeout("tcp", endpoint, time.Duration(5)*time.Second)
		if err != nil {
			glog.Errorf("Dial failed: %v", err)
			inConn.Close()
			continue
		}
		proxyConnection(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// addService creates and registers a service proxy for the given service on
// the specified port.
// It returns the net.Listener of the service proxy.
func (proxier *Proxier) addService(service string, port int) (net.Listener, error) {
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, err
	}
	proxier.addServiceCommon(service, l)
	return l, nil
}

// used to globally lock around unused ports. Only used in testing.
var unusedPortLock sync.Mutex

// addService starts listening for a new service, returning the port it's using.
// For testing on a system with unknown ports used.
func (proxier *Proxier) addServiceOnUnusedPort(service string) (string, error) {
	unusedPortLock.Lock()
	defer unusedPortLock.Unlock()
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", err
	}
	_, port, err := net.SplitHostPort(l.Addr().String())
	if err != nil {
		return "", err
	}
	portNum, err := strconv.Atoi(port)
	if err != nil {
		return "", err
	}
	proxier.setServiceInfo(service, &serviceInfo{
		port:     portNum,
		active:   true,
		listener: l,
	})
	proxier.addServiceCommon(service, l)
	return port, nil
}

func (proxier *Proxier) addServiceCommon(service string, l net.Listener) {
	glog.Infof("Listening for %s on %s", service, l.Addr().String())
	go proxier.AcceptHandler(service, l)
}

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier Proxier) OnUpdate(services []api.Service) {
	glog.Infof("Received update notice: %+v", services)
	activeServices := util.StringSet{}
	for _, service := range services {
		activeServices.Insert(service.ID)
		info, exists := proxier.getServiceInfo(service.ID)
		if exists && info.port == service.Port {
			continue
		}
		if exists {
			proxier.StopProxy(service.ID)
		}
		glog.Infof("Adding a new service %s on port %d", service.ID, service.Port)
		listener, err := proxier.addService(service.ID, service.Port)
		if err != nil {
			glog.Infof("Failed to start listening for %s on %d", service.ID, service.Port)
			continue
		}
		proxier.setServiceInfo(service.ID, &serviceInfo{
			port:     service.Port,
			active:   true,
			listener: listener,
		})
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices.Has(name) {
			proxier.stopProxyInternal(info)
		}
	}
}
