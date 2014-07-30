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
	port     int
	active   bool
	listener net.Listener
	lock     sync.Mutex
}

// Proxier is a simple proxy for tcp connections between a localhost:lport and services that provide
// the actual implementations.
type Proxier struct {
	loadBalancer LoadBalancer
	serviceMap   map[string]*serviceInfo
	// protects 'serviceMap'
	serviceLock sync.Mutex
}

// NewProxier returns a newly created and correctly initialized instance of Proxier.
func NewProxier(loadBalancer LoadBalancer) *Proxier {
	return &Proxier{loadBalancer: loadBalancer, serviceMap: make(map[string]*serviceInfo)}
}

func copyBytes(in, out *net.TCPConn) {
	glog.Infof("Copying from %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	_, err := io.Copy(in, out)
	if err != nil {
		glog.Errorf("I/O error: %v", err)
	}

	in.CloseRead()
	out.CloseWrite()
}

// proxyConnection creates a bidirectional byte shuffler.
// It copies bytes to/from each connection.
func proxyConnection(in, out *net.TCPConn) {
	glog.Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes(in, out)
	go copyBytes(out, in)
}

// StopProxy stops a proxy for the named service.  It stops the proxy loop and closes the socket.
func (proxier *Proxier) StopProxy(service string) error {
	// TODO: delete from map here?
	info, found := proxier.getServiceInfo(service)
	if !found {
		return fmt.Errorf("unknown service: %s", service)
	}
	info.lock.Lock()
	defer info.lock.Unlock()
	return proxier.stopProxyInternal(info)
}

// Requires that info.lock be held before calling.
func (proxier *Proxier) stopProxyInternal(info *serviceInfo) error {
	info.active = false
	return info.listener.Close()
}

func (proxier *Proxier) getServiceInfo(service string) (*serviceInfo, bool) {
	proxier.serviceLock.Lock()
	defer proxier.serviceLock.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service string, info *serviceInfo) {
	proxier.serviceLock.Lock()
	defer proxier.serviceLock.Unlock()
	proxier.serviceMap[service] = info
}

// AcceptHandler begins accepting incoming connections from listener and proxying the connections to the load-balanced endpoints.
// It never returns.
func (proxier *Proxier) AcceptHandler(service string, listener net.Listener) {
	info, found := proxier.getServiceInfo(service)
	if !found {
		glog.Errorf("Failed to find service: %s", service)
		return
	}
	for {
		info.lock.Lock()
		if !info.active {
			info.lock.Unlock()
			break
		}
		info.lock.Unlock()
		inConn, err := listener.Accept()
		if err != nil {
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.Infof("Accepted connection from: %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())

		// Figure out where this request should go.
		endpoint, err := proxier.loadBalancer.LoadBalance(service, inConn.RemoteAddr())
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s %v", service, err)
			inConn.Close()
			continue
		}

		glog.Infof("Mapped service %s to endpoint %s", service, endpoint)
		outConn, err := net.DialTimeout("tcp", endpoint, time.Duration(5)*time.Second)
		// We basically need to take everything from inConn and send to outConn
		// and anything coming from outConn needs to be sent to inConn.
		if err != nil {
			glog.Errorf("Dial failed: %v", err)
			inConn.Close()
			continue
		}
		proxyConnection(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// addService starts listening for a new service on a given port.
func (proxier *Proxier) addService(service string, port int) (net.Listener, error) {
	// Make sure we can start listening on the port before saying all's well.
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, err
	}
	proxier.addServiceCommon(service, l)
	return l, nil
}

// used to globally lock around unused ports.  Only used in testing.
var unusedPortLock sync.Mutex

// addService starts listening for a new service, returning the port it's using.
// For testing on a system with unknown ports used.
func (proxier *Proxier) addServiceOnUnusedPort(service string) (string, error) {
	unusedPortLock.Lock()
	defer unusedPortLock.Unlock()
	// Make sure we can start listening on the port before saying all's well.
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
	// If that succeeds, start the accepting loop.
	go proxier.AcceptHandler(service, l)
}

// OnUpdate receives update notices for the updated services and start listening newly added services.
// It implements "github.com/GoogleCloudPlatform/kubernetes/pkg/proxy/config".ServiceConfigHandler.OnUpdate.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.Infof("Received update notice: %+v", services)
	serviceNames := util.StringSet{}

	for _, service := range services {
		serviceNames.Insert(service.ID)
		info, exists := proxier.getServiceInfo(service.ID)
		if exists && info.port == service.Port {
			continue
		}
		if exists {
			// Stop the old proxier.
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

	proxier.serviceLock.Lock()
	defer proxier.serviceLock.Unlock()
	for name, info := range proxier.serviceMap {
		info.lock.Lock()
		if !serviceNames.Has(name) && info.active {
			glog.Infof("Removing service: %s", name)
			proxier.stopProxyInternal(info)
		}
		info.lock.Unlock()
	}
}
