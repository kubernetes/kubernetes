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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

// Proxier is a simple proxy for tcp connections between a localhost:lport and services that provide
// the actual implementations.
type Proxier struct {
	loadBalancer LoadBalancer
	serviceMap   map[string]int
}

func NewProxier(loadBalancer LoadBalancer) *Proxier {
	return &Proxier{loadBalancer: loadBalancer, serviceMap: make(map[string]int)}
}

func CopyBytes(in, out *net.TCPConn) {
	glog.Infof("Copying from %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	_, err := io.Copy(in, out)
	if err != nil && err != io.EOF {
		glog.Errorf("I/O error: %v", err)
	}

	in.CloseRead()
	out.CloseWrite()
}

// Create a bidirectional byte shuffler. Copies bytes to/from each connection.
func ProxyConnection(in, out *net.TCPConn) {
	glog.Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go CopyBytes(in, out)
	go CopyBytes(out, in)
}

func (proxier Proxier) AcceptHandler(service string, listener net.Listener) {
	for {
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
		go ProxyConnection(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// AddService starts listening for a new service on a given port.
func (proxier Proxier) AddService(service string, port int) error {
	// Make sure we can start listening on the port before saying all's well.
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	proxier.addServiceCommon(service, l)
	return nil
}

// addService starts listening for a new service, returning the port it's using.
// For testing on a system with unknown ports used.
func (proxier Proxier) addServiceOnUnusedPort(service string) (string, error) {
	// Make sure we can start listening on the port before saying all's well.
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", err
	}
	proxier.addServiceCommon(service, l)
	_, port, err := net.SplitHostPort(l.Addr().String())
	return port, nil
}

func (proxier Proxier) addServiceCommon(service string, l net.Listener) {
	glog.Infof("Listening for %s on %s", service, l.Addr().String())
	// If that succeeds, start the accepting loop.
	go proxier.AcceptHandler(service, l)
}

func (proxier Proxier) OnUpdate(services []api.Service) {
	glog.Infof("Received update notice: %+v", services)
	for _, service := range services {
		port, exists := proxier.serviceMap[service.ID]
		if !exists || port != service.Port {
			glog.Infof("Adding a new service %s on port %d", service.ID, service.Port)
			err := proxier.AddService(service.ID, service.Port)
			if err == nil {
				proxier.serviceMap[service.ID] = service.Port
			} else {
				glog.Infof("Failed to start listening for %s on %d", service.ID, service.Port)
			}
		}
	}
}
