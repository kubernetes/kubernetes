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
// Simple proxy for tcp connections between a localhost:lport and services that provide
// the actual implementations.

package proxy

import (
	"fmt"
	"io"
	"log"
	"net"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type Proxier struct {
	loadBalancer LoadBalancer
	serviceMap   map[string]int
}

func NewProxier(loadBalancer LoadBalancer) *Proxier {
	return &Proxier{loadBalancer: loadBalancer, serviceMap: make(map[string]int)}
}

func CopyBytes(in, out *net.TCPConn) {
	log.Printf("Copying from %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	_, err := io.Copy(in, out)
	if err != nil && err != io.EOF {
		log.Printf("I/O error: %v", err)
	}

	in.CloseRead()
	out.CloseWrite()
}

// Create a bidirectional byte shuffler. Copies bytes to/from each connection.
func ProxyConnection(in, out *net.TCPConn) {
	log.Printf("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go CopyBytes(in, out)
	go CopyBytes(out, in)
}

func (proxier Proxier) AcceptHandler(service string, listener net.Listener) {
	for {
		inConn, err := listener.Accept()
		if err != nil {
			log.Printf("Accept failed: %v", err)
			continue
		}
		log.Printf("Accepted connection from: %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())

		// Figure out where this request should go.
		endpoint, err := proxier.loadBalancer.LoadBalance(service, inConn.RemoteAddr())
		if err != nil {
			log.Printf("Couldn't find an endpoint for %s %v", service, err)
			inConn.Close()
			continue
		}

		log.Printf("Mapped service %s to endpoint %s", service, endpoint)
		outConn, err := net.DialTimeout("tcp", endpoint, time.Duration(5)*time.Second)
		// We basically need to take everything from inConn and send to outConn
		// and anything coming from outConn needs to be sent to inConn.
		if err != nil {
			log.Printf("Dial failed: %v", err)
			inConn.Close()
			continue
		}
		go ProxyConnection(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// AddService starts listening for a new service on a given port.
func (proxier Proxier) AddService(service string, port int) error {
	// Make sure we can start listening on the port before saying all's well.
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	log.Printf("Listening for %s on %d", service, port)
	// If that succeeds, start the accepting loop.
	go proxier.AcceptHandler(service, ln)
	return nil
}

func (proxier Proxier) OnUpdate(services []api.Service) {
	log.Printf("Received update notice: %+v", services)
	for _, service := range services {
		port, exists := proxier.serviceMap[service.ID]
		if !exists || port != service.Port {
			log.Printf("Adding a new service %s on port %d", service.ID, service.Port)
			err := proxier.AddService(service.ID, service.Port)
			if err == nil {
				proxier.serviceMap[service.ID] = service.Port
			} else {
				log.Printf("Failed to start listening for %s on %d", service.ID, service.Port)
			}
		}
	}
}
