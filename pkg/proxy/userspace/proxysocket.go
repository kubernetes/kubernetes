/*
Copyright 2015 The Kubernetes Authors.

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
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/proxy"
)

// Abstraction over TCP/UDP sockets which are proxied.
type ProxySocket interface {
	// Addr gets the net.Addr for a ProxySocket.
	Addr() net.Addr
	// Close stops the ProxySocket from accepting incoming connections.
	// Each implementation should comment on the impact of calling Close
	// while sessions are active.
	Close() error
	// ProxyLoop proxies incoming connections for the specified service to the service endpoints.
	ProxyLoop(service proxy.ServicePortName, info *ServiceInfo, loadBalancer LoadBalancer)
	// ListenPort returns the host port that the ProxySocket is listening on
	ListenPort() int
}

func newProxySocket(protocol v1.Protocol, ip net.IP, port int) (ProxySocket, error) {
	host := ""
	if ip != nil {
		host = ip.String()
	}

	switch strings.ToUpper(string(protocol)) {
	case "TCP":
		listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		return &tcpProxySocket{Listener: listener, port: port}, nil
	case "UDP":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		return &udpProxySocket{UDPConn: conn, port: port}, nil
	case "SCTP":
		return nil, fmt.Errorf("SCTP is not supported for user space proxy")
	}
	return nil, fmt.Errorf("unknown protocol %q", protocol)
}

// How long we wait for a connection to a backend in seconds
var EndpointDialTimeouts = []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, 1 * time.Second, 2 * time.Second}

// tcpProxySocket implements ProxySocket.  Close() is implemented by net.Listener.  When Close() is called,
// no new connections are allowed but existing connections are left untouched.
type tcpProxySocket struct {
	net.Listener
	port int
}

func (tcp *tcpProxySocket) ListenPort() int {
	return tcp.port
}

// TryConnectEndpoints attempts to connect to the next available endpoint for the given service, cycling
// through until it is able to successfully connect, or it has tried with all timeouts in EndpointDialTimeouts.
func TryConnectEndpoints(service proxy.ServicePortName, srcAddr net.Addr, protocol string, loadBalancer LoadBalancer) (out net.Conn, err error) {
	sessionAffinityReset := false
	for _, dialTimeout := range EndpointDialTimeouts {
		endpoint, err := loadBalancer.NextEndpoint(service, srcAddr, sessionAffinityReset)
		if err != nil {
			klog.Errorf("Couldn't find an endpoint for %s: %v", service, err)
			return nil, err
		}
		klog.V(3).Infof("Mapped service %q to endpoint %s", service, endpoint)
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		outConn, err := net.DialTimeout(protocol, endpoint, dialTimeout)
		if err != nil {
			if isTooManyFDsError(err) {
				panic("Dial failed: " + err.Error())
			}
			klog.Errorf("Dial failed: %v", err)
			sessionAffinityReset = true
			continue
		}
		return outConn, nil
	}
	return nil, fmt.Errorf("failed to connect to an endpoint.")
}

func (tcp *tcpProxySocket) ProxyLoop(service proxy.ServicePortName, myInfo *ServiceInfo, loadBalancer LoadBalancer) {
	for {
		if !myInfo.IsAlive() {
			// The service port was closed or replaced.
			return
		}
		// Block until a connection is made.
		inConn, err := tcp.Accept()
		if err != nil {
			if isTooManyFDsError(err) {
				panic("Accept failed: " + err.Error())
			}

			if isClosedError(err) {
				return
			}
			if !myInfo.IsAlive() {
				// Then the service port was just closed so the accept failure is to be expected.
				return
			}
			klog.Errorf("Accept failed: %v", err)
			continue
		}
		klog.V(3).Infof("Accepted TCP connection from %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		outConn, err := TryConnectEndpoints(service, inConn.(*net.TCPConn).RemoteAddr(), "tcp", loadBalancer)
		if err != nil {
			klog.Errorf("Failed to connect to balancer: %v", err)
			inConn.Close()
			continue
		}
		// Spin up an async copy loop.
		go ProxyTCP(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// ProxyTCP proxies data bi-directionally between in and out.
func ProxyTCP(in, out *net.TCPConn) {
	var wg sync.WaitGroup
	wg.Add(2)
	klog.V(4).Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes("from backend", in, out, &wg)
	go copyBytes("to backend", out, in, &wg)
	wg.Wait()
}

func copyBytes(direction string, dest, src *net.TCPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	klog.V(4).Infof("Copying %s: %s -> %s", direction, src.RemoteAddr(), dest.RemoteAddr())
	n, err := io.Copy(dest, src)
	if err != nil {
		if !isClosedError(err) {
			klog.Errorf("I/O error: %v", err)
		}
	}
	klog.V(4).Infof("Copied %d bytes %s: %s -> %s", n, direction, src.RemoteAddr(), dest.RemoteAddr())
	dest.Close()
	src.Close()
}

// udpProxySocket implements ProxySocket.  Close() is implemented by net.UDPConn.  When Close() is called,
// no new connections are allowed and existing connections are broken.
// TODO: We could lame-duck this ourselves, if it becomes important.
type udpProxySocket struct {
	*net.UDPConn
	port int
}

func (udp *udpProxySocket) ListenPort() int {
	return udp.port
}

func (udp *udpProxySocket) Addr() net.Addr {
	return udp.LocalAddr()
}

// Holds all the known UDP clients that have not timed out.
type ClientCache struct {
	Mu      sync.Mutex
	Clients map[string]net.Conn // addr string -> connection
}

func newClientCache() *ClientCache {
	return &ClientCache{Clients: map[string]net.Conn{}}
}

func (udp *udpProxySocket) ProxyLoop(service proxy.ServicePortName, myInfo *ServiceInfo, loadBalancer LoadBalancer) {
	var buffer [4096]byte // 4KiB should be enough for most whole-packets
	for {
		if !myInfo.IsAlive() {
			// The service port was closed or replaced.
			break
		}

		// Block until data arrives.
		// TODO: Accumulate a histogram of n or something, to fine tune the buffer size.
		n, cliAddr, err := udp.ReadFrom(buffer[0:])
		if err != nil {
			if e, ok := err.(net.Error); ok {
				if e.Temporary() {
					klog.V(1).Infof("ReadFrom had a temporary failure: %v", err)
					continue
				}
			}
			klog.Errorf("ReadFrom failed, exiting ProxyLoop: %v", err)
			break
		}
		// If this is a client we know already, reuse the connection and goroutine.
		svrConn, err := udp.getBackendConn(myInfo.ActiveClients, cliAddr, loadBalancer, service, myInfo.Timeout)
		if err != nil {
			continue
		}
		// TODO: It would be nice to let the goroutine handle this write, but we don't
		// really want to copy the buffer.  We could do a pool of buffers or something.
		_, err = svrConn.Write(buffer[0:n])
		if err != nil {
			if !logTimeout(err) {
				klog.Errorf("Write failed: %v", err)
				// TODO: Maybe tear down the goroutine for this client/server pair?
			}
			continue
		}
		err = svrConn.SetDeadline(time.Now().Add(myInfo.Timeout))
		if err != nil {
			klog.Errorf("SetDeadline failed: %v", err)
			continue
		}
	}
}

func (udp *udpProxySocket) getBackendConn(activeClients *ClientCache, cliAddr net.Addr, loadBalancer LoadBalancer, service proxy.ServicePortName, timeout time.Duration) (net.Conn, error) {
	activeClients.Mu.Lock()
	defer activeClients.Mu.Unlock()

	svrConn, found := activeClients.Clients[cliAddr.String()]
	if !found {
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		klog.V(3).Infof("New UDP connection from %s", cliAddr)
		var err error
		svrConn, err = TryConnectEndpoints(service, cliAddr, "udp", loadBalancer)
		if err != nil {
			return nil, err
		}
		if err = svrConn.SetDeadline(time.Now().Add(timeout)); err != nil {
			klog.Errorf("SetDeadline failed: %v", err)
			return nil, err
		}
		activeClients.Clients[cliAddr.String()] = svrConn
		go func(cliAddr net.Addr, svrConn net.Conn, activeClients *ClientCache, timeout time.Duration) {
			defer runtime.HandleCrash()
			udp.proxyClient(cliAddr, svrConn, activeClients, timeout)
		}(cliAddr, svrConn, activeClients, timeout)
	}
	return svrConn, nil
}

// This function is expected to be called as a goroutine.
// TODO: Track and log bytes copied, like TCP
func (udp *udpProxySocket) proxyClient(cliAddr net.Addr, svrConn net.Conn, activeClients *ClientCache, timeout time.Duration) {
	defer svrConn.Close()
	var buffer [4096]byte
	for {
		n, err := svrConn.Read(buffer[0:])
		if err != nil {
			if !logTimeout(err) {
				klog.Errorf("Read failed: %v", err)
			}
			break
		}
		err = svrConn.SetDeadline(time.Now().Add(timeout))
		if err != nil {
			klog.Errorf("SetDeadline failed: %v", err)
			break
		}
		_, err = udp.WriteTo(buffer[0:n], cliAddr)
		if err != nil {
			if !logTimeout(err) {
				klog.Errorf("WriteTo failed: %v", err)
			}
			break
		}
	}
	activeClients.Mu.Lock()
	delete(activeClients.Clients, cliAddr.String())
	activeClients.Mu.Unlock()
}
