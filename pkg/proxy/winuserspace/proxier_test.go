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
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/proxy"
	netshtest "k8s.io/kubernetes/pkg/util/netsh/testing"
)

const (
	udpIdleTimeoutForTest = 250 * time.Millisecond
)

func joinHostPort(host string, port int) string {
	return net.JoinHostPort(host, fmt.Sprintf("%d", port))
}

func waitForClosedPortTCP(p *Proxier, proxyPort int) error {
	for i := 0; i < 50; i++ {
		conn, err := net.Dial("tcp", joinHostPort("", proxyPort))
		if err != nil {
			return nil
		}
		conn.Close()
		time.Sleep(1 * time.Millisecond)
	}
	return fmt.Errorf("port %d still open", proxyPort)
}

func waitForClosedPortUDP(p *Proxier, proxyPort int) error {
	for i := 0; i < 50; i++ {
		conn, err := net.Dial("udp", joinHostPort("", proxyPort))
		if err != nil {
			return nil
		}
		conn.SetReadDeadline(time.Now().Add(10 * time.Millisecond))
		// To detect a closed UDP port write, then read.
		_, err = conn.Write([]byte("x"))
		if err != nil {
			if e, ok := err.(net.Error); ok && !e.Timeout() {
				return nil
			}
		}
		var buf [4]byte
		_, err = conn.Read(buf[0:])
		if err != nil {
			if e, ok := err.(net.Error); ok && !e.Timeout() {
				return nil
			}
		}
		conn.Close()
		time.Sleep(1 * time.Millisecond)
	}
	return fmt.Errorf("port %d still open", proxyPort)
}

// udpEchoServer is a simple echo server in UDP, intended for testing the proxy.
type udpEchoServer struct {
	net.PacketConn
}

func newUDPEchoServer() (*udpEchoServer, error) {
	packetconn, err := net.ListenPacket("udp", ":0")
	if err != nil {
		return nil, err
	}
	return &udpEchoServer{packetconn}, nil
}

func (r *udpEchoServer) Loop() {
	var buffer [4096]byte
	for {
		n, cliAddr, err := r.ReadFrom(buffer[0:])
		if err != nil {
			fmt.Printf("ReadFrom failed: %v\n", err)
			continue
		}
		r.WriteTo(buffer[0:n], cliAddr)
	}
}

var tcpServerPort int32
var udpServerPort int32

func TestMain(m *testing.M) {
	// Don't handle panics
	runtime.ReallyCrash = true

	// TCP setup.
	tcp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(r.URL.Path[1:]))
	}))
	defer tcp.Close()

	u, err := url.Parse(tcp.URL)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	_, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	tcpServerPortValue, err := strconv.Atoi(port)
	if err != nil {
		panic(fmt.Sprintf("failed to atoi(%s): %v", port, err))
	}
	tcpServerPort = int32(tcpServerPortValue)

	// UDP setup.
	udp, err := newUDPEchoServer()
	if err != nil {
		panic(fmt.Sprintf("failed to make a UDP server: %v", err))
	}
	_, port, err = net.SplitHostPort(udp.LocalAddr().String())
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	udpServerPortValue, err := strconv.Atoi(port)
	if err != nil {
		panic(fmt.Sprintf("failed to atoi(%s): %v", port, err))
	}
	udpServerPort = int32(udpServerPortValue)
	go udp.Loop()

	ret := m.Run()
	// it should be safe to call Close() multiple times.
	tcp.Close()
	os.Exit(ret)
}

func testEchoTCP(t *testing.T, address string, port int) {
	path := "aaaaa"
	res, err := http.Get("http://" + address + ":" + fmt.Sprintf("%d", port) + "/" + path)
	if err != nil {
		t.Fatalf("error connecting to server: %v", err)
	}
	defer res.Body.Close()
	data, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Errorf("error reading data: %v %v", err, string(data))
	}
	if string(data) != path {
		t.Errorf("expected: %s, got %s", path, string(data))
	}
}

func testEchoUDP(t *testing.T, address string, port int) {
	data := "abc123"

	conn, err := net.Dial("udp", joinHostPort(address, port))
	if err != nil {
		t.Fatalf("error connecting to server: %v", err)
	}
	if _, err := conn.Write([]byte(data)); err != nil {
		t.Fatalf("error sending to server: %v", err)
	}
	var resp [1024]byte
	n, err := conn.Read(resp[0:])
	if err != nil {
		t.Errorf("error receiving data: %v", err)
	}
	if string(resp[0:n]) != data {
		t.Errorf("expected: %s, got %s", data, string(resp[0:n]))
	}
}

func waitForNumProxyLoops(t *testing.T, p *Proxier, want int32) {
	var got int32
	for i := 0; i < 600; i++ {
		got = atomic.LoadInt32(&p.numProxyLoops)
		if got == want {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Errorf("expected %d ProxyLoops running, got %d", want, got)
}

func waitForNumProxyClients(t *testing.T, s *serviceInfo, want int, timeout time.Duration) {
	var got int
	now := time.Now()
	deadline := now.Add(timeout)
	for time.Now().Before(deadline) {
		s.activeClients.mu.Lock()
		got = len(s.activeClients.clients)
		s.activeClients.mu.Unlock()
		if got == want {
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Errorf("expected %d ProxyClients live, got %d", want, got)
}

func getPortNum(t *testing.T, addr string) int {
	_, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Errorf("error getting port from %s", addr)
		return 0
	}
	portNum, err := strconv.Atoi(portStr)
	if err != nil {
		t.Errorf("error getting port from %s", addr)
		return 0
	}

	return portNum
}

func TestTCPProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

func TestUDPProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoUDP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

func TestUDPProxyTimeout(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	waitForNumProxyLoops(t, p, 1)
	testEchoUDP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	// When connecting to a UDP service endpoint, there should be a Conn for proxy.
	waitForNumProxyClients(t, svcInfo, 1, time.Second)
	// If conn has no activity for serviceInfo.timeout since last Read/Write, it should be closed because of timeout.
	waitForNumProxyClients(t, svcInfo, 0, 2*time.Second)
}

func TestMultiPortProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	serviceP := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo-p"}, Port: "p"}
	serviceQ := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo-q"}, Port: "q"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Protocol: "TCP", Port: tcpServerPort}},
		}},
	})
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: serviceQ.Name, Namespace: serviceQ.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "q", Protocol: "UDP", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalNameP := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: serviceP.Namespace, Name: serviceP.Name}, Port: serviceP.Port, PortalIPName: listenIP}
	svcInfoP, err := p.addServicePortPortal(servicePortPortalNameP, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfoP.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)

	servicePortPortalNameQ := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: serviceQ.Namespace, Name: serviceQ.Name}, Port: serviceQ.Port, PortalIPName: listenIP}
	svcInfoQ, err := p.addServicePortPortal(servicePortPortalNameQ, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoUDP(t, "127.0.0.1", getPortNum(t, svcInfoQ.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 2)
}

func TestMultiPortOnServiceAdd(t *testing.T) {
	lb := NewLoadBalancerRR()
	serviceP := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	serviceQ := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "q"}
	serviceX := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "x"}

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: "0.0.0.0", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     0,
			Protocol: "TCP",
		}, {
			Name:     "q",
			Port:     0,
			Protocol: "UDP",
		}}},
	})
	waitForNumProxyLoops(t, p, 2)

	servicePortPortalNameP := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: serviceP.Namespace, Name: serviceP.Name}, Port: serviceP.Port, PortalIPName: listenIP}
	svcInfo, exists := p.getServiceInfo(servicePortPortalNameP)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalNameP)
	}
	if svcInfo.portal.ip != "0.0.0.0" || svcInfo.portal.port != 0 || svcInfo.protocol != "TCP" {
		t.Errorf("unexpected serviceInfo for %s: %#v", serviceP, svcInfo)
	}

	servicePortPortalNameQ := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: serviceQ.Namespace, Name: serviceQ.Name}, Port: serviceQ.Port, PortalIPName: listenIP}
	svcInfo, exists = p.getServiceInfo(servicePortPortalNameQ)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalNameQ)
	}
	if svcInfo.portal.ip != "0.0.0.0" || svcInfo.portal.port != 0 || svcInfo.protocol != "UDP" {
		t.Errorf("unexpected serviceInfo for %s: %#v", serviceQ, svcInfo)
	}

	servicePortPortalNameX := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: serviceX.Namespace, Name: serviceX.Name}, Port: serviceX.Port, PortalIPName: listenIP}
	svcInfo, exists = p.getServiceInfo(servicePortPortalNameX)
	if exists {
		t.Fatalf("found unwanted serviceInfo for %s: %#v", serviceX, svcInfo)
	}
}

// Helper: Stops the proxy for the named service.
func stopProxyByName(proxier *Proxier, service ServicePortPortalName) error {
	info, found := proxier.getServiceInfo(service)
	if !found {
		return fmt.Errorf("unknown service: %s", service)
	}
	return proxier.stopProxy(service, info)
}

func TestTCPProxyStop(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: service.Namespace, Name: service.Name},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	if !svcInfo.isAlive() {
		t.Fatalf("wrong value for isAlive(): expected true")
	}
	conn, err := net.Dial("tcp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	stopProxyByName(p, servicePortPortalName)
	if svcInfo.isAlive() {
		t.Fatalf("wrong value for isAlive(): expected false")
	}
	// Wait for the port to really close.
	if err := waitForClosedPortTCP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)
}

func TestUDPProxyStop(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: service.Namespace, Name: service.Name},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	stopProxyByName(p, servicePortPortalName)
	// Wait for the port to really close.
	if err := waitForClosedPortUDP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)
}

func TestTCPProxyUpdateDelete(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: service.Namespace, Name: service.Name},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	fmt.Println("here0")
	conn, err := net.Dial("tcp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceDelete(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "TCP",
		}}},
	})
	if err := waitForClosedPortTCP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)
}

func TestUDPProxyUpdateDelete(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: service.Namespace, Name: service.Name},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceDelete(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "UDP",
		}}},
	})
	if err := waitForClosedPortUDP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)
}

func TestTCPProxyUpdateDeleteUpdate(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	endpoint := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceDelete(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "TCP",
		}}},
	})
	if err := waitForClosedPortTCP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)

	// need to add endpoint here because it got clean up during service delete
	lb.OnEndpointsAdd(endpoint)
	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "TCP",
		}}},
	})
	svcInfo, exists := p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalName)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

func TestUDPProxyUpdateDeleteUpdate(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	endpoint := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", getPortNum(t, svcInfo.socket.Addr().String())))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceDelete(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "UDP",
		}}},
	})
	if err := waitForClosedPortUDP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	waitForNumProxyLoops(t, p, 0)

	// need to add endpoint here because it got clean up during service delete
	lb.OnEndpointsAdd(endpoint)
	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "UDP",
		}}},
	})
	svcInfo, exists := p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalName)
	}
	testEchoUDP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

func TestTCPProxyUpdatePort(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     0,
			Protocol: "TCP",
		}}},
	})
	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	svcInfo, exists := p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalName)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	// This is a bit async, but this should be sufficient.
	time.Sleep(500 * time.Millisecond)
	waitForNumProxyLoops(t, p, 1)
}

func TestUDPProxyUpdatePort(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: udpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "UDP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     0,
			Protocol: "UDP",
		}}},
	})
	// Wait for the socket to actually get free.
	if err := waitForClosedPortUDP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	svcInfo, exists := p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalName)
	}
	testEchoUDP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

func TestProxyUpdatePublicIPs(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	})

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)

	p.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Name:     "p",
				Port:     int32(svcInfo.portal.port),
				Protocol: "TCP",
			}},
			ClusterIP:   svcInfo.portal.ip,
			ExternalIPs: []string{"0.0.0.0"},
		},
	})
	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, getPortNum(t, svcInfo.socket.Addr().String())); err != nil {
		t.Fatalf(err.Error())
	}
	svcInfo, exists := p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortPortalName)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	// This is a bit async, but this should be sufficient.
	time.Sleep(500 * time.Millisecond)
	waitForNumProxyLoops(t, p, 1)
}

func TestProxyUpdatePortal(t *testing.T) {
	lb := NewLoadBalancerRR()
	service := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "testnamespace", Name: "echo"}, Port: "p"}
	endpoint := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: "p", Port: tcpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	listenIP := "0.0.0.0"
	p, err := createProxier(lb, net.ParseIP(listenIP), netshtest.NewFake(), net.ParseIP("127.0.0.1"), time.Minute, udpIdleTimeoutForTest)
	if err != nil {
		t.Fatal(err)
	}
	waitForNumProxyLoops(t, p, 0)

	servicePortPortalName := ServicePortPortalName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: service.Port, PortalIPName: listenIP}
	svcInfo, err := p.addServicePortPortal(servicePortPortalName, "TCP", listenIP, 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)

	svcv0 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(svcInfo.portal.port),
			Protocol: "TCP",
		}}},
	}

	svcv1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: "", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(svcInfo.portal.port),
			Protocol: "TCP",
		}}},
	}

	p.OnServiceUpdate(svcv0, svcv1)
	_, exists := p.getServiceInfo(servicePortPortalName)
	if exists {
		t.Fatalf("service with empty ClusterIP should not be included in the proxy")
	}

	svcv2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: "None", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(getPortNum(t, svcInfo.socket.Addr().String())),
			Protocol: "TCP",
		}}},
	}
	p.OnServiceUpdate(svcv1, svcv2)
	_, exists = p.getServiceInfo(servicePortPortalName)
	if exists {
		t.Fatalf("service with 'None' as ClusterIP should not be included in the proxy")
	}

	svcv3 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Spec: v1.ServiceSpec{ClusterIP: listenIP, Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     int32(svcInfo.portal.port),
			Protocol: "TCP",
		}}},
	}
	p.OnServiceUpdate(svcv2, svcv3)
	lb.OnEndpointsAdd(endpoint)
	svcInfo, exists = p.getServiceInfo(servicePortPortalName)
	if !exists {
		t.Fatalf("service with ClusterIP set not found in the proxy")
	}
	testEchoTCP(t, "127.0.0.1", getPortNum(t, svcInfo.socket.Addr().String()))
	waitForNumProxyLoops(t, p, 1)
}

// TODO(justinsb): Add test for nodePort conflict detection, once we have nodePort wired in
