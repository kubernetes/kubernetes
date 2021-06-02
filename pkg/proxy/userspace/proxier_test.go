/*
Copyright 2014 The Kubernetes Authors.

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
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/proxy"
	ipttest "k8s.io/kubernetes/pkg/util/iptables/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
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

func waitForProxyFinished(t *testing.T, svcInfo *ServiceInfo) {
	if err := wait.PollImmediate(50*time.Millisecond, 30*time.Second, func() (bool, error) {
		return svcInfo.IsFinished(), nil
	}); err != nil {
		t.Errorf("timed out waiting for proxy socket to finish: %v", err)
	}
}

func waitForServiceInfo(t *testing.T, p *Proxier, servicePortName proxy.ServicePortName, service *v1.Service) *ServiceInfo {
	var svcInfo *ServiceInfo
	var exists bool
	wait.PollImmediate(50*time.Millisecond, 3*time.Second, func() (bool, error) {
		svcInfo, exists = p.getServiceInfo(servicePortName)
		return exists, nil
	})
	if !exists {
		t.Fatalf("can't find serviceInfo for %s", servicePortName)
	}
	if !svcInfo.IsAlive() {
		t.Fatalf("expected IsAlive() true for %s", servicePortName)
	}

	var servicePort *v1.ServicePort
	for _, port := range service.Spec.Ports {
		if port.Name == servicePortName.Port {
			servicePort = &port
			break
		}
	}
	if servicePort == nil {
		t.Errorf("failed to find service %s port with name %q", servicePortName.NamespacedName, servicePortName.Port)
	}
	if svcInfo.portal.ip.String() != service.Spec.ClusterIP || int32(svcInfo.portal.port) != servicePort.Port || svcInfo.protocol != servicePort.Protocol {
		t.Errorf("unexpected serviceInfo for %s: %#v", servicePortName, svcInfo)
	}

	// Wait for proxy socket to start up
	if err := wait.PollImmediate(50*time.Millisecond, 30*time.Second, func() (bool, error) {
		return svcInfo.IsStarted(), nil
	}); err != nil {
		t.Errorf("timed out waiting for proxy socket %s to start: %v", servicePortName, err)
	}

	return svcInfo
}

// addServiceAndWaitForInfoIndex adds the service to the proxy and waits for the
// named port to be ready
func addServiceAndWaitForInfo(t *testing.T, p *Proxier, servicePortName proxy.ServicePortName, service *v1.Service) *ServiceInfo {
	p.OnServiceAdd(service)
	return waitForServiceInfo(t, p, servicePortName, service)
}

// deleteServiceAndWait deletes the servicein the proxy and waits until it
// has been cleaned up. waitFunc will be called to wait for the service
// port's socket to close.
func deleteServiceAndWait(t *testing.T, p *Proxier, svcInfo *ServiceInfo, service *v1.Service, waitFunc func(*Proxier, int) error) {
	p.OnServiceDelete(service)
	// Wait for the port to really close.
	if err := waitFunc(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	waitForProxyFinished(t, svcInfo)
	if svcInfo.IsAlive() {
		t.Fatalf("wrong value for IsAlive(): expected false")
	}
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

func waitForNumProxyClients(t *testing.T, s *ServiceInfo, want int, timeout time.Duration) {
	var got int
	now := time.Now()
	deadline := now.Add(timeout)
	for time.Now().Before(deadline) {
		s.ActiveClients.Mu.Lock()
		got = len(s.ActiveClients.Clients)
		s.ActiveClients.Mu.Unlock()
		if got == want {
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Errorf("expected %d ProxyClients live, got %d", want, got)
}

func startProxier(p *Proxier, t *testing.T) {
	go func() {
		p.SyncLoop()
	}()
	p.OnServiceSynced()
	p.OnEndpointsSynced()
}

func newServiceObject(namespace, name, clusterIP string, ports []v1.ServicePort) (*v1.Service, []proxy.ServicePortName) {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec: v1.ServiceSpec{
			ClusterIP: clusterIP,
			Ports:     ports,
		},
	}

	servicePorts := make([]proxy.ServicePortName, len(ports))
	for i, port := range ports {
		servicePorts[i] = proxy.ServicePortName{
			NamespacedName: types.NamespacedName{
				Namespace: namespace,
				Name:      name,
			},
			Port: port.Name,
		}
	}

	return service, servicePorts
}

func TestTCPProxy(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 80, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxy(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 80, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxyTimeout(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 80, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
	// When connecting to a UDP service endpoint, there should be a Conn for proxy.
	waitForNumProxyClients(t, svcInfo, 1, time.Second)
	// If conn has no activity for serviceInfo.timeout since last Read/Write, it should be closed because of timeout.
	waitForNumProxyClients(t, svcInfo, 0, 2*time.Second)
}

func TestMultiPortProxy(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{
		{Name: "p", Port: 80, Protocol: "TCP"},
		{Name: "q", Port: 80, Protocol: "UDP"},
	})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Protocol: service.Spec.Ports[0].Protocol, Port: tcpServerPort}},
		}},
	})
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[1].Port, Protocol: service.Spec.Ports[1].Protocol, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)

	svcInfo = waitForServiceInfo(t, p, ports[1], service)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestMultiPortOnServiceAdd(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{
		{Name: "p", Port: 80, Protocol: "TCP"},
		{Name: "q", Port: 81, Protocol: "UDP"},
	})

	lb := NewLoadBalancerRR()
	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	// ports p and q should exist
	_ = addServiceAndWaitForInfo(t, p, ports[0], service)
	_ = waitForServiceInfo(t, p, ports[1], service)

	// non-existent port x should not exist
	serviceX := proxy.ServicePortName{NamespacedName: ports[0].NamespacedName, Port: "x"}
	svcInfo, exists := p.getServiceInfo(serviceX)
	if exists {
		t.Fatalf("found unwanted serviceInfo for %s: %#v", serviceX, svcInfo)
	}
}

func TestTCPProxyStop(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 80, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortTCP)
}

func TestUDPProxyStop(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 80, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortUDP)
}

func TestTCPProxyUpdateDelete(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 9997, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortTCP)
}

func TestUDPProxyUpdateDelete(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 9997, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortUDP)
}

func TestTCPProxyUpdateDeleteUpdate(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 9997, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	endpoint := &v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortTCP)

	// need to add endpoint here because it got clean up during service delete
	lb.OnEndpointsAdd(endpoint)
	svcInfo = addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxyUpdateDeleteUpdate(t *testing.T) {
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 9997, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	endpoint := &v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	// Wait for the port to really close.
	deleteServiceAndWait(t, p, svcInfo, service, waitForClosedPortUDP)

	// need to add endpoint here because it got clean up during service delete
	lb.OnEndpointsAdd(endpoint)
	svcInfo = addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestTCPProxyUpdatePort(t *testing.T) {
	origPort := int32(99)
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: origPort, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)

	newService := service.DeepCopy()
	newService.Spec.Ports[0].Port = 100
	p.OnServiceUpdate(service, newService)
	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, int(origPort)); err != nil {
		t.Fatalf(err.Error())
	}
	waitForProxyFinished(t, svcInfo)

	svcInfo = waitForServiceInfo(t, p, ports[0], newService)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxyUpdatePort(t *testing.T) {
	origPort := int32(99)
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: origPort, Protocol: "UDP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: udpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)

	newService := service.DeepCopy()
	newService.Spec.Ports[0].Port = 100
	p.OnServiceUpdate(service, newService)
	// Wait for the socket to actually get free.
	if err := waitForClosedPortUDP(p, int(origPort)); err != nil {
		t.Fatalf(err.Error())
	}
	waitForProxyFinished(t, svcInfo)

	svcInfo = waitForServiceInfo(t, p, ports[0], newService)
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestProxyUpdatePublicIPs(t *testing.T) {
	origPort := int32(9997)
	service, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: origPort, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	lb.OnEndpointsAdd(&v1.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	})

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], service)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)

	newService := service.DeepCopy()
	newService.Spec.ExternalIPs = []string{"4.3.2.1"}
	p.OnServiceUpdate(service, newService)

	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, int(origPort)); err != nil {
		t.Fatalf(err.Error())
	}
	waitForProxyFinished(t, svcInfo)

	svcInfo = waitForServiceInfo(t, p, ports[0], newService)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestProxyUpdatePortal(t *testing.T) {
	svcv0, ports := newServiceObject("testnamespace", "echo", "1.2.3.4", []v1.ServicePort{{Name: "p", Port: 9997, Protocol: "TCP"}})

	lb := NewLoadBalancerRR()
	endpoint := &v1.Endpoints{
		ObjectMeta: svcv0.ObjectMeta,
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Name: ports[0].Port, Port: tcpServerPort}},
		}},
	}
	lb.OnEndpointsAdd(endpoint)

	fexec := makeFakeExec()

	p, err := createProxier(lb, net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Second, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}
	startProxier(p, t)
	defer p.shutdown()

	svcInfo := addServiceAndWaitForInfo(t, p, ports[0], svcv0)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)

	svcv1 := svcv0.DeepCopy()
	svcv1.Spec.ClusterIP = ""
	p.OnServiceUpdate(svcv0, svcv1)

	// Wait for the service to be removed because it had an empty ClusterIP
	var exists bool
	for i := 0; i < 50; i++ {
		_, exists = p.getServiceInfo(ports[0])
		if !exists {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if exists {
		t.Fatalf("service with empty ClusterIP should not be included in the proxy")
	}
	waitForProxyFinished(t, svcInfo)

	svcv2 := svcv0.DeepCopy()
	svcv2.Spec.ClusterIP = "None"
	p.OnServiceUpdate(svcv1, svcv2)
	_, exists = p.getServiceInfo(ports[0])
	if exists {
		t.Fatalf("service with 'None' as ClusterIP should not be included in the proxy")
	}

	// Set the ClusterIP again and make sure the proxy opens the port
	lb.OnEndpointsAdd(endpoint)
	p.OnServiceUpdate(svcv2, svcv0)
	svcInfo = waitForServiceInfo(t, p, ports[0], svcv0)
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

type fakeRunner struct{}

// assert fakeAsyncRunner is a ProxyProvider
var _ asyncRunnerInterface = &fakeRunner{}

func (f fakeRunner) Run() {
}

func (f fakeRunner) Loop(stop <-chan struct{}) {
}

func TestOnServiceAddChangeMap(t *testing.T) {
	fexec := makeFakeExec()

	// Use long minSyncPeriod so we can test that immediate syncs work
	p, err := createProxier(NewLoadBalancerRR(), net.ParseIP("0.0.0.0"), ipttest.NewFake(), fexec, net.ParseIP("127.0.0.1"), nil, time.Minute, time.Minute, udpIdleTimeoutForTest, newProxySocket)
	if err != nil {
		t.Fatal(err)
	}

	// Fake out sync runner
	p.syncRunner = fakeRunner{}

	serviceMeta := metav1.ObjectMeta{Namespace: "testnamespace", Name: "testname"}
	service := &v1.Service{
		ObjectMeta: serviceMeta,
		Spec: v1.ServiceSpec{ClusterIP: "1.2.3.4", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     99,
			Protocol: "TCP",
		}}},
	}

	serviceUpdate := &v1.Service{
		ObjectMeta: serviceMeta,
		Spec: v1.ServiceSpec{ClusterIP: "1.2.3.5", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     100,
			Protocol: "TCP",
		}}},
	}

	serviceUpdate2 := &v1.Service{
		ObjectMeta: serviceMeta,
		Spec: v1.ServiceSpec{ClusterIP: "1.2.3.6", Ports: []v1.ServicePort{{
			Name:     "p",
			Port:     101,
			Protocol: "TCP",
		}}},
	}

	type onServiceTest struct {
		detail         string
		changes        []serviceChange
		expectedChange *serviceChange
	}

	tests := []onServiceTest{
		{
			detail: "add",
			changes: []serviceChange{
				{current: service},
			},
			expectedChange: &serviceChange{
				current: service,
			},
		},
		{
			detail: "add+update=add",
			changes: []serviceChange{
				{current: service},
				{
					previous: service,
					current:  serviceUpdate,
				},
			},
			expectedChange: &serviceChange{
				current: serviceUpdate,
			},
		},
		{
			detail: "add+del=none",
			changes: []serviceChange{
				{current: service},
				{previous: service},
			},
		},
		{
			detail: "update+update=update",
			changes: []serviceChange{
				{
					previous: service,
					current:  serviceUpdate,
				},
				{
					previous: serviceUpdate,
					current:  serviceUpdate2,
				},
			},
			expectedChange: &serviceChange{
				previous: service,
				current:  serviceUpdate2,
			},
		},
		{
			detail: "update+del=del",
			changes: []serviceChange{
				{
					previous: service,
					current:  serviceUpdate,
				},
				{previous: serviceUpdate},
			},
			// change collapsing always keeps the oldest service
			// info since correct unmerging depends on the least
			// recent update, not the most current.
			expectedChange: &serviceChange{
				previous: service,
			},
		},
		{
			detail: "del+add=update",
			changes: []serviceChange{
				{previous: service},
				{current: serviceUpdate},
			},
			expectedChange: &serviceChange{
				previous: service,
				current:  serviceUpdate,
			},
		},
	}

	for _, test := range tests {
		for _, change := range test.changes {
			p.serviceChange(change.previous, change.current, test.detail)
		}

		if test.expectedChange != nil {
			if len(p.serviceChanges) != 1 {
				t.Fatalf("[%s] expected 1 service change but found %d", test.detail, len(p.serviceChanges))
			}
			expectedService := test.expectedChange.current
			if expectedService == nil {
				expectedService = test.expectedChange.previous
			}
			svcName := types.NamespacedName{Namespace: expectedService.Namespace, Name: expectedService.Name}

			change, ok := p.serviceChanges[svcName]
			if !ok {
				t.Fatalf("[%s] did not find service change for %v", test.detail, svcName)
			}
			if !reflect.DeepEqual(change.previous, test.expectedChange.previous) {
				t.Fatalf("[%s] change previous service and expected previous service don't match\nchange: %+v\nexp:    %+v", test.detail, change.previous, test.expectedChange.previous)
			}
			if !reflect.DeepEqual(change.current, test.expectedChange.current) {
				t.Fatalf("[%s] change current service and expected current service don't match\nchange: %+v\nexp:    %+v", test.detail, change.current, test.expectedChange.current)
			}
		} else {
			if len(p.serviceChanges) != 0 {
				t.Fatalf("[%s] expected no service changes but found %d", test.detail, len(p.serviceChanges))
			}
		}
	}
}

func TestNoopEndpointSlice(t *testing.T) {
	p := Proxier{}
	p.OnEndpointSliceAdd(&discovery.EndpointSlice{})
	p.OnEndpointSliceUpdate(&discovery.EndpointSlice{}, &discovery.EndpointSlice{})
	p.OnEndpointSliceDelete(&discovery.EndpointSlice{})
	p.OnEndpointSlicesSynced()
}

func makeFakeExec() *fakeexec.FakeExec {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
		},
	}
	return &fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}
}

// TODO(justinsb): Add test for nodePort conflict detection, once we have nodePort wired in
