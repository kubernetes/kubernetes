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
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"
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

// The iptables logic has to be tested in a proper end-to-end test, so this just stubs everything out.
type fakeIptables struct{}

func (fake *fakeIptables) EnsureChain(table iptables.Table, chain iptables.Chain) (bool, error) {
	return false, nil
}

func (fake *fakeIptables) FlushChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

func (fake *fakeIptables) EnsureRule(table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	return false, nil
}

func (fake *fakeIptables) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
	return nil
}

var tcpServerPort string
var udpServerPort string

func init() {
	// TCP setup.
	tcp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(r.URL.Path[1:]))
	}))
	u, err := url.Parse(tcp.URL)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	_, tcpServerPort, err = net.SplitHostPort(u.Host)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}

	// UDP setup.
	udp, err := newUDPEchoServer()
	if err != nil {
		panic(fmt.Sprintf("failed to make a UDP server: %v", err))
	}
	_, udpServerPort, err = net.SplitHostPort(udp.LocalAddr().String())
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	go udp.Loop()
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

func TestTCPProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "TCP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", udpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "UDP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

// Helper: Stops the proxy for the named service.
func stopProxyByName(proxier *Proxier, service string) error {
	info, found := proxier.getServiceInfo(service)
	if !found {
		return fmt.Errorf("unknown service: %s", service)
	}
	return proxier.stopProxy(service, info)
}

func TestTCPProxyStop(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "TCP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	stopProxyByName(p, "echo")
	// Wait for the port to really close.
	if err := waitForClosedPortTCP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestUDPProxyStop(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", udpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "UDP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	stopProxyByName(p, "echo")
	// Wait for the port to really close.
	if err := waitForClosedPortUDP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestTCPProxyUpdateDelete(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "TCP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortTCP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestUDPProxyUpdateDelete(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", udpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "UDP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortUDP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestTCPProxyUpdateDeleteUpdate(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "TCP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortTCP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	p.OnUpdate([]api.Service{
		{ObjectMeta: api.ObjectMeta{Name: "echo"}, Spec: api.ServiceSpec{Port: svcInfo.proxyPort, Protocol: "TCP", ProxyPort: svcInfo.proxyPort}, Status: api.ServiceStatus{}},
	})
	testEchoTCP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestUDPProxyUpdateDeleteUpdate(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", udpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "UDP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortUDP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	p.OnUpdate([]api.Service{
		{ObjectMeta: api.ObjectMeta{Name: "echo"}, Spec: api.ServiceSpec{Port: svcInfo.proxyPort, Protocol: "UDP", ProxyPort: svcInfo.proxyPort}, Status: api.ServiceStatus{}},
	})
	testEchoUDP(t, "127.0.0.1", svcInfo.proxyPort)
}

func TestTCPProxyUpdatePort(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "TCP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}

	// add a new dummy listener in order to get a port that is free
	l, _ := net.Listen("tcp", ":0")
	_, newPortStr, _ := net.SplitHostPort(l.Addr().String())
	newPort, _ := strconv.Atoi(newPortStr)
	l.Close()

	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, newPort); err != nil {
		t.Fatalf(err.Error())
	}
	if svcInfo.proxyPort == newPort {
		t.Errorf("expected difference, got %d %d", newPort, svcInfo.proxyPort)
	}
	p.OnUpdate([]api.Service{
		{ObjectMeta: api.ObjectMeta{Name: "echo"}, Spec: api.ServiceSpec{Port: newPort, Protocol: "TCP", ProxyPort: newPort}, Status: api.ServiceStatus{}},
	})
	if err := waitForClosedPortTCP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	testEchoTCP(t, "127.0.0.1", newPort)

	// Ensure the old port is released and re-usable.
	l, err = net.Listen("tcp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("can't claim released port: %s", err)
	}
	l.Close()
}

func TestUDPProxyUpdatePort(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			ObjectMeta: api.ObjectMeta{Name: "echo"},
			Endpoints:  []string{net.JoinHostPort("127.0.0.1", udpServerPort)},
		},
	})

	p := NewProxier(lb, net.ParseIP("127.0.0.1"), &fakeIptables{})

	svcInfo, err := p.addServiceOnPort("echo", "UDP", 0, time.Second)
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}

	// add a new dummy listener in order to get a port that is free
	pc, _ := net.ListenPacket("udp", ":0")
	_, newPortStr, _ := net.SplitHostPort(pc.LocalAddr().String())
	newPort, _ := strconv.Atoi(newPortStr)
	pc.Close()

	// Wait for the socket to actually get free.
	if err := waitForClosedPortUDP(p, newPort); err != nil {
		t.Fatalf(err.Error())
	}
	if svcInfo.proxyPort == newPort {
		t.Errorf("expected difference, got %d %d", newPort, svcInfo.proxyPort)
	}
	p.OnUpdate([]api.Service{
		{ObjectMeta: api.ObjectMeta{Name: "echo"}, Spec: api.ServiceSpec{Port: newPort, Protocol: "UDP", ProxyPort: newPort}, Status: api.ServiceStatus{}},
	})
	if err := waitForClosedPortUDP(p, svcInfo.proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	testEchoUDP(t, "127.0.0.1", newPort)

	// Ensure the old port is released and re-usable.
	pc, err = net.ListenPacket("udp", joinHostPort("", svcInfo.proxyPort))
	if err != nil {
		t.Fatalf("can't claim released port: %s", err)
	}
	pc.Close()
}

// TODO: Test UDP timeouts.
