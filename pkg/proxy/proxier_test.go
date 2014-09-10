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
)

func waitForClosedPortTCP(p *Proxier, proxyPort string) error {
	for i := 0; i < 50; i++ {
		_, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
		if err != nil {
			return nil
		}
		time.Sleep(1 * time.Millisecond)
	}
	return fmt.Errorf("port %s still open", proxyPort)
}

var tcpServerPort string

func init() {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(r.URL.Path[1:]))
	}))
	u, err := url.Parse(ts.URL)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
	_, tcpServerPort, err = net.SplitHostPort(u.Host)
	if err != nil {
		panic(fmt.Sprintf("failed to parse: %v", err))
	}
}

func testEchoTCP(t *testing.T, address, port string) {
	path := "aaaaa"
	res, err := http.Get("http://" + address + ":" + port + "/" + path)
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

func TestTCPProxy(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			JSONBase:  api.JSONBase{ID: "echo"},
			Endpoints: []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, "127.0.0.1")

	proxyPort, err := p.addServiceOnUnusedPort("echo", "TCP")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoTCP(t, "127.0.0.1", proxyPort)
}

func TestTCPProxyStop(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			JSONBase:  api.JSONBase{ID: "echo"},
			Endpoints: []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, "127.0.0.1")

	proxyPort, err := p.addServiceOnUnusedPort("echo", "TCP")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.StopProxy("echo")
	// Wait for the port to really close.
	if err := waitForClosedPortTCP(p, proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestTCPProxyUpdateDelete(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			JSONBase:  api.JSONBase{ID: "echo"},
			Endpoints: []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, "127.0.0.1")

	proxyPort, err := p.addServiceOnUnusedPort("echo", "TCP")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortTCP(p, proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestTCPProxyUpdateDeleteUpdate(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			JSONBase:  api.JSONBase{ID: "echo"},
			Endpoints: []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, "127.0.0.1")

	proxyPort, err := p.addServiceOnUnusedPort("echo", "TCP")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	if err := waitForClosedPortTCP(p, proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	proxyPortNum, _ := strconv.Atoi(proxyPort)
	p.OnUpdate([]api.Service{
		{JSONBase: api.JSONBase{ID: "echo"}, Port: proxyPortNum, Protocol: "TCP"},
	})
	testEchoTCP(t, "127.0.0.1", proxyPort)
}

func TestTCPProxyUpdatePort(t *testing.T) {
	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{
			JSONBase:  api.JSONBase{ID: "echo"},
			Endpoints: []string{net.JoinHostPort("127.0.0.1", tcpServerPort)},
		},
	})

	p := NewProxier(lb, "127.0.0.1")

	proxyPort, err := p.addServiceOnUnusedPort("echo", "TCP")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}

	// add a new dummy listener in order to get a port that is free
	l, _ := net.Listen("tcp", ":0")
	_, newPort, _ := net.SplitHostPort(l.Addr().String())
	newPortNum, _ := strconv.Atoi(newPort)
	l.Close()

	// Wait for the socket to actually get free.
	if err := waitForClosedPortTCP(p, newPort); err != nil {
		t.Fatalf(err.Error())
	}
	if proxyPort == newPort {
		t.Errorf("expected difference, got %s %s", newPort, proxyPort)
	}
	p.OnUpdate([]api.Service{
		{JSONBase: api.JSONBase{ID: "echo"}, Port: newPortNum, Protocol: "TCP"},
	})
	if err := waitForClosedPortTCP(p, proxyPort); err != nil {
		t.Fatalf(err.Error())
	}
	testEchoTCP(t, "127.0.0.1", newPort)

	// Ensure the old port is released and re-usable.
	l, err = net.Listen("tcp", net.JoinHostPort("", proxyPort))
	if err != nil {
		t.Fatalf("can't claim released port: %s", err)
	}
	l.Close()
}
