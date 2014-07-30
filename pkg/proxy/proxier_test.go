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
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// a simple echoServer that only accepts one connection. Returns port actually
// being listened on, or an error.
func echoServer(t *testing.T, addr string) (string, error) {
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return "", fmt.Errorf("failed to start echo service: %v", err)
	}
	go func() {
		defer l.Close()
		conn, err := l.Accept()
		if err != nil {
			t.Errorf("failed to accept new conn to echo service: %v", err)
		}
		io.Copy(conn, conn)
		conn.Close()
	}()
	_, port, err := net.SplitHostPort(l.Addr().String())
	return port, err
}

func testEchoConnection(t *testing.T, address, port string) {
	conn, err := net.Dial("tcp", net.JoinHostPort(address, port))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	magic := "aaaaa"
	if _, err := conn.Write([]byte(magic)); err != nil {
		t.Fatalf("error writing to proxy: %v", err)
	}
	buf := make([]byte, 5)
	if _, err := conn.Read(buf); err != nil {
		t.Fatalf("error reading from proxy: %v", err)
	}
	if string(buf) != magic {
		t.Fatalf("bad echo from proxy: got: %q, expected %q", string(buf), magic)
	}
}

func TestProxy(t *testing.T) {
	port, err := echoServer(t, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{
		{JSONBase: api.JSONBase{ID: "echo"}, Endpoints: []string{net.JoinHostPort("127.0.0.1", port)}}})

	p := NewProxier(lb)

	proxyPort, err := p.addServiceOnUnusedPort("echo")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	testEchoConnection(t, "127.0.0.1", proxyPort)
}

func TestProxyStop(t *testing.T) {
	port, err := echoServer(t, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{{JSONBase: api.JSONBase{ID: "echo"}, Endpoints: []string{net.JoinHostPort("127.0.0.1", port)}}})

	p := NewProxier(lb)

	proxyPort, err := p.addServiceOnUnusedPort("echo")
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
	time.Sleep(2 * time.Second)
	_, err = net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err == nil {
		t.Fatalf("Unexpected non-error.")
	}
}

func TestProxyUpdateDelete(t *testing.T) {
	port, err := echoServer(t, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{{JSONBase: api.JSONBase{ID: "echo"}, Endpoints: []string{net.JoinHostPort("127.0.0.1", port)}}})

	p := NewProxier(lb)

	proxyPort, err := p.addServiceOnUnusedPort("echo")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err != nil {
		t.Fatalf("error connecting to proxy: %v", err)
	}
	conn.Close()

	p.OnUpdate([]api.Service{})
	// Wait for the port to close.
	time.Sleep(2 * time.Second)
	_, err = net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err == nil {
		t.Fatalf("Unexpected non-error.")
	}
}

func TestProxyUpdatePort(t *testing.T) {
	port, err := echoServer(t, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{{JSONBase: api.JSONBase{ID: "echo"}, Endpoints: []string{net.JoinHostPort("127.0.0.1", port)}}})

	p := NewProxier(lb)

	proxyPort, err := p.addServiceOnUnusedPort("echo")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}

	// add a new dummy listener in order to get a port that is free
	l, _ := net.Listen("tcp", ":0")
	_, port, _ = net.SplitHostPort(l.Addr().String())
	portNum, _ := strconv.Atoi(port)
	l.Close()

	// Wait for the socket to actually get free.
	time.Sleep(2 * time.Second)
	p.OnUpdate([]api.Service{
		{JSONBase: api.JSONBase{ID: "echo"}, Port: portNum},
	})
	time.Sleep(2 * time.Second)
	_, err = net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
	if err == nil {
		t.Fatalf("Unexpected non-error.")
	}
	testEchoConnection(t, "127.0.0.1", port)
}
