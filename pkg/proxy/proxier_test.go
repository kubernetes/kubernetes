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
	"testing"

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

func TestProxy(t *testing.T) {
	port, err := echoServer(t, "127.0.0.1:")
	if err != nil {
		t.Fatal(err)
	}

	lb := NewLoadBalancerRR()
	lb.OnUpdate([]api.Endpoints{{"echo", []string{net.JoinHostPort("127.0.0.1", port)}}})

	p := NewProxier(lb)

	proxyPort, err := p.addServiceOnUnusedPort("echo")
	if err != nil {
		t.Fatalf("error adding new service: %#v", err)
	}
	conn, err := net.Dial("tcp", net.JoinHostPort("127.0.0.1", proxyPort))
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
