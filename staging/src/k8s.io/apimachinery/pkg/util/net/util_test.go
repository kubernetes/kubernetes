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

package net

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"golang.org/x/net/http2"

	netutils "k8s.io/utils/net"
)

func getIPNet(cidr string) *net.IPNet {
	_, ipnet, _ := netutils.ParseCIDRSloppy(cidr)
	return ipnet
}

func TestIPNetEqual(t *testing.T) {
	testCases := []struct {
		ipnet1 *net.IPNet
		ipnet2 *net.IPNet
		expect bool
	}{
		// null case
		{
			getIPNet("10.0.0.1/24"),
			getIPNet(""),
			false,
		},
		{
			getIPNet("10.0.0.0/24"),
			getIPNet("10.0.0.0/24"),
			true,
		},
		{
			getIPNet("10.0.0.0/24"),
			getIPNet("10.0.0.1/24"),
			true,
		},
		{
			getIPNet("10.0.0.0/25"),
			getIPNet("10.0.0.0/24"),
			false,
		},
		{
			getIPNet("10.0.1.0/24"),
			getIPNet("10.0.0.0/24"),
			false,
		},
	}

	for _, tc := range testCases {
		if tc.expect != IPNetEqual(tc.ipnet1, tc.ipnet2) {
			t.Errorf("Expect equality of %s and %s be to %v", tc.ipnet1.String(), tc.ipnet2.String(), tc.expect)
		}
	}
}

func TestIsConnectionRefused(t *testing.T) {
	testCases := []struct {
		err    error
		expect bool
	}{
		{
			&url.Error{Err: &net.OpError{Err: syscall.ECONNRESET}},
			false,
		},
		{
			&url.Error{Err: &net.OpError{Err: syscall.ECONNREFUSED}},
			true,
		},
		{&url.Error{Err: &net.OpError{Err: &os.SyscallError{Err: syscall.ECONNREFUSED}}},
			true,
		},
	}

	for _, tc := range testCases {
		if result := IsConnectionRefused(tc.err); result != tc.expect {
			t.Errorf("Expect to be %v, but actual is %v", tc.expect, result)
		}
	}
}

type tcpLB struct {
	t         *testing.T
	ln        net.Listener
	serverURL string
	dials     int32
}

func (lb *tcpLB) handleConnection(in net.Conn, stopCh chan struct{}) {
	out, err := net.Dial("tcp", lb.serverURL)
	if err != nil {
		lb.t.Log(err)
		return
	}
	go io.Copy(out, in)
	go io.Copy(in, out)
	<-stopCh
	if err := out.Close(); err != nil {
		lb.t.Fatalf("failed to close connection: %v", err)
	}
}

func (lb *tcpLB) serve(stopCh chan struct{}) {
	conn, err := lb.ln.Accept()
	if err != nil {
		lb.t.Fatalf("failed to accept: %v", err)
	}
	atomic.AddInt32(&lb.dials, 1)
	go lb.handleConnection(conn, stopCh)
}

func newLB(t *testing.T, serverURL string) *tcpLB {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to bind: %v", err)
	}
	lb := tcpLB{
		serverURL: serverURL,
		ln:        ln,
		t:         t,
	}
	return &lb
}

func TestIsConnectionReset(t *testing.T) {
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, %s", r.Proto)
	}))
	ts.EnableHTTP2 = true
	ts.StartTLS()
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatalf("failed to parse URL from %q: %v", ts.URL, err)
	}
	lb := newLB(t, u.Host)
	defer lb.ln.Close()
	stopCh := make(chan struct{})
	go lb.serve(stopCh)

	c := ts.Client()
	transport, ok := ts.Client().Transport.(*http.Transport)
	if !ok {
		t.Fatalf("failed to assert *http.Transport")
	}
	t2, err := http2.ConfigureTransports(transport)
	if err != nil {
		t.Fatalf("failed to configure *http.Transport: %+v", err)
	}
	t2.ReadIdleTimeout = time.Second
	t2.PingTimeout = time.Second
	// Create an HTTP2 connection to reuse later
	resp, err := c.Get("https://" + lb.ln.Addr().String())
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if string(data) != "Hello, HTTP/2.0" {
		t.Fatalf("unexpected response: %s", data)
	}

	// Deliberately let the LB stop proxying traffic for the current
	// connection. This mimics a broken TCP connection that's not properly
	// closed.
	close(stopCh)
	_, err = c.Get("https://" + lb.ln.Addr().String())
	if !IsHTTP2ConnectionLost(err) {
		t.Fatalf("expected HTTP2ConnectionLost error, got %v", err)
	}
}
