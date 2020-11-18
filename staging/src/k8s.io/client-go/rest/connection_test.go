/*
Copyright 2019 The Kubernetes Authors.

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

package rest

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilnet "k8s.io/apimachinery/pkg/util/net"
)

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

func setEnv(key, value string) func() {
	originalValue := os.Getenv(key)
	os.Setenv(key, value)
	return func() {
		os.Setenv(key, originalValue)
	}
}

const (
	readIdleTimeout int = 1
	pingTimeout     int = 1
)

func TestReconnectBrokenTCP(t *testing.T) {
	defer setEnv("HTTP2_READ_IDLE_TIMEOUT_SECONDS", strconv.Itoa(readIdleTimeout))()
	defer setEnv("HTTP2_PING_TIMEOUT_SECONDS", strconv.Itoa(pingTimeout))()
	defer setEnv("DISABLE_HTTP2", "")()
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
	transport, ok := ts.Client().Transport.(*http.Transport)
	if !ok {
		t.Fatalf("failed to assert *http.Transport")
	}
	config := &Config{
		Host:      "https://" + lb.ln.Addr().String(),
		Transport: utilnet.SetTransportDefaults(transport),
		Timeout:   1 * time.Second,
		// These fields are required to create a REST client.
		ContentConfig: ContentConfig{
			GroupVersion:         &schema.GroupVersion{},
			NegotiatedSerializer: &serializer.CodecFactory{},
		},
	}
	client, err := RESTClientFor(config)
	if err != nil {
		t.Fatalf("failed to create REST client: %v", err)
	}
	data, err := client.Get().AbsPath("/").DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("unexpected err: %s: %v", data, err)
	}
	if string(data) != "Hello, HTTP/2.0" {
		t.Fatalf("unexpected response: %s", data)
	}

	// Deliberately let the LB stop proxying traffic for the current
	// connection. This mimics a broken TCP connection that's not properly
	// closed.
	close(stopCh)

	stopCh = make(chan struct{})
	go lb.serve(stopCh)
	// Sleep enough time for the HTTP/2 health check to detect and close
	// the broken TCP connection.
	time.Sleep(time.Duration(1+readIdleTimeout+pingTimeout) * time.Second)
	// If the HTTP/2 health check were disabled, the broken connection
	// would still be in the connection pool, the following request would
	// then reuse the broken connection instead of creating a new one, and
	// thus would fail.
	data, err = client.Get().AbsPath("/").DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if string(data) != "Hello, HTTP/2.0" {
		t.Fatalf("unexpected response: %s", data)
	}
	dials := atomic.LoadInt32(&lb.dials)
	if dials != 2 {
		t.Fatalf("expected %d dials, got %d", 2, dials)
	}
}
