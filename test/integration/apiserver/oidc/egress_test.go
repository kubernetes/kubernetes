/*
Copyright 2025 The Kubernetes Authors.

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

package oidc

import (
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func runEgressProxy(t testing.TB, udsName string, ready chan<- struct{}) {
	t.Helper()

	l, err := net.Listen("unix", udsName)
	if err != nil {
		t.Errorf("unexpected UDS error: %v", err)
		return
	}

	var called atomic.Bool
	httpConnectProxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ready" {
			t.Log("egress proxy ready")
			w.WriteHeader(http.StatusOK)
			return
		}

		called.Store(true)

		if r.Method != http.MethodConnect {
			http.Error(w, "this proxy only supports CONNECT passthrough", http.StatusMethodNotAllowed)
			return
		}

		backendConn, err := (&net.Dialer{}).DialContext(r.Context(), "tcp", r.Host)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = backendConn.Close() }()

		hijacker, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijacking not supported", http.StatusInternalServerError)
			return
		}

		requestHijackedConn, _, err := hijacker.Hijack()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = requestHijackedConn.Close() }()

		// use t.Errorf for all errors after this Write since the client may think the connection is good
		_, err = requestHijackedConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		if err != nil {
			t.Errorf("unexpected established error: %v", err)
			return
		}

		writerComplete := make(chan struct{})
		readerComplete := make(chan struct{})

		go func() {
			_, err := io.Copy(backendConn, requestHijackedConn)
			if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
				t.Errorf("unexpected writer error: %v", err)
			}
			close(writerComplete)
		}()

		go func() {
			_, err := io.Copy(requestHijackedConn, backendConn)
			if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
				t.Errorf("unexpected reader error: %v", err)
			}
			close(readerComplete)
		}()

		// Wait for one half the connection to exit. Once it does,
		// the defer will clean up the other half of the connection.
		select {
		case <-writerComplete:
		case <-readerComplete:
		}
	})

	server := http.Server{Handler: httpConnectProxy}

	t.Cleanup(func() {
		if !called.Load() {
			t.Errorf("egress proxy was not called")
		}

		err := server.Shutdown(context.Background())
		t.Logf("shutdown exit error: %v", err)
	})

	var once sync.Once
	readyCheckClient := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", udsName)
			},
		},
	}
	go func() {
		if err := wait.PollUntilContextCancel(t.Context(), time.Second, false, func(ctx context.Context) (bool, error) {
			resp, err := readyCheckClient.Get("http://host.does.not.matter/ready")
			if err != nil {
				t.Logf("egress proxy error: %v", err)
				return false, nil
			}
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Logf("egress proxy unexpected status code: %v", resp.StatusCode)
				return false, nil
			}
			once.Do(func() { close(ready) })
			return true, nil
		}); err != nil {
			t.Errorf("egress proxy is not ready: %v", err)
		}
	}()

	err = server.Serve(l)
	t.Logf("egress exit error: %v", err)
}
