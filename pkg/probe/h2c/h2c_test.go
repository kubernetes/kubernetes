/*
Copyright The Kubernetes Authors.

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

package h2c

import (
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"golang.org/x/net/http2"

	"k8s.io/kubernetes/pkg/probe"
)

func TestNew(t *testing.T) {
	t.Run("Should: implement Prober interface", func(t *testing.T) {
		s := New()
		assert.Implements(t, (*Prober)(nil), s)
	})
}

// startH2CServer serves HTTP/2 in cleartext (h2c / prior knowledge) using only the vendored
// x/net/http2 server, matching what h2c-capable clients negotiate.
func startH2CServer(t *testing.T, handler http.Handler) (addr string, cleanup func()) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	h2s := &http2.Server{}
	base := &http.Server{Handler: handler}
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			cc := c
			go func() {
				h2s.ServeConn(cc, &http2.ServeConnOpts{
					Handler:    handler,
					BaseConfig: base,
				})
			}()
		}
	}()
	return l.Addr().String(), func() { _ = l.Close() }
}

func TestH2CProber_Probe(t *testing.T) {
	t.Run("Should: return failure with nil error when connection is refused", func(t *testing.T) {
		p := New()
		req, err := http.NewRequest(http.MethodGet, "http://127.0.0.1:1/healthz", nil)
		if err != nil {
			t.Fatal(err)
		}
		res, msg, err := p.Probe(req, time.Second)
		assert.Equal(t, probe.Failure, res)
		assert.NoError(t, err)
		assert.Contains(t, msg, "dial tcp")
	})

	t.Run("Should: return success when h2c server responds 200", func(t *testing.T) {
		const body = "healthy"
		addr, cleanup := startH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, http.MethodGet, r.Method)
			assert.Equal(t, "HTTP/2.0", r.Proto, "probe must use HTTP/2 on the wire, not HTTP/1.1")
			w.WriteHeader(http.StatusOK)
			_, _ = fmt.Fprint(w, body)
		}))
		t.Cleanup(cleanup)

		// Allow the server accept loop to start (same delay style as pkg/probe/grpc tests).
		time.Sleep(2 * time.Second)

		url := fmt.Sprintf("http://%s/readyz", addr)
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			t.Fatal(err)
		}
		p := New()
		res, msg, err := p.Probe(req, 5*time.Second)
		assert.Equal(t, probe.Success, res)
		assert.NoError(t, err)
		assert.Equal(t, body, msg)
	})

	t.Run("Should: return failure with nil error on non-2xx status", func(t *testing.T) {
		addr, cleanup := startH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = fmt.Fprint(w, "unavailable")
		}))
		t.Cleanup(cleanup)
		time.Sleep(2 * time.Second)

		url := fmt.Sprintf("http://%s/", addr)
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			t.Fatal(err)
		}
		p := New()
		res, msg, err := p.Probe(req, 5*time.Second)
		assert.Equal(t, probe.Failure, res)
		assert.NoError(t, err)
		assert.Contains(t, msg, "503")
	})

	t.Run("Should: return failure when request times out", func(t *testing.T) {
		addr, cleanup := startH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(10 * time.Second)
			w.WriteHeader(http.StatusOK)
		}))
		t.Cleanup(cleanup)
		time.Sleep(2 * time.Second)

		url := fmt.Sprintf("http://%s/slow", addr)
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			t.Fatal(err)
		}
		p := New()
		res, _, err := p.Probe(req, 500*time.Millisecond)
		assert.Equal(t, probe.Failure, res)
		assert.NoError(t, err)
	})
}
