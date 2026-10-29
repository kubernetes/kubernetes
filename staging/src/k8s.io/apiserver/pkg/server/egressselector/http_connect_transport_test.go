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

package egressselector

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"testing"
	"testing/synctest"
	"time"
)

type pipeHTTPConnectConnector struct {
	conn net.Conn
}

func (c *pipeHTTPConnectConnector) connect(context.Context) (proxier, error) {
	return &httpConnectProxier{conn: c.conn, proxyAddress: "proxy"}, nil
}

func TestHTTPConnectTransportBoundsStalledDial(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const requestTimeout = time.Second
		const dialTimeout = 30 * time.Second

		clientConn, proxyConn := net.Pipe()
		defer clientConn.Close()
		defer proxyConn.Close()

		connectRead := make(chan error, 1)
		proxyClosed := make(chan error, 1)
		go func() {
			req, err := http.ReadRequest(bufio.NewReader(proxyConn))
			if err == nil {
				err = req.Body.Close()
				if req.Method != http.MethodConnect || req.RequestURI != "webhook.default.svc:443" {
					err = fmt.Errorf("unexpected CONNECT request: %s %s", req.Method, req.RequestURI)
				}
			}
			connectRead <- err
			if err != nil {
				return
			}
			var data [1]byte
			_, err = proxyConn.Read(data[:])
			proxyClosed <- err
		}()

		dialer := (&dialerCreator{
			connector: &pipeHTTPConnectConnector{conn: clientConn},
		}).createDialer()
		dialDone := make(chan struct {
			err     error
			elapsed time.Duration
		}, 1)
		transport := &http.Transport{
			TLSHandshakeTimeout: 10 * time.Second,
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				start := time.Now()
				conn, err := dialer(ctx, network, addr)
				dialDone <- struct {
					err     error
					elapsed time.Duration
				}{err: err, elapsed: time.Since(start)}
				return conn, err
			},
		}
		defer transport.CloseIdleConnections()

		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://webhook.default.svc/validate", nil)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		resp, err := (&http.Client{Transport: transport}).Do(req)
		if resp != nil {
			resp.Body.Close()
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected request deadline exceeded, got %v", err)
		}
		if elapsed := time.Since(start); elapsed != requestTimeout {
			t.Fatalf("request took %v, want %v", elapsed, requestTimeout)
		}
		synctest.Wait()
		select {
		case err := <-connectRead:
			if err != nil {
				t.Fatalf("reading CONNECT request: %v", err)
			}
		default:
			t.Fatal("request did not reach the CONNECT proxy")
		}

		// net/http detaches dialing from the request deadline. The dial must
		// still have its own bound, even though the request has already failed.
		time.Sleep(dialTimeout - time.Since(start))
		synctest.Wait()
		select {
		case result := <-dialDone:
			if !errors.Is(result.err, context.DeadlineExceeded) && !errors.Is(result.err, context.Canceled) {
				t.Errorf("expected dial cancellation, got %v", result.err)
			}
			if result.elapsed > dialTimeout {
				t.Errorf("dial took %v, want at most %v", result.elapsed, dialTimeout)
			}
			t.Logf("request timed out after %v; CONNECT dial finished after %v", requestTimeout, result.elapsed)
		default:
			t.Errorf("CONNECT dial is still blocked after %v despite the request timing out after %v", dialTimeout, requestTimeout)
		}
		select {
		case err := <-proxyClosed:
			if !errors.Is(err, io.EOF) {
				t.Errorf("expected the proxy connection to close, got %v", err)
			}
		default:
			t.Errorf("proxy connection is still open after %v", dialTimeout)
		}
	})
}

func TestHTTPConnectEstablishedConnectionSurvivesCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		clientConn, proxyConn := net.Pipe()
		defer clientConn.Close()
		defer proxyConn.Close()

		proxyDone := make(chan error, 1)
		go func() {
			req, err := http.ReadRequest(bufio.NewReader(proxyConn))
			if err == nil {
				err = req.Body.Close()
			}
			if err == nil {
				_, err = io.WriteString(proxyConn, "HTTP/1.1 200 Connection Established\r\n\r\n")
			}
			var data [1]byte
			if err == nil {
				_, err = io.ReadFull(proxyConn, data[:])
			}
			if err == nil {
				_, err = proxyConn.Write(data[:])
			}
			proxyDone <- err
		}()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		dialer := (&dialerCreator{
			connector: &pipeHTTPConnectConnector{conn: clientConn},
		}).createDialer()
		conn, err := dialer(ctx, "tcp", "webhook.default.svc:443")
		if err != nil {
			t.Fatalf("establishing CONNECT tunnel: %v", err)
		}
		defer conn.Close()
		cancel()
		synctest.Wait()

		if _, err := conn.Write([]byte("x")); err != nil {
			t.Fatalf("writing after dial context cancellation: %v", err)
		}
		var data [1]byte
		if _, err := io.ReadFull(conn, data[:]); err != nil {
			t.Fatalf("reading after dial context cancellation: %v", err)
		}
		if data[0] != 'x' {
			t.Errorf("unexpected tunneled data: %q", data)
		}
		if err := <-proxyDone; err != nil {
			t.Fatalf("serving CONNECT tunnel: %v", err)
		}
	})
}
