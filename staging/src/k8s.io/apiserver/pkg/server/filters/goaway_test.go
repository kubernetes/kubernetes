/*
Copyright 2020 The Kubernetes Authors.

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

package filters

import (
	"crypto/tls"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/net/http2"
)

func TestProbabilisticGoawayDecider(t *testing.T) {
	cases := []struct {
		name         string
		chance       float64
		nextFn       func(chance float64) func() float64
		expectGOAWAY bool
	}{
		{
			name:   "always not GOAWAY",
			chance: 0,
			nextFn: func(chance float64) func() float64 {
				return rand.Float64
			},
			expectGOAWAY: false,
		},
		{
			name:   "always GOAWAY",
			chance: 1,
			nextFn: func(chance float64) func() float64 {
				return rand.Float64
			},
			expectGOAWAY: true,
		},
		{
			name:   "hit GOAWAY",
			chance: rand.Float64() + 0.01,
			nextFn: func(chance float64) func() float64 {
				return func() float64 {
					return chance - 0.001
				}
			},
			expectGOAWAY: true,
		},
		{
			name:   "does not hit GOAWAY",
			chance: rand.Float64() + 0.01,
			nextFn: func(chance float64) func() float64 {
				return func() float64 {
					return chance + 0.001
				}
			},
			expectGOAWAY: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := probabilisticGoawayDecider{chance: tc.chance, next: tc.nextFn(tc.chance)}
			result := d.Goaway(nil)
			if result != tc.expectGOAWAY {
				t.Errorf("expect GOAWAY: %v, got: %v", tc.expectGOAWAY, result)
			}
		})
	}
}

// TestClientReceivedGOAWAY tests the in-flight watch requests will not be affected and new requests use a
// connection after client received GOAWAY, and server response watch request with GOAWAY will not break client
// watching body read.
func TestClientReceivedGOAWAY(t *testing.T) {
	const (
		urlNormal          = "/normal"
		urlWatch           = "/watch"
		urlGoaway          = "/goaway"
		urlWatchWithGoaway = "/watch-with-goaway"
	)

	const (
		// indicate the bytes watch request will be sent
		// used to check if watch request was broke by GOAWAY
		watchExpectSendBytes = 5
	)

	watchHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timer := time.NewTicker(time.Second)

		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(200)

		flusher, _ := w.(http.Flusher)
		flusher.Flush()

		count := 0
		for {
			select {
			case <-timer.C:
				n, err := w.Write([]byte("w"))
				if err != nil {
					return
				}
				flusher.Flush()
				count += n
				if count == watchExpectSendBytes {
					return
				}
			}
		}
	})

	mux := http.NewServeMux()
	mux.Handle(urlNormal, WithProbabilisticGoaway(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hello"))
		return
	}), 0))
	mux.Handle(urlWatch, WithProbabilisticGoaway(watchHandler, 0))
	mux.Handle(urlGoaway, WithProbabilisticGoaway(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hello"))
		return
	}), 1))
	mux.Handle(urlWatchWithGoaway, WithProbabilisticGoaway(watchHandler, 1))

	s := httptest.NewUnstartedServer(mux)

	http2Options := &http2.Server{}

	if err := http2.ConfigureServer(s.Config, http2Options); err != nil {
		t.Fatalf("failed to configure test server to be HTTP2 server, err: %v", err)
	}

	s.TLS = s.Config.TLSConfig
	s.StartTLS()
	defer s.Close()

	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{http2.NextProtoTLS},
	}

	cases := []struct {
		name string
		reqs []string
		// expectConnections always equals to GOAWAY requests(urlGoaway or urlWatchWithGoaway) + 1
		expectConnections int
	}{
		{
			name:              "all normal requests use only one connection",
			reqs:              []string{urlNormal, urlNormal, urlNormal},
			expectConnections: 1,
		},
		{
			name:              "got GOAWAY after set-up watch",
			reqs:              []string{urlNormal, urlWatch, urlGoaway, urlNormal, urlNormal},
			expectConnections: 2,
		},
		{
			name:              "got GOAWAY after set-up watch, and set-up a new watch",
			reqs:              []string{urlNormal, urlWatch, urlGoaway, urlWatch, urlNormal, urlNormal},
			expectConnections: 2,
		},
		{
			name:              "got 2 GOAWAY after set-up watch",
			reqs:              []string{urlNormal, urlWatch, urlGoaway, urlGoaway, urlNormal, urlNormal},
			expectConnections: 3,
		},
		{
			name:              "combine with watch-with-goaway",
			reqs:              []string{urlNormal, urlWatchWithGoaway, urlNormal, urlWatch, urlGoaway, urlNormal, urlNormal},
			expectConnections: 3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// localAddr indicates how many TCP connection set up
			localAddr := make([]string, 0)

			// init HTTP2 client
			client := http.Client{
				Transport: &http2.Transport{
					TLSClientConfig: tlsConfig,
					DialTLS: func(network, addr string, cfg *tls.Config) (conn net.Conn, err error) {
						conn, err = tls.Dial(network, addr, cfg)
						if err != nil {
							t.Fatalf("unexpect connection err: %v", err)
						}
						localAddr = append(localAddr, conn.LocalAddr().String())
						return
					},
				},
			}

			watchChs := make([]chan int, 0)
			for _, url := range tc.reqs {
				req, err := http.NewRequest(http.MethodGet, s.URL+url, nil)
				if err != nil {
					t.Fatalf("unexpect new request error: %v", err)
				}
				resp, err := client.Do(req)
				if err != nil {
					t.Fatalf("failed request test server, err: %v", err)
				}

				// encounter watch bytes received, does not expect to be broken
				if url == urlWatch || url == urlWatchWithGoaway {
					ch := make(chan int)
					watchChs = append(watchChs, ch)
					go func() {
						count := 0
						for {
							buffer := make([]byte, 1)
							n, err := resp.Body.Read(buffer)
							if err != nil {
								// urlWatch will receive io.EOF,
								// urlWatchWithGoaway will receive http2.GoAwayError
								if err != io.EOF {
									if _, ok := err.(http2.GoAwayError); !ok {
										t.Errorf("watch received not EOF err: %v", err)
									}
								}
								ch <- count
								return
							}
							count += n
						}
					}()
				}
			}

			// check TCP connection count
			if tc.expectConnections != len(localAddr) {
				t.Fatalf("expect TCP connection: %d, actual: %d", tc.expectConnections, len(localAddr))
			}

			// check if watch request is broken by GOAWAY response
			watchTimeout := time.NewTimer(time.Second * 10)
			for _, watchCh := range watchChs {
				select {
				case n := <-watchCh:
					if n != watchExpectSendBytes {
						t.Fatalf("in-flight watch was broken by GOAWAY response, expect go bytes: %d, actual got: %d", watchExpectSendBytes, n)
					}
				case <-watchTimeout.C:
					t.Error("watch receive timeout")
				}
			}
		})
	}
}

func TestHTTP1Requests(t *testing.T) {
	s := httptest.NewUnstartedServer(WithProbabilisticGoaway(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hello"))
		return
	}), 1))

	http2Options := &http2.Server{}

	if err := http2.ConfigureServer(s.Config, http2Options); err != nil {
		t.Fatalf("failed to configure test server to be HTTP2 server, err: %v", err)
	}

	s.TLS = s.Config.TLSConfig
	s.StartTLS()
	defer s.Close()

	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"http/1.1"},
	}

	client := http.Client{
		Transport: &http.Transport{
			TLSClientConfig: tlsConfig,
		},
	}

	resp, err := client.Get(s.URL)
	if err != nil {
		t.Fatalf("failed to request the server, err: %v", err)
	}

	if v := resp.Header.Get("Connection"); v != "" {
		t.Errorf("expect response HTTP header Connection to be empty, but got: %s", v)
	}
}
