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
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"

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

// TestGOAWAYHTTP1Requests tests GOAWAY filter will not affect HTTP1.1 requests.
func TestGOAWAYHTTP1Requests(t *testing.T) {
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
