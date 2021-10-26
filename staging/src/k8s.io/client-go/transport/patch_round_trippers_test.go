/*
Copyright 2021 The Kubernetes Authors.

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

package transport

import (
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

func TestAltSvcRoundTripperRequest(t *testing.T) {
	u, err := url.Parse("https://orighost.com/")
	if err != nil {
		t.Fatal(err)
	}
	fakeClock := clock.NewFakeClock(time.Now())

	tests := []struct {
		name  string
		cache map[string]alternative
		last  string
		want  string
	}{
		{
			name: "no cached hosts",
			want: u.Host,
		},
		{
			name: "no cached hosts but last present",
			last: "lasthost.com",
			want: "lasthost.com",
		},
		{
			name:  "cached host should be used",
			cache: map[string]alternative{"lasthost.com": {}},
			want:  "lasthost.com",
		},
		{
			name:  "local host has preference",
			cache: map[string]alternative{"lasthost.com": {}, "localhost.com": {local: true}},
			want:  "localhost.com",
		},
		{
			name:  "last host has preference",
			cache: map[string]alternative{"lasthost.com": {}, "localhost.com": {local: true}},
			last:  "lasthost.com",
			want:  "lasthost.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &testRoundTripper{
				Response: &http.Response{
					Status:     "OK",
					StatusCode: http.StatusOK,
				},
				Err: nil,
			}
			req := &http.Request{
				URL: u,
			}
			rtAltSvc := &alternativeServiceRoundTripper{
				cache: tt.cache,
				last:  tt.last,
				rt:    rt,
				clock: fakeClock,
			}
			rtAltSvc.RoundTrip(req)
			if rt.Request.URL.Host != tt.want {
				t.Errorf("unexpected requested url %s, want %s.", rt.Request.Host, tt.want)
			}

			if len(rt.Request.Host) > 0 && rt.Request.Host != "kubernetes.default" {
				t.Errorf("unexpected requested Host header %s, want %s.", rt.Request.Host, u.Host)
			}

		})
	}
}

type mockNetError struct {
	temporary bool
	timeout   bool
}

func (e *mockNetError) Error() string {
	return fmt.Sprintf("mockNetworkError: timeout %v, temporary %v", e.timeout, e.temporary)
}

func (e *mockNetError) Temporary() bool {
	return e.temporary
}

func (e *mockNetError) Timeout() bool {
	return e.timeout
}

func TestAltSvcRoundTripperResponse(t *testing.T) {
	u, err := url.Parse("https://orighost.com/")
	if err != nil {
		t.Fatal(err)
	}
	fakeClock := clock.NewFakeClock(time.Now())
	tests := []struct {
		name     string
		response *http.Response
		err      error
		// original status
		cache map[string]alternative
		last  string
		// desired status
		wantCache map[string]alternative
		wantLast  string
		wantErr   bool
	}{
		{
			name: "no alternative services",
			response: &http.Response{
				Status:     "OK",
				StatusCode: http.StatusOK,
			},
		},
		{
			name: "one alternative service",
			response: &http.Response{
				Status:     "OK",
				StatusCode: http.StatusOK,
				Header: map[string][]string{
					"Alt-Svc": {`h2="www.domain.com:443"`},
				}},
			wantCache: map[string]alternative{"www.domain.com:443": {local: false}},
		},
		{
			name: "alternative service and error",
			response: &http.Response{
				Status:     "Fail",
				StatusCode: http.StatusConflict,
				Header: map[string][]string{
					"Alt-Svc": {`h2="www.domain.com:443"`},
				}},
			err:     fmt.Errorf("error"),
			wantErr: true,
		},
		{
			name:     "no alternative service and network error",
			response: &http.Response{},
			err:      &mockNetError{},
			wantErr:  true,
		},
		{
			name:      "alternative service and network error",
			response:  &http.Response{},
			err:       &mockNetError{},
			cache:     map[string]alternative{"lasthost.com": {local: false}},
			wantCache: map[string]alternative{"lasthost.com": {local: false, blocked: true, blockedTs: fakeClock.Now()}},
			wantErr:   true,
		},
		{
			name:      "alternative service and certificate error",
			response:  &http.Response{},
			err:       fmt.Errorf("x509: certificate is valid for %s", u.Host),
			cache:     map[string]alternative{"lasthost.com": {local: false}},
			wantCache: map[string]alternative{"lasthost.com": {local: false, blocked: true, blockedTs: time.Time{}}},
			wantErr:   true,
		},
		{
			name:      "one alternative service blocked expired",
			response:  &http.Response{},
			cache:     map[string]alternative{"www.domain.com:443": {local: false, blocked: true, blockedTs: fakeClock.Now().Add(-40 * time.Second)}},
			wantCache: map[string]alternative{"www.domain.com:443": {local: false}},
			wantLast:  "www.domain.com:443",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &testRoundTripper{
				Response: tt.response,
				Err:      tt.err,
			}
			req := &http.Request{
				URL: u,
			}
			rtAltSvc := &alternativeServiceRoundTripper{
				cache: tt.cache,
				last:  tt.last,
				clock: fakeClock,
				rt:    rt,
			}
			_, err := rtAltSvc.RoundTrip(req)

			if (err != nil) != tt.wantErr {
				t.Errorf("parseAltSvcHeader() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(rtAltSvc.cache, tt.wantCache) {
				t.Errorf("cache = %v, want %v", rtAltSvc.cache, tt.wantCache)
			}

			if rtAltSvc.last != tt.wantLast {
				t.Errorf("last = %s, want %s", rtAltSvc.last, tt.wantLast)
			}

		})
	}
}

func Test_parseAltSvcHeader(t *testing.T) {
	tests := []struct {
		name     string
		header   string
		origHost string
		want     []string
		wantErr  bool
	}{
		{
			name:     "clear cache",
			header:   `clear`,
			origHost: "testhost.com",
			want:     []string{},
		},
		{
			name:     "single alternate service without host",
			header:   `h2=":443"`,
			origHost: "testhost.com",
			want:     []string{`testhost.com:443`},
		},
		{
			name:   "single alternate service dns name",
			header: `h2="www.domain.com:443"`,
			want:   []string{`www.domain.com:443`},
		},
		{
			name:   "single alternate service IPv4 address",
			header: `h2="192.168.1.2:443"`,
			want:   []string{`192.168.1.2:443`},
		},
		{
			name:   "single alternate service IPv6 address",
			header: `h2="[2001:db8:aaaa:bbbb::]:443"`,
			want:   []string{`[2001:db8:aaaa:bbbb::]:443`},
		},
		{
			name:     "single alternate service and max age",
			header:   `h2=":443"; ma=100`,
			origHost: "testhost.com",
			want:     []string{`testhost.com:443`},
		},
		{
			name:     "single alternate service and persist",
			header:   `h2=":443"; persist=1`,
			origHost: "testhost.com",
			want:     []string{`testhost.com:443`},
		},
		{
			name:     "multiple hosts with options",
			header:   `h2="alt.example.com:443"; ma=2592000,  h2=":443"; persist=1`,
			origHost: "testhost.com",
			want:     []string{`alt.example.com:443`, `testhost.com:443`},
		},
		{
			name:   "ignore unknown options",
			header: `h2="alt.example.com:443"; ma=2592000; v=7,  h2="test.com:443"; persist=1`,
			want:   []string{`alt.example.com:443`, `test.com:443`},
		},
		{
			name:    "single alternate invalid service IPv6 address (not enclosed square brackets)",
			header:  `h2="2001:db8:aaaa:bbbb:::443"`,
			want:    []string{},
			wantErr: true,
		},
		{
			name:    "missing host and orig host",
			header:  `h2=":443"`,
			want:    []string{},
			wantErr: true,
		},
		{
			name:    "missing port",
			header:  `h2="www.domain.com"`,
			want:    []string{},
			wantErr: true,
		},
		{
			name:    "unsupported alpn protocol",
			header:  `h3=":443"`,
			want:    []string{},
			wantErr: true,
		},
		{
			name:    "missing alternative",
			header:  `ma=2500`,
			want:    []string{},
			wantErr: true,
		},
		{
			name:    "invalid parameters",
			header:  `test=test2`,
			want:    []string{},
			wantErr: true,
		}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseAltSvcHeader(tt.header, tt.origHost)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseAltSvcHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseAltSvcHeader() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_alternativeServiceRoundTripper_getAltSvc(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())

	tests := []struct {
		name  string
		cache map[string]alternative
		last  string
		host  string
		want  string
	}{
		{
			name:  "no hosts present on cache",
			cache: map[string]alternative{},
			host:  "original.host",
			want:  "original.host",
		},
		{
			name:  "last set",
			cache: map[string]alternative{},
			last:  "alternative.host",
			host:  "original.host",
			want:  "alternative.host",
		},
		{
			name:  "use alternative host",
			cache: map[string]alternative{"otherhost.host": {}},
			host:  "original.host",
			want:  "otherhost.host",
		},
		{
			name:  "prefer requested host if it is an alternative host",
			cache: map[string]alternative{"otherhost.host": {}, "original.host": {}},
			host:  "original.host",
			want:  "original.host",
		},
		{
			name:  "use alternative host, local takes precedence",
			cache: map[string]alternative{"otherhost.host": {}, "localhost.host": {local: true}},
			host:  "original.host",
			want:  "localhost.host",
		},
		{
			name:  "use alternative host, don't use blocked host",
			cache: map[string]alternative{"otherhost.host": {}, "localhost.host": {local: true, blocked: true}},
			host:  "original.host",
			want:  "otherhost.host",
		},
		{
			name:  "use original host if not alternative service available",
			cache: map[string]alternative{"otherhost.host": {blocked: true}, "localhost.host": {local: true, blocked: true}},
			host:  "original.host",
			want:  "original.host",
		},
		{
			name:  "use alternative host, use blocked host if expired",
			cache: map[string]alternative{"otherhost.host": {}, "localhost.host": {local: true, blocked: true, blockedTs: fakeClock.Now().Add(-40 * time.Second)}},
			host:  "original.host",
			want:  "localhost.host",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &alternativeServiceRoundTripper{
				cache: tt.cache,
				last:  tt.last,
				clock: fakeClock,
			}
			if got := rt.getAltSvc(tt.host); got != tt.want {
				t.Errorf("alternativeServiceRoundTripper.getAltSvc() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_alternativeServiceRoundTripper_addAlternativeServices(t *testing.T) {
	tests := []struct {
		name  string
		cache map[string]alternative
		hosts []string
		want  map[string]alternative
	}{
		{
			name:  "add new hosts to the cache",
			hosts: []string{"otherhost.host", "original.host"},
			want:  map[string]alternative{"otherhost.host": {}, "original.host": {}},
		},
		{
			name:  "replace cache with new hosts",
			cache: map[string]alternative{"otherhost.host": {}, "original.host": {}},
			hosts: []string{"new.host"},
			want:  map[string]alternative{"new.host": {}},
		},
		{
			name:  "keep hosts information on the cache",
			cache: map[string]alternative{"otherhost.host": {}, "original.host": {blocked: true}},
			hosts: []string{"original.host"},
			want:  map[string]alternative{"original.host": {blocked: true}},
		},
		{
			name:  "set local",
			hosts: []string{"localhost"},
			want:  map[string]alternative{"localhost": {local: true}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &alternativeServiceRoundTripper{
				cache: tt.cache,
			}
			rt.addAlternativeServices(tt.hosts)

			if !reflect.DeepEqual(rt.cache, tt.want) {
				t.Errorf("cache = %v, want %v", rt.cache, tt.want)
			}
		})
	}
}

func Test_alternativeServiceRoundTripper_blockHost(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())

	tests := []struct {
		name    string
		cache   map[string]alternative
		host    string
		forever bool
		want    map[string]alternative
	}{
		{
			name:    "block host",
			cache:   map[string]alternative{"otherhost.host": {}, "original.host": {}},
			host:    "otherhost.host",
			forever: false,
			want:    map[string]alternative{"otherhost.host": {blocked: true, blockedTs: fakeClock.Now()}, "original.host": {}},
		},
		{
			name:    "block host forever",
			cache:   map[string]alternative{"otherhost.host": {}, "original.host": {}},
			host:    "otherhost.host",
			forever: true,
			want:    map[string]alternative{"otherhost.host": {blocked: true, blockedTs: time.Time{}}, "original.host": {}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &alternativeServiceRoundTripper{
				cache: tt.cache,
				clock: fakeClock,
			}
			rt.blockHost(tt.host, tt.forever)

			if !reflect.DeepEqual(rt.cache, tt.want) {
				t.Errorf("cache = %v, want %v", rt.cache, tt.want)
			}
		})
	}
}
