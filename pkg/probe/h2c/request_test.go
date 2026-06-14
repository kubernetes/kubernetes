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
	"net/http"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestFormatURL(t *testing.T) {
	testCases := []struct {
		scheme string
		host   string
		port   int
		path   string
		result string
	}{
		{"http", "localhost", 93, "", "http://localhost:93"},
		{"http", "localhost", 93, "/path", "http://localhost:93/path"},
		{"http", "localhost", 93, "?foo", "http://localhost:93?foo"},
		{"http", "localhost", 93, "/path?bar", "http://localhost:93/path?bar"},
	}
	for _, test := range testCases {
		url := formatURL(test.scheme, test.host, test.port, test.path)
		if url.String() != test.result {
			t.Errorf("Expected %s, got %s", test.result, url.String())
		}
	}
}

func Test_v1HeaderToHTTPHeader(t *testing.T) {
	tests := []struct {
		name       string
		headerList []v1.HTTPHeader
		want       http.Header
	}{
		{
			name: "not empty input",
			headerList: []v1.HTTPHeader{
				{Name: "Connection", Value: "Keep-Alive"},
				{Name: "Content-Type", Value: "text/html"},
				{Name: "Accept-Ranges", Value: "bytes"},
			},
			want: http.Header{
				"Connection":    {"Keep-Alive"},
				"Content-Type":  {"text/html"},
				"Accept-Ranges": {"bytes"},
			},
		},
		{
			name: "case insensitive",
			headerList: []v1.HTTPHeader{
				{Name: "HOST", Value: "example.com"},
				{Name: "FOO-bAR", Value: "value"},
			},
			want: http.Header{
				"Host":    {"example.com"},
				"Foo-Bar": {"value"},
			},
		},
		{
			name:       "empty input",
			headerList: []v1.HTTPHeader{},
			want:       http.Header{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := v1HeaderToHTTPHeader(tt.headerList); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("v1HeaderToHTTPHeader() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHeaderConversion(t *testing.T) {
	testCases := []struct {
		headers  []v1.HTTPHeader
		expected http.Header
	}{
		{
			[]v1.HTTPHeader{
				{
					Name:  "Accept",
					Value: "application/json",
				},
			},
			http.Header{
				"Accept": []string{"application/json"},
			},
		},
		{
			[]v1.HTTPHeader{
				{Name: "accept", Value: "application/json"},
			},
			http.Header{
				"Accept": []string{"application/json"},
			},
		},
		{
			[]v1.HTTPHeader{
				{Name: "accept", Value: "application/json"},
				{Name: "Accept", Value: "*/*"},
				{Name: "pragma", Value: "no-cache"},
				{Name: "X-forwarded-for", Value: "username"},
			},
			http.Header{
				"Accept":          []string{"application/json", "*/*"},
				"Pragma":          []string{"no-cache"},
				"X-Forwarded-For": []string{"username"},
			},
		},
	}
	for _, test := range testCases {
		headers := v1HeaderToHTTPHeader(test.headers)
		if !reflect.DeepEqual(headers, test.expected) {
			t.Errorf("Expected %v, got %v", test.expected, headers)
		}
	}
}

func TestNewRequestForH2CGetAction(t *testing.T) {
	testCases := []struct {
		name       string
		action     *v1.H2CGetAction
		podIP      string
		fragment   string
		wantURL    string
		wantMethod string
		wantErr    bool
		wantHeader map[string]string
	}{
		{
			name:       "nil action",
			action:     nil,
			podIP:      "127.0.0.1",
			fragment:   "probe",
			wantErr:    true,
		},
		{
			name:       "empty path defaults to slash",
			action:     &v1.H2CGetAction{Port: 8080, Path: ""},
			podIP:      "10.0.0.1",
			fragment:   "probe",
			wantURL:    "http://10.0.0.1:8080/",
			wantMethod: http.MethodGet,
		},
		{
			name:       "path and port",
			action:     &v1.H2CGetAction{Port: 9090, Path: "/healthz"},
			podIP:      "127.0.0.1",
			fragment:   "probe",
			wantURL:    "http://127.0.0.1:9090/healthz",
			wantMethod: http.MethodGet,
		},
		{
			name: "custom headers",
			action: &v1.H2CGetAction{
				Port: 8443,
				Path: "/readyz",
				HTTPHeaders: []v1.HTTPHeader{
					{Name: "Host", Value: "app.internal"},
					{Name: "X-Custom", Value: "1"},
				},
			},
			podIP:      "192.168.1.1",
			fragment:   "probe",
			wantURL:    "http://192.168.1.1:8443/readyz",
			wantMethod: http.MethodGet,
			wantHeader: map[string]string{
				"Host":     "app.internal",
				"X-Custom": "1",
			},
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			req, err := NewRequestForH2CGetAction(test.action, test.podIP, test.fragment)
			if test.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if req.URL.String() != test.wantURL {
				t.Errorf("Expected URL %s, got %s", test.wantURL, req.URL.String())
			}
			if req.Method != test.wantMethod {
				t.Errorf("Expected method %s, got %s", test.wantMethod, req.Method)
			}
			if ua := req.Header.Get("User-Agent"); !strings.HasPrefix(ua, "kube-"+test.fragment) {
				t.Errorf("Expected User-Agent prefix kube-%s, got %q", test.fragment, ua)
			}
			if req.Header.Get("Accept") != "*/*" {
				t.Errorf("Expected Accept */*, got %q", req.Header.Get("Accept"))
			}
			for k, want := range test.wantHeader {
				if got := req.Header.Get(k); got != want {
					t.Errorf("header %s: want %q, got %q", k, want, got)
				}
			}
		})
	}
}
