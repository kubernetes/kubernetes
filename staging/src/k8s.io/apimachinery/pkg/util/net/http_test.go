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
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestCloneTLSConfig(t *testing.T) {
	expected := sets.NewString(
		// These fields are copied in CloneTLSConfig
		"Rand",
		"Time",
		"Certificates",
		"RootCAs",
		"NextProtos",
		"ServerName",
		"InsecureSkipVerify",
		"CipherSuites",
		"PreferServerCipherSuites",
		"MinVersion",
		"MaxVersion",
		"CurvePreferences",
		"NameToCertificate",
		"GetCertificate",
		"ClientAuth",
		"ClientCAs",
		"ClientSessionCache",

		// These fields are not copied
		"SessionTicketsDisabled",
		"SessionTicketKey",

		// These fields are unexported
		"serverInitOnce",
		"mutex",
		"sessionTicketKeys",
	)

	// See #33936.
	if strings.HasPrefix(runtime.Version(), "go1.7") {
		expected.Insert("DynamicRecordSizingDisabled", "Renegotiation")
	}

	fields := sets.NewString()
	structType := reflect.TypeOf(tls.Config{})
	for i := 0; i < structType.NumField(); i++ {
		fields.Insert(structType.Field(i).Name)
	}

	if missing := expected.Difference(fields); len(missing) > 0 {
		t.Errorf("Expected fields that were not seen in http.Transport: %v", missing.List())
	}
	if extra := fields.Difference(expected); len(extra) > 0 {
		t.Errorf("New fields seen in http.Transport: %v\nAdd to CopyClientTLSConfig if client-relevant, then add to expected list in TestCopyClientTLSConfig", extra.List())
	}
}

func TestGetClientIP(t *testing.T) {
	ipString := "10.0.0.1"
	ip := net.ParseIP(ipString)
	invalidIPString := "invalidIPString"
	testCases := []struct {
		Request    http.Request
		ExpectedIP net.IP
	}{
		{
			Request: http.Request{},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {invalidIPString},
				},
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString},
				},
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString + "," + ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				// RemoteAddr is in the form host:port
				RemoteAddr: ipString + ":1234",
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				RemoteAddr: invalidIPString,
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString},
				},
				// RemoteAddr is in the form host:port
				RemoteAddr: ipString,
			},
			ExpectedIP: ip,
		},
	}

	for i, test := range testCases {
		if a, e := GetClientIP(&test.Request), test.ExpectedIP; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i, e, a)
		}
	}
}

func TestProxierWithNoProxyCIDR(t *testing.T) {
	testCases := []struct {
		name    string
		noProxy string
		url     string

		expectedDelegated bool
	}{
		{
			name:              "no env",
			url:               "https://192.168.143.1/api",
			expectedDelegated: true,
		},
		{
			name:              "no cidr",
			noProxy:           "192.168.63.1",
			url:               "https://192.168.143.1/api",
			expectedDelegated: true,
		},
		{
			name:              "hostname",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://my-hostname/api",
			expectedDelegated: true,
		},
		{
			name:              "match second cidr",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://192.168.143.1/api",
			expectedDelegated: false,
		},
		{
			name:              "match second cidr with host:port",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://192.168.143.1:8443/api",
			expectedDelegated: false,
		},
	}

	for _, test := range testCases {
		os.Setenv("NO_PROXY", test.noProxy)
		actualDelegated := false
		proxyFunc := NewProxierWithNoProxyCIDR(func(req *http.Request) (*url.URL, error) {
			actualDelegated = true
			return nil, nil
		})

		req, err := http.NewRequest("GET", test.url, nil)
		if err != nil {
			t.Errorf("%s: unexpected err: %v", test.name, err)
			continue
		}
		if _, err := proxyFunc(req); err != nil {
			t.Errorf("%s: unexpected err: %v", test.name, err)
			continue
		}

		if test.expectedDelegated != actualDelegated {
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedDelegated, actualDelegated)
			continue
		}
	}
}

type fakeTLSClientConfigHolder struct {
	called bool
}

func (f *fakeTLSClientConfigHolder) TLSClientConfig() *tls.Config {
	f.called = true
	return nil
}
func (f *fakeTLSClientConfigHolder) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}

func TestTLSClientConfigHolder(t *testing.T) {
	rt := &fakeTLSClientConfigHolder{}
	TLSClientConfig(rt)

	if !rt.called {
		t.Errorf("didn't find tls config")
	}
}
