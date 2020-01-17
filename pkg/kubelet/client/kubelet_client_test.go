/*
Copyright 2014 The Kubernetes Authors.

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

package client

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strconv"
	"testing"

	restclient "k8s.io/client-go/rest"
)

func TestMakeTransportInvalid(t *testing.T) {
	config := &KubeletClientConfig{
		EnableHTTPS: true,
		//Invalid certificate and key path
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertinvalid.cer",
			KeyFile:  "../../client/testdata/mycertinvalid.key",
			CAFile:   "../../client/testdata/myCA.cer",
		},
	}

	rt, err := MakeTransport(config)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if rt != nil {
		t.Error("rt should be nil as we provided invalid cert file")
	}
}

func TestMakeTransportValid(t *testing.T) {
	config := &KubeletClientConfig{
		Port:        1234,
		EnableHTTPS: true,
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHTTPS is true.
			KeyFile: "../../client/testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHTTPS is true.
			CAFile: "../../client/testdata/myCA.cer",
		},
	}

	rt, err := MakeTransport(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if rt == nil {
		t.Error("rt should not be nil")
	}
}

func TestMakeInsecureTransport(t *testing.T) {
	testServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	testURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, portStr, err := net.SplitHostPort(testURL.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.ParseUint(portStr, 10, 32)
	if err != nil {
		t.Fatal(err)
	}

	config := &KubeletClientConfig{
		Port:        uint(port),
		EnableHTTPS: true,
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHTTPS is true.
			KeyFile: "../../client/testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHTTPS is true.
			CAFile: "../../client/testdata/myCA.cer",
		},
	}

	rt, err := MakeInsecureTransport(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if rt == nil {
		t.Error("rt should not be nil")
	}

	req, err := http.NewRequest(http.MethodGet, testServer.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		dump, err := httputil.DumpResponse(response, true)
		if err != nil {
			t.Fatal(err)
		}
		t.Fatal(string(dump))
	}
}
