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
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/probe"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func TestHTTPKubeletClient(t *testing.T) {
	expectObj := probe.Success
	body, err := json.Marshal(expectObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	if _, err := url.Parse(testServer.URL); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestNewKubeletClient(t *testing.T) {
	config := &KubeletClientConfig{
		EnableHttps: false,
	}

	client, err := NewStaticKubeletClient(config)
	if err != nil {
		t.Errorf("Error while trying to create a client: %v", err)
	}
	if client == nil {
		t.Error("client is nil.")
	}
}

func TestNewKubeletClientTLSInvalid(t *testing.T) {
	config := &KubeletClientConfig{
		EnableHttps: true,
		//Invalid certificate and key path
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertinvalid.cer",
			KeyFile:  "../../client/testdata/mycertinvalid.key",
			CAFile:   "../../client/testdata/myCA.cer",
		},
	}

	client, err := NewStaticKubeletClient(config)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if client != nil {
		t.Error("client should be nil as we provided invalid cert file")
	}
}

func TestNewKubeletClientTLSValid(t *testing.T) {
	config := &KubeletClientConfig{
		Port:        1234,
		EnableHttps: true,
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHttps is true.
			KeyFile: "../../client/testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHttps is true.
			CAFile: "../../client/testdata/myCA.cer",
		},
	}

	client, err := NewStaticKubeletClient(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if client == nil {
		t.Error("client should not be nil")
	}

	{
		scheme, port, transport, err := client.GetConnectionInfo(nil, "foo")
		if err != nil {
			t.Errorf("Error getting info: %v", err)
		}
		if scheme != "https" {
			t.Errorf("Expected https, got %s", scheme)
		}
		if port != 1234 {
			t.Errorf("Expected 1234, got %d", port)
		}
		if transport == nil {
			t.Errorf("Expected transport, got nil")
		}
	}

	{
		_, _, _, err := client.GetConnectionInfo(nil, "foo bar")
		if err == nil {
			t.Errorf("Expected error getting connection info for invalid node name, got none")
		}
	}
}
