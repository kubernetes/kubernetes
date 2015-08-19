/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util"
)

func TestHTTPKubeletClient(t *testing.T) {
	expectObj := probe.Success
	body, err := json.Marshal(expectObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	_, err = url.Parse(testServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestNewKubeletClient(t *testing.T) {
	config := &KubeletConfig{
		EnableHttps: false,
	}

	client, err := NewKubeletClient(config)
	if err != nil {
		t.Errorf("Error while trying to create a client: %v", err)
	}
	if client == nil {
		t.Error("client is nil.")
	}
}

func TestNewKubeletClientTLSInvalid(t *testing.T) {
	config := &KubeletConfig{
		EnableHttps: true,
		//Invalid certificate and key path
		TLSClientConfig: TLSClientConfig{
			CertFile: "../testdata/mycertinvalid.cer",
			KeyFile:  "../testdata/mycertinvalid.key",
			CAFile:   "../testdata/myCA.cer",
		},
	}

	client, err := NewKubeletClient(config)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if client != nil {
		t.Error("client should be nil as we provided invalid cert file")
	}
}

func TestNewKubeletClientTLSValid(t *testing.T) {
	config := &KubeletConfig{
		EnableHttps: true,
		TLSClientConfig: TLSClientConfig{
			CertFile: "../testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHttps is true.
			KeyFile: "../testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHttps is true.
			CAFile: "../testdata/myCA.cer",
		},
	}

	client, err := NewKubeletClient(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if client == nil {
		t.Error("client should not be nil")
	}
}
