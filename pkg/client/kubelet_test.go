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

package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
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

	hostURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	c := &HTTPKubeletClient{
		Client: http.DefaultClient,
		Config: &KubeletConfig{Port: uint(port)},
	}
	gotObj, _, err := c.HealthCheck(parts[0])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if gotObj != expectObj {
		t.Errorf("expected: %#v, got %#v", expectObj, gotObj)
	}
}

func TestHTTPKubeletClientError(t *testing.T) {
	expectObj := probe.Failure
	fakeHandler := util.FakeHandler{
		StatusCode:   500,
		ResponseBody: "Internal server error",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	hostURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	parts := strings.Split(hostURL.Host, ":")

	port, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	c := &HTTPKubeletClient{
		Client: http.DefaultClient,
		Config: &KubeletConfig{Port: uint(port)},
	}
	gotObj, _, err := c.HealthCheck(parts[0])
	if gotObj != expectObj {
		t.Errorf("expected: %#v, got %#v", expectObj, gotObj)
	}
}

func TestNewKubeletClient(t *testing.T) {
	config := &KubeletConfig{
		Port:        9000,
		EnableHttps: false,
	}

	client, err := NewKubeletClient(config)
	if err != nil {
		t.Errorf("Error while trying to create a client: %v", err)
	}
	if client == nil {
		t.Error("client is nil.")
	}

	host := "127.0.0.1"
	healthStatus, _, err := client.HealthCheck(host)
	if healthStatus != probe.Failure {
		t.Errorf("Expected %v and got %v.", probe.Failure, healthStatus)
	}
	if err != nil {
		t.Error("Expected a nil error")
	}
}

func TestNewKubeletClientTLSInvalid(t *testing.T) {
	config := &KubeletConfig{
		Port:        9000,
		EnableHttps: true,
		//Invalid certificate and key path
		TLSClientConfig: TLSClientConfig{
			CertFile: "./testdata/mycertinvalid.cer",
			KeyFile:  "./testdata/mycertinvalid.key",
			CAFile:   "./testdata/myCA.cer",
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
		Port:        9000,
		EnableHttps: true,
		TLSClientConfig: TLSClientConfig{
			CertFile: "./testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHttps is true.
			KeyFile: "./testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHttps is true.
			CAFile: "./testdata/myCA.cer",
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
