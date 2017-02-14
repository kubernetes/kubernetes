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

package node

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

type rawJsonWebSignatureFake struct {
	Payload    string `json:"payload,omitempty"`
	Signatures string `json:"signatures,omitempty"`
	Protected  string `json:"protected,omitempty"`
	Header     string `json:"header,omitempty"`
	Signature  string `json:"signature,omitempty"`
}

func TestRetrieveTrustedClusterInfo(t *testing.T) {
	j := rawJsonWebSignatureFake{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		default:
			output, err := json.Marshal(j)
			if err != nil {
				t.Errorf("unexpected encoding error: %v", err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(output)
		}
	}))
	defer srv.Close()

	pURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("encountered an error while trying to parse httptest server url: %v", err)
	}
	host, port, err := net.SplitHostPort(pURL.Host)
	if err != nil {
		t.Fatalf("encountered an error while trying to split host and port info from httptest server: %v", err)
	}
	iPort, err := strconv.Atoi(port)
	if err != nil {
		t.Fatalf("encountered an error while trying to convert string to int (httptest server port): %v", err)
	}
	tests := []struct {
		h       string
		p       int
		payload string
		expect  bool
	}{
		{
			h:       host,
			p:       iPort,
			payload: "",
			expect:  false,
		},
		{
			h:       host,
			p:       iPort,
			payload: "foo",
			expect:  false,
		},
	}
	for _, rt := range tests {
		j.Payload = rt.payload
		nc := &kubeadmapi.TokenDiscovery{Addresses: []string{rt.h + ":" + strconv.Itoa(rt.p)}}
		_, actual := RetrieveTrustedClusterInfo(nc)
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed createClients:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}
