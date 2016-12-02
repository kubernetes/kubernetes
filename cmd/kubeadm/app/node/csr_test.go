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
	"net/http"
	"net/http/httptest"
	"testing"

	certclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

func TestPerformTLSBootstrap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		default:
			output, err := json.Marshal(nil)
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

	tests := []struct {
		h      string
		expect bool
	}{
		{
			h:      "",
			expect: false,
		},
		{
			h:      "localhost",
			expect: false,
		},
		{
			h:      srv.URL,
			expect: false,
		},
	}
	for _, rt := range tests {
		cd := &ConnectionDetails{}
		r := &restclient.Config{Host: rt.h}
		tmpConfig, err := certclient.NewForConfig(r)
		if err != nil {
			t.Fatalf("encountered an error while trying to get New Cert Client: %v", err)
		}
		cd.CertClient = tmpConfig
		_, actual := PerformTLSBootstrap(cd)
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed createClients:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}
