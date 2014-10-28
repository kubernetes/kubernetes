// +build integration,!no-etcd

/*
Copyright 2014 Google Inc. All rights reserved.

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

package integration

// This file tests authentication and (soon) authorization of HTTP requests to a master object.
// It does not use the client in pkg/client/... because authentication and authorization needs
// to work for any client of the HTTP interface.

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
)

func init() {
	requireEtcd()
}

// TestWhoAmI passes a known Bearer Token to the master's /_whoami endpoint and checks that
// the master authenticates the user.
func TestWhoAmI(t *testing.T) {
	deleteAllEtcdKeys()

	// Write a token file.
	json := `
abc123,alice,1
xyz987,bob,2
`
	f, err := ioutil.TempFile("", "auth_integration_test")
	f.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(f.Name())
	if err := ioutil.WriteFile(f.Name(), []byte(json), 0700); err != nil {
		t.Fatalf("unexpected error writing tokenfile: %v", err)
	}

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		EtcdHelper:        helper,
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		TokenAuthFile:     f.Name(),
	})

	s := httptest.NewServer(m.Handler)
	defer s.Close()

	// TODO: also test TLS, using e.g NewUnsafeTLSTransport() and NewClientCertTLSTransport() (see pkg/client/helper.go)
	transport := http.DefaultTransport

	testCases := []struct {
		name     string
		token    string
		expected string
		succeeds bool
	}{
		{"Valid token", "abc123", "AUTHENTICATED AS alice", true},
		{"Unknown token", "456jkl", "", false},
		{"No token", "", "", false},
	}
	for _, tc := range testCases {
		req, err := http.NewRequest("GET", s.URL+"/_whoami", nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", tc.token))

		resp, err := transport.RoundTrip(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if tc.succeeds {
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			actual := string(body)
			if tc.expected != actual {
				t.Errorf("case: %s expected: %v got: %v", tc.name, tc.expected, actual)
			}
		} else {
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("case: %s expected Unauthorized, got: %v", tc.name, resp.StatusCode)
			}

		}
	}
}
