/*
Copyright 2017 The Kubernetes Authors.

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

package app

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

type fakeServiceResolver struct {
	// The hostport to resolve a fake address to.
	resolveHostport string
	// The last used namespace and name.
	lastNamespace, lastName string
}

func (r *fakeServiceResolver) ResolveEndpoint(namespace, name string) (*url.URL, error) {
	r.lastNamespace = namespace
	r.lastName = name
	return url.Parse(r.resolveHostport)
}

func TestAuthorizationWebhookDialer(t *testing.T) {
	tt := []struct {
		expectedNamespace, expectedName string
		serverAddress                   string
		expectedError                   error
		// If set, pass a dialer that always return the error "dialer error".
		useBlackHoleDialer bool
	}{
		{
			expectedNamespace: "servicenamespace",
			expectedName:      "servicename",
			serverAddress:     "http://servicename.servicenamespace.svc",
		},
		{
			expectedNamespace: "servicenamespace",
			expectedName:      "servicename",
			serverAddress:     "http://servicename.servicenamespace.svc",
		},
		{
			serverAddress:      "http://somename.svc",
			expectedError:      fmt.Errorf("some dial error"),
			useBlackHoleDialer: true,
		},
		{
			serverAddress:      "http://servicename.servicenamespace.svc.com",
			expectedError:      fmt.Errorf("some dial error"),
			useBlackHoleDialer: true,
		},
		{
			serverAddress:      "http://someunlikelyhostname.someunlikelydomain.someunlikelytld",
			expectedError:      fmt.Errorf("some dial error"),
			useBlackHoleDialer: true,
		},
	}
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "hello")
	}))
	defer testServer.Close()
	resolver := fakeServiceResolver{
		resolveHostport: testServer.URL,
	}
	blackHoleDial := func(string, string) (net.Conn, error) {
		return nil, errors.New("some dial error")
	}

	for i, ttt := range tt {
		var transport *http.Transport
		defaultTransport := &http.Transport{}
		if ttt.useBlackHoleDialer {
			defaultTransport.Dial = blackHoleDial
		}
		transport = &http.Transport{
			Dial: newWebhookDialer(&resolver, defaultTransport)}
		client := &http.Client{Transport: transport}
		response, err := client.Get(ttt.serverAddress)
		if err != nil {
			if ttt.expectedError == nil {
				t.Errorf("[%v] unexpected error: %v", i, err)
			} else if strings.Index(err.Error(), ttt.expectedError.Error()) == -1 {
				t.Errorf("[%v] error mismatch: expected: %v, actual: %v",
					i, ttt.expectedError, err)
			}
			continue
		}
		defer response.Body.Close()
		responseBytes, err := ioutil.ReadAll(response.Body)
		if err != nil {
			t.Fatalf("[%v] unexpected error: %v", i, err)
		}
		responseStr := string(responseBytes)
		if responseStr != "hello" {
			t.Errorf("[%v] unexpected response: '%v'", i, responseStr)
		}
		if ttt.expectedNamespace != resolver.lastNamespace ||
			ttt.expectedName != resolver.lastName {
			t.Errorf("[%v] unexpected name and namespace: expected: name=%v, namespace=%v; actual: name=%v, namespace=%v",
				i, ttt.expectedName, ttt.expectedNamespace, resolver.lastName, resolver.lastNamespace)
		}
	}
}
