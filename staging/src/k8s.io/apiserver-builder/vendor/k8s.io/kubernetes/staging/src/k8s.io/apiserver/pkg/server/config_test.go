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

package server

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/client-go/rest"
)

func TestNewWithDelegate(t *testing.T) {
	delegateConfig := NewConfig(codecs)
	delegateConfig.PublicAddress = net.ParseIP("192.168.10.4")
	delegateConfig.RequestContextMapper = genericapirequest.NewRequestContextMapper()
	delegateConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	delegateConfig.LoopbackClientConfig = &rest.Config{}
	delegateConfig.FallThroughHandler = mux.NewPathRecorderMux()
	delegateConfig.SwaggerConfig = DefaultSwaggerConfig()

	delegateHealthzCalled := false
	delegateConfig.HealthzChecks = append(delegateConfig.HealthzChecks, healthz.NamedCheck("delegate-health", func(r *http.Request) error {
		delegateHealthzCalled = true
		return fmt.Errorf("delegate failed healthcheck")
	}))

	delegateConfig.FallThroughHandler.HandleFunc("/foo", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	})

	delegateServer, err := delegateConfig.SkipComplete().New()
	if err != nil {
		t.Fatal(err)
	}

	delegateServer.AddPostStartHook("delegate-post-start-hook", func(context PostStartHookContext) error {
		return nil
	})

	// this wires up swagger
	delegateServer.PrepareRun()

	wrappingConfig := NewConfig(codecs)
	wrappingConfig.PublicAddress = net.ParseIP("192.168.10.4")
	wrappingConfig.RequestContextMapper = genericapirequest.NewRequestContextMapper()
	wrappingConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	wrappingConfig.LoopbackClientConfig = &rest.Config{}
	wrappingConfig.FallThroughHandler = mux.NewPathRecorderMux()
	wrappingConfig.SwaggerConfig = DefaultSwaggerConfig()

	wrappingHealthzCalled := false
	wrappingConfig.HealthzChecks = append(wrappingConfig.HealthzChecks, healthz.NamedCheck("wrapping-health", func(r *http.Request) error {
		wrappingHealthzCalled = true
		return fmt.Errorf("wrapping failed healthcheck")
	}))

	wrappingConfig.FallThroughHandler.HandleFunc("/bar", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	})

	wrappingServer, err := wrappingConfig.Complete().NewWithDelegate(delegateServer)
	if err != nil {
		t.Fatal(err)
	}

	wrappingServer.AddPostStartHook("wrapping-post-start-hook", func(context PostStartHookContext) error {
		return nil
	})

	stopCh := make(chan struct{})
	defer close(stopCh)
	wrappingServer.PrepareRun()
	wrappingServer.RunPostStartHooks()

	server := httptest.NewServer(wrappingServer.Handler)
	defer server.Close()

	checkPath(server.URL, http.StatusOK, `{
  "paths": [
    "/apis",
    "/bar",
    "/foo",
    "/healthz",
    "/healthz/delegate-health",
    "/healthz/ping",
    "/healthz/poststarthook/delegate-post-start-hook",
    "/healthz/poststarthook/wrapping-post-start-hook",
    "/healthz/wrapping-health",
    "/swaggerapi/"
  ]
}`, t)
	checkPath(server.URL+"/healthz", http.StatusInternalServerError, `[+]ping ok
[-]wrapping-health failed: reason withheld
[-]delegate-health failed: reason withheld
[+]poststarthook/delegate-post-start-hook ok
[+]poststarthook/wrapping-post-start-hook ok
healthz check failed
`, t)

	checkPath(server.URL+"/healthz/delegate-health", http.StatusInternalServerError, `internal server error: delegate failed healthcheck
`, t)
	checkPath(server.URL+"/healthz/wrapping-health", http.StatusInternalServerError, `internal server error: wrapping failed healthcheck
`, t)
	checkPath(server.URL+"/healthz/poststarthook/delegate-post-start-hook", http.StatusOK, `ok`, t)
	checkPath(server.URL+"/healthz/poststarthook/wrapping-post-start-hook", http.StatusOK, `ok`, t)
	checkPath(server.URL+"/foo", http.StatusForbidden, ``, t)
	checkPath(server.URL+"/bar", http.StatusUnauthorized, ``, t)
}

func checkPath(url string, expectedStatusCode int, expectedBody string, t *testing.T) {
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	dump, _ := httputil.DumpResponse(resp, true)
	t.Log(string(dump))

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if e, a := expectedBody, string(body); e != a {
		t.Errorf("%q expected %v, got %v", url, e, a)
	}
	if e, a := expectedStatusCode, resp.StatusCode; e != a {
		t.Errorf("%q expected %v, got %v", url, e, a)
	}
}
