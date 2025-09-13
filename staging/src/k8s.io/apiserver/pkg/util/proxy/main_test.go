/*
Copyright 2024 The Kubernetes Authors.

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

package proxy

import (
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"

	"k8s.io/apiserver/pkg/util/proxy/metrics"
)

// Shared test servers for the proxy package.
var (
	// spdyServer is the upstream SPDY server that both the StreamTranslatorHandler
	// and TunnelingHandler connect to. It is shared across all tests in this package.
	spdyServer *httptest.Server
	// spdyServerMux is the router for the spdyServer. Tests register their specific handlers here.
	spdyServerMux *http.ServeMux
	spdyServerURL *url.URL

	// streamTranslatorServer is the server that exposes the StreamTranslatorHandler.
	// Test clients connect to this server. It is shared across all translator tests.
	streamTranslatorServer *httptest.Server
	// streamTranslatorServerMux is the router for the streamTranslatorServer.
	// Tests register their specific StreamTranslatorHandler configurations here.
	streamTranslatorServerMux *http.ServeMux
	streamTranslatorServerURL *url.URL

	// tunnelingServer is the server that exposes the TunnelingHandler.
	// Test clients connect to this server. It is shared across all tunneling tests.
	tunnelingServer *httptest.Server
	// tunnelingServerMux is the router for the tunnelingServer.
	// Tests register their specific TunnelingHandler configurations here.
	tunnelingServerMux *http.ServeMux
	tunnelingServerURL *url.URL
)

// TestMain sets up the shared SPDY, StreamTranslator, and Tunneling servers for the entire test suite.
// This avoids the overhead of creating new servers for each test and eliminates race conditions
// related to server startup and shutdown.
func TestMain(m *testing.M) {
	metrics.Register()

	// Upstream SPDY server setup (used by both translator and tunneling tests)
	spdyServerMux = http.NewServeMux()
	spdyServer = httptest.NewServer(spdyServerMux)
	var err error
	spdyServerURL, err = url.Parse(spdyServer.URL)
	if err != nil {
		log.Fatalf("Failed to parse SPDY server URL: %v", err)
	}

	// StreamTranslator server setup
	streamTranslatorServerMux = http.NewServeMux()
	streamTranslatorServer = httptest.NewServer(streamTranslatorServerMux)
	streamTranslatorServerURL, err = url.Parse(streamTranslatorServer.URL)
	if err != nil {
		log.Fatalf("Failed to parse StreamTranslator server URL: %v", err)
	}

	// Tunneling server setup
	tunnelingServerMux = http.NewServeMux()
	tunnelingServer = httptest.NewServer(tunnelingServerMux)
	tunnelingServerURL, err = url.Parse(tunnelingServer.URL)
	if err != nil {
		log.Fatalf("Failed to parse Tunneling server URL: %v", err)
	}

	// Run tests
	exitCode := m.Run()

	// Teardown
	spdyServer.Close()
	streamTranslatorServer.Close()
	tunnelingServer.Close()

	os.Exit(exitCode)
}
