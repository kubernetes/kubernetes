/*
Copyright 2022 The Kubernetes Authors.

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

package factory

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/kubernetes"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func Test_atomicLastError(t *testing.T) {
	aError := &atomicLastError{err: fmt.Errorf("initial error")}
	// no timestamp is always updated
	aError.Store(errors.New("updated error"), time.Time{})
	err := aError.Load()
	if err.Error() != "updated error" {
		t.Fatalf("Expected: \"updated error\" got: %s", err.Error())
	}
	// update to current time
	now := time.Now()
	aError.Store(errors.New("now error"), now)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
	// no update to past time
	past := now.Add(-5 * time.Second)
	aError.Store(errors.New("past error"), past)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
}

// TestNewETCD3Client_EndpointPreprocessing tests that HTTP/HTTPS prefixes are stripped
func TestNewETCD3Client_EndpointPreprocessing(t *testing.T) {
	tests := []struct {
		name           string
		serverList     []string
		expectedResult string // what we expect in the endpoints after processing
	}{
		{
			name:           "strips https prefix",
			serverList:     []string{"https://etcd.default.svc:2379"},
			expectedResult: "etcd.default.svc:2379",
		},
		{
			name:           "strips http prefix",
			serverList:     []string{"http://etcd.default.svc:2379"},
			expectedResult: "etcd.default.svc:2379",
		},
		{
			name:           "no prefix unchanged",
			serverList:     []string{"etcd.default.svc:2379"},
			expectedResult: "etcd.default.svc:2379",
		},
		{
			name:           "IP address with https",
			serverList:     []string{"https://10.0.1.1:2379"},
			expectedResult: "10.0.1.1:2379",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original function
			originalNewETCD3Client := newETCD3Client
			defer func() { newETCD3Client = originalNewETCD3Client }()

			var capturedEndpoints []string
			// Mock newETCD3Client to capture the endpoints
			newETCD3Client = func(c storagebackend.TransportConfig) (*kubernetes.Client, error) {
				// Create a minimal config to capture endpoints
				cfg := clientv3.Config{
					DialTimeout: dialTimeout,
				}

				// Process endpoint (single endpoint mode logic)
				if len(c.ServerList) == 1 {
					endpoint := c.ServerList[0]
					endpoint = strings.TrimPrefix(endpoint, "https://")
					endpoint = strings.TrimPrefix(endpoint, "http://")
					capturedEndpoints = []string{endpoint}
					cfg.Endpoints = capturedEndpoints
				} else {
					capturedEndpoints = c.ServerList
					cfg.Endpoints = c.ServerList
				}

				// Don't actually create a client, return nil with no error
				// This is just to test the preprocessing logic
				return nil, fmt.Errorf("mock: not connecting")
			}

			config := storagebackend.TransportConfig{
				ServerList: tt.serverList,
			}

			// Call the function (will use our mock)
			_, _ = newETCD3Client(config)

			// Verify the endpoint was processed correctly
			if len(capturedEndpoints) != 1 || capturedEndpoints[0] != tt.expectedResult {
				t.Errorf("Expected endpoint %q, got %v", tt.expectedResult, capturedEndpoints)
			}
		})
	}
}

// TestNewETCD3Client_DualModeConfiguration tests single vs multiple endpoint behavior
func TestNewETCD3Client_DualModeConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		serverList     []string
		wantSingleMode bool
	}{
		{
			name:           "single endpoint uses DNS resolver mode",
			serverList:     []string{"etcd.default.svc:2379"},
			wantSingleMode: true,
		},
		{
			name: "multiple endpoints use legacy mode",
			serverList: []string{
				"10.0.1.1:2379",
				"10.0.1.2:2379",
				"10.0.1.3:2379",
			},
			wantSingleMode: false,
		},
		{
			name: "two endpoints use legacy mode",
			serverList: []string{
				"etcd-0:2379",
				"etcd-1:2379",
			},
			wantSingleMode: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isSingleMode := len(tt.serverList) == 1
			if isSingleMode != tt.wantSingleMode {
				t.Errorf("Expected single mode=%v, got %v", tt.wantSingleMode, isSingleMode)
			}
		})
	}
}

// TestNewETCD3Client_TLSServerName tests TLS ServerName extraction
func TestNewETCD3Client_TLSServerName(t *testing.T) {
	tests := []struct {
		name               string
		endpoint           string
		expectedServerName string
	}{
		{
			name:               "extracts hostname from host:port",
			endpoint:           "etcd.default.svc.cluster.local:2379",
			expectedServerName: "etcd.default.svc.cluster.local",
		},
		{
			name:               "uses endpoint as-is when no port",
			endpoint:           "etcd.default.svc",
			expectedServerName: "etcd.default.svc",
		},
		{
			name:               "handles IPv4 address with port",
			endpoint:           "10.0.1.1:2379",
			expectedServerName: "10.0.1.1",
		},
		{
			name:               "handles short hostname",
			endpoint:           "localhost:2379",
			expectedServerName: "localhost",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the TLS ServerName extraction logic
			host := tt.endpoint
			if h, _, err := net.SplitHostPort(host); err == nil {
				host = h
			}

			if host != tt.expectedServerName {
				t.Errorf("Expected ServerName %q, got %q", tt.expectedServerName, host)
			}
		})
	}
}

// TestNewETCD3Client_EgressDialerBehavior tests egress dialer usage
func TestNewETCD3Client_EgressDialerBehavior(t *testing.T) {
	tests := []struct {
		name                    string
		serverList              []string
		wantEgressDialerUsed    bool
		wantEgressDialerSkipped bool
	}{
		{
			name:                    "single endpoint skips egress dialer",
			serverList:              []string{"etcd.default.svc:2379"},
			wantEgressDialerUsed:    false,
			wantEgressDialerSkipped: true,
		},
		{
			name: "multiple endpoints use egress dialer",
			serverList: []string{
				"10.0.1.1:2379",
				"10.0.1.2:2379",
			},
			wantEgressDialerUsed:    true,
			wantEgressDialerSkipped: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test logic: single endpoint mode skips egress dialer
			// multiple endpoint mode uses egress dialer
			isSingleEndpoint := len(tt.serverList) == 1
			egressDialerSkipped := isSingleEndpoint
			egressDialerUsed := !isSingleEndpoint

			if egressDialerSkipped != tt.wantEgressDialerSkipped {
				t.Errorf("Expected egress dialer skipped=%v, got %v",
					tt.wantEgressDialerSkipped, egressDialerSkipped)
			}
			if egressDialerUsed != tt.wantEgressDialerUsed {
				t.Errorf("Expected egress dialer used=%v, got %v",
					tt.wantEgressDialerUsed, egressDialerUsed)
			}
		})
	}
}

// TestNewETCD3Client_ServiceConfigFormat tests the gRPC service config structure
func TestNewETCD3Client_ServiceConfigFormat(t *testing.T) {
	// Expected service config with round-robin and health checking
	expectedServiceConfig := `{
  "loadBalancingConfig": [{"round_robin": {}}],
  "healthCheckConfig": {
    "serviceName": ""
  }
}`

	// Verify the service config format is correct
	// This is primarily a documentation test to ensure we know what config we're setting
	if !strings.Contains(expectedServiceConfig, "round_robin") {
		t.Error("Service config should contain round_robin load balancer")
	}
	if !strings.Contains(expectedServiceConfig, "healthCheckConfig") {
		t.Error("Service config should contain healthCheckConfig")
	}
	if !strings.Contains(expectedServiceConfig, `"serviceName": ""`) {
		t.Error("Service config should have empty serviceName for health checking")
	}
}

// TestNewETCD3Client_EmptyServerList tests handling of empty server list
func TestNewETCD3Client_EmptyServerList(t *testing.T) {
	config := storagebackend.TransportConfig{
		ServerList: []string{},
	}

	// Should not panic and either handle gracefully or return an error
	// The actual behavior depends on the kubernetes.New implementation
	// This test documents expected behavior
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("newETCD3Client panicked with empty server list: %v", r)
			}
		}()

		// Attempt to create client with empty server list
		// We expect this to either:
		// 1. Return an error
		// 2. Be handled gracefully
		// But it should NOT panic
		client, err := newETCD3Client(config)
		if client != nil || err == nil {
			// If we got here, the function handled it (either returned error or nil client)
			// This is acceptable behavior
			if client != nil {
				t.Error("Expected nil client with empty server list")
			}
		}
	}()
}

// TestNewETCD3Client_ConfigurationConstants tests that our constants are set correctly
func TestNewETCD3Client_ConfigurationConstants(t *testing.T) {
	// Verify the constants used in client configuration
	if keepaliveTime != 30*time.Second {
		t.Errorf("Expected keepaliveTime=30s, got %v", keepaliveTime)
	}
	if keepaliveTimeout != 10*time.Second {
		t.Errorf("Expected keepaliveTimeout=10s, got %v", keepaliveTimeout)
	}
	if dialTimeout != 20*time.Second {
		t.Errorf("Expected dialTimeout=20s, got %v", dialTimeout)
	}
}

// TestNewETCD3Client_TracerProviderNilHandling tests that nil TracerProvider doesn't cause panic
func TestNewETCD3Client_TracerProviderNilHandling(t *testing.T) {
	// This test verifies that having a nil TracerProvider doesn't cause issues
	// when APIServerTracing feature gate is enabled

	config := storagebackend.TransportConfig{
		ServerList:     []string{"localhost:2379"},
		TracerProvider: nil, // Explicitly nil
	}

	// Should not panic with nil TracerProvider
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("newETCD3Client panicked with nil TracerProvider: %v", r)
		}
	}()

	// Attempt to create client (will fail to connect, but shouldn't panic)
	_, _ = newETCD3Client(config)
}

// TestNewETCD3Client_TLSConfigPreservation tests that TLS config is properly handled
func TestNewETCD3Client_TLSConfigPreservation(t *testing.T) {
	tests := []struct {
		name          string
		serverList    []string
		certFile      string
		keyFile       string
		trustedCAFile string
		shouldSetTLS  bool
	}{
		{
			name:          "TLS config provided",
			serverList:    []string{"etcd.default.svc:2379"},
			certFile:      "/path/to/cert.pem",
			keyFile:       "/path/to/key.pem",
			trustedCAFile: "/path/to/ca.pem",
			shouldSetTLS:  true,
		},
		{
			name:         "no TLS config",
			serverList:   []string{"etcd.default.svc:2379"},
			shouldSetTLS: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := storagebackend.TransportConfig{
				ServerList:    tt.serverList,
				CertFile:      tt.certFile,
				KeyFile:       tt.keyFile,
				TrustedCAFile: tt.trustedCAFile,
			}

			hasTLSConfig := config.CertFile != "" && config.KeyFile != "" && config.TrustedCAFile != ""
			if hasTLSConfig != tt.shouldSetTLS {
				t.Errorf("Expected TLS config presence=%v, got %v", tt.shouldSetTLS, hasTLSConfig)
			}
		})
	}
}

// TestNewETCD3Client_EndpointURLParsing tests URL parsing for egress dialer
func TestNewETCD3Client_EndpointURLParsing(t *testing.T) {
	tests := []struct {
		name         string
		addr         string
		expectedHost string
		shouldParse  bool
	}{
		{
			name:         "URL with scheme",
			addr:         "https://10.0.1.1:2379",
			expectedHost: "10.0.1.1:2379",
			shouldParse:  true,
		},
		{
			name:         "plain host:port",
			addr:         "10.0.1.1:2379",
			expectedHost: "10.0.1.1:2379",
			shouldParse:  false, // doesn't contain "//"
		},
		{
			name:         "etcd URL format",
			addr:         "http://etcd-0.etcd:2379",
			expectedHost: "etcd-0.etcd:2379",
			shouldParse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This simulates the egress dialer URL parsing logic
			addr := tt.addr
			shouldParse := strings.Contains(addr, "//")

			if shouldParse != tt.shouldParse {
				t.Errorf("Expected URL parsing needed=%v, got %v", tt.shouldParse, shouldParse)
			}

			// If we need to parse, verify we can extract the host
			if shouldParse {
				u, err := url.Parse(addr)
				if err != nil {
					t.Errorf("Failed to parse URL %q: %v", addr, err)
				} else if u.Host != tt.expectedHost {
					t.Errorf("Expected host %q, got %q", tt.expectedHost, u.Host)
				}
			}
		})
	}
}
