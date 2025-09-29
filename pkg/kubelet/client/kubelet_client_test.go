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
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strconv"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/server/egressselector"
)

func kubeletTestCertHelper(valid bool) KubeletTLSConfig {
	if valid {
		return KubeletTLSConfig{
			CertFile: "testdata/mycertvalid.cer",
			KeyFile:  "testdata/mycertvalid.key",
			CAFile:   "testdata/myCA.cer",
		}
	}
	return KubeletTLSConfig{
		CertFile: "testdata/mycertinvalid.cer",
		KeyFile:  "testdata/mycertinvalid.key",
		CAFile:   "testdata/myCA.cer",
	}
}

func kubeletTestRoundTripHelper(t *testing.T, rt http.RoundTripper, addr string) {
	req, err := http.NewRequest(http.MethodGet, addr, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		dump, err := httputil.DumpResponse(resp, true)
		if err != nil {
			t.Fatal(err)
		}
		t.Fatal(string(dump))
	}
}

func TestMakeTransportInvalid(t *testing.T) {
	config := &KubeletClientConfig{
		// Invalid certificate and key path
		TLSClientConfig: kubeletTestCertHelper(false),
	}
	rt, err := MakeTransport(config)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if rt != nil {
		t.Error("rt should be nil as we provided invalid cert file")
	}
}

func TestMakeTransportValid(t *testing.T) {
	config := &KubeletClientConfig{
		Port:            1234,
		TLSClientConfig: kubeletTestCertHelper(true),
	}

	rt, err := MakeTransport(config)
	if err != nil {
		t.Errorf("Not expecting an error %#v", err)
	}
	if rt == nil {
		t.Error("rt should not be nil")
	}
}

func TestMakeTransportWithLookUp(t *testing.T) {
	dialFunc := func(ctx context.Context, network, addr string) (net.Conn, error) {
		return net.Dial(network, addr)
	}
	tests := []struct {
		name        string
		config      *KubeletClientConfig
		expectError bool
	}{
		{
			"test makeTransport with Lookup closure initialized",
			&KubeletClientConfig{
				Port:            1234,
				TLSClientConfig: kubeletTestCertHelper(true),
				Lookup: func(_ egressselector.NetworkContext) (utilnet.DialFunc, error) {
					return dialFunc, nil
				},
			},
			false,
		},
		{
			"test makeTransport with Lookup closure returning error",
			&KubeletClientConfig{
				Port:            1234,
				TLSClientConfig: kubeletTestCertHelper(true),
				Lookup: func(_ egressselector.NetworkContext) (utilnet.DialFunc, error) {
					return nil, errors.New("mock error")
				},
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			rt, err := MakeInsecureTransport(tt.config)
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none: Lookup func is invalid")
				}
				return
			}
			if rt == nil {
				t.Fatalf("rt should not be nil")
			}
			testServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.WriteHeader(http.StatusOK)
			}))
			defer testServer.Close()
			kubeletTestRoundTripHelper(t, rt, testServer.URL)
		})
	}
}

func TestMakeInsecureTransport(t *testing.T) {
	testServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	testURL, err := url.Parse(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, portStr, err := net.SplitHostPort(testURL.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.ParseUint(portStr, 10, 32)
	if err != nil {
		t.Fatal(err)
	}

	config := &KubeletClientConfig{
		Port:            uint(port),
		TLSClientConfig: kubeletTestCertHelper(true),
	}

	rt, err := MakeInsecureTransport(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if rt == nil {
		t.Error("rt should not be nil")
	}
	kubeletTestRoundTripHelper(t, rt, testServer.URL)
}

// Mock NodeGetter for testing
type mockNodeGetter struct {
	node *v1.Node
	err  error
}

func (m *mockNodeGetter) Get(ctx context.Context, name string, options metav1.GetOptions) (*v1.Node, error) {
	return m.node, m.err
}

func TestNewNodeConnectionInfoGetter(t *testing.T) {
	tests := []struct {
		name        string
		nodes       NodeGetter
		config      KubeletClientConfig
		expectError bool
	}{
		{
			name:  "valid config",
			nodes: &mockNodeGetter{},
			config: KubeletClientConfig{
				Port:                  10250,
				PreferredAddressTypes: []string{"InternalIP", "ExternalIP"},
				TLSClientConfig:       kubeletTestCertHelper(true),
			},
			expectError: false,
		},
		{
			name:  "invalid cert file",
			nodes: &mockNodeGetter{},
			config: KubeletClientConfig{
				Port:                  10250,
				PreferredAddressTypes: []string{"InternalIP"},
				TLSClientConfig:       kubeletTestCertHelper(false),
			},
			expectError: true,
		},
		{
			name:  "empty preferred address types",
			nodes: &mockNodeGetter{},
			config: KubeletClientConfig{
				Port:                  10250,
				PreferredAddressTypes: []string{},
				TLSClientConfig:       kubeletTestCertHelper(true),
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			getter, err := NewNodeConnectionInfoGetter(tt.nodes, tt.config)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if getter == nil {
				t.Fatalf("expected non-nil getter")
			}

			// Verify the created getter has correct properties
			ncig, ok := getter.(*NodeConnectionInfoGetter)
			if !ok {
				t.Fatalf("expected *NodeConnectionInfoGetter, got %T", getter)
			}

			if ncig.nodes != tt.nodes {
				t.Errorf("expected nodes to be set correctly")
			}

			if ncig.scheme != "https" {
				t.Errorf("expected scheme to be 'https', got %s", ncig.scheme)
			}

			if ncig.defaultPort != int(tt.config.Port) {
				t.Errorf("expected defaultPort to be %d, got %d", tt.config.Port, ncig.defaultPort)
			}

			if len(ncig.preferredAddressTypes) != len(tt.config.PreferredAddressTypes) {
				t.Errorf("expected %d preferred address types, got %d",
					len(tt.config.PreferredAddressTypes), len(ncig.preferredAddressTypes))
			}
		})
	}
}

func TestGetConnectionInfo(t *testing.T) {
	tests := []struct {
		name         string
		nodeGetter   NodeGetter
		nodeName     types.NodeName
		expectedHost string
		expectedPort string
		expectError  bool
	}{
		{
			name: "valid node with kubelet endpoint port",
			nodeGetter: &mockNodeGetter{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
					Status: v1.NodeStatus{
						Addresses: []v1.NodeAddress{
							{Type: v1.NodeInternalIP, Address: "192.168.1.10"},
							{Type: v1.NodeExternalIP, Address: "203.0.113.10"},
						},
						DaemonEndpoints: v1.NodeDaemonEndpoints{
							KubeletEndpoint: v1.DaemonEndpoint{Port: 10250},
						},
					},
				},
			},
			nodeName:     "test-node",
			expectedHost: "203.0.113.10",
			expectedPort: "10250",
			expectError:  false,
		},
		{
			name: "valid node without kubelet endpoint port (uses default)",
			nodeGetter: &mockNodeGetter{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
					Status: v1.NodeStatus{
						Addresses: []v1.NodeAddress{
							{Type: v1.NodeInternalIP, Address: "192.168.1.10"},
						},
						DaemonEndpoints: v1.NodeDaemonEndpoints{
							KubeletEndpoint: v1.DaemonEndpoint{Port: 0},
						},
					},
				},
			},
			nodeName:     "test-node",
			expectedHost: "192.168.1.10",
			expectedPort: "10250", // default port
			expectError:  false,
		},
		{
			name: "node with external IP preferred",
			nodeGetter: &mockNodeGetter{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
					Status: v1.NodeStatus{
						Addresses: []v1.NodeAddress{
							{Type: v1.NodeInternalIP, Address: "192.168.1.10"},
							{Type: v1.NodeExternalIP, Address: "203.0.113.10"},
						},
						DaemonEndpoints: v1.NodeDaemonEndpoints{
							KubeletEndpoint: v1.DaemonEndpoint{Port: 10250},
						},
					},
				},
			},
			nodeName:     "test-node",
			expectedHost: "203.0.113.10", // External IP should be preferred when available
			expectedPort: "10250",
			expectError:  false,
		},
		{
			name: "node not found",
			nodeGetter: &mockNodeGetter{
				err: errors.New("node not found"),
			},
			nodeName:    "nonexistent-node",
			expectError: true,
		},
		{
			name: "node with no addresses",
			nodeGetter: &mockNodeGetter{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
					Status: v1.NodeStatus{
						Addresses: []v1.NodeAddress{},
						DaemonEndpoints: v1.NodeDaemonEndpoints{
							KubeletEndpoint: v1.DaemonEndpoint{Port: 10250},
						},
					},
				},
			},
			nodeName:    "test-node",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			config := KubeletClientConfig{
				Port:                  10250,
				PreferredAddressTypes: []string{"ExternalIP", "InternalIP"},
				TLSClientConfig:       kubeletTestCertHelper(true),
			}
			getter, err := NewNodeConnectionInfoGetter(tt.nodeGetter, config)
			if err != nil {
				t.Fatalf("failed to create NodeConnectionInfoGetter: %v", err)
			}

			tCtx := context.Background()
			connInfo, err := getter.GetConnectionInfo(tCtx, tt.nodeName)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if connInfo == nil {
				t.Fatalf("expected non-nil connection info")
			}

			if connInfo.Scheme != "https" {
				t.Errorf("expected scheme 'https', got %s", connInfo.Scheme)
			}

			if connInfo.Hostname != tt.expectedHost {
				t.Errorf("expected hostname %s, got %s", tt.expectedHost, connInfo.Hostname)
			}

			if connInfo.Port != tt.expectedPort {
				t.Errorf("expected port %s, got %s", tt.expectedPort, connInfo.Port)
			}

			if connInfo.Transport == nil {
				t.Errorf("expected non-nil Transport")
			}

			if connInfo.InsecureSkipTLSVerifyTransport == nil {
				t.Errorf("expected non-nil InsecureSkipTLSVerifyTransport")
			}
		})
	}
}
