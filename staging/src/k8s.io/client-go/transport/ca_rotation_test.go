/*
Copyright The Kubernetes Authors.

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

package transport

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	clientgofeaturegate "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/tools/metrics"
	"k8s.io/client-go/util/cert"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	// Use the same rootCACert as transport_test.go
	testCACert1 = `-----BEGIN CERTIFICATE-----
MIIC4DCCAcqgAwIBAgIBATALBgkqhkiG9w0BAQswIzEhMB8GA1UEAwwYMTAuMTMu
MTI5LjEwNkAxNDIxMzU5MDU4MB4XDTE1MDExNTIxNTczN1oXDTE2MDExNTIxNTcz
OFowIzEhMB8GA1UEAwwYMTAuMTMuMTI5LjEwNkAxNDIxMzU5MDU4MIIBIjANBgkq
hkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAunDRXGwsiYWGFDlWH6kjGun+PshDGeZX
xtx9lUnL8pIRWH3wX6f13PO9sktaOWW0T0mlo6k2bMlSLlSZgG9H6og0W6gLS3vq
s4VavZ6DbXIwemZG2vbRwsvR+t4G6Nbwelm6F8RFnA1Fwt428pavmNQ/wgYzo+T1
1eS+HiN4ACnSoDSx3QRWcgBkB1g6VReofVjx63i0J+w8Q/41L9GUuLqquFxu6ZnH
60vTB55lHgFiDLjA1FkEz2dGvGh/wtnFlRvjaPC54JH2K1mPYAUXTreoeJtLJKX0
ycoiyB24+zGCniUmgIsmQWRPaOPircexCp1BOeze82BT1LCZNTVaxQIDAQABoyMw
ITAOBgNVHQ8BAf8EBAMCAKQwDwYDVR0TAQH/BAUwAwEB/zALBgkqhkiG9w0BAQsD
ggEBADMxsUuAFlsYDpF4fRCzXXwrhbtj4oQwcHpbu+rnOPHCZupiafzZpDu+rw4x
YGPnCb594bRTQn4pAu3Ac18NbLD5pV3uioAkv8oPkgr8aUhXqiv7KdDiaWm6sbAL
EHiXVBBAFvQws10HMqMoKtO8f1XDNAUkWduakR/U6yMgvOPwS7xl0eUTqyRB6zGb
K55q2dejiFWaFqB/y78txzvz6UlOZKE44g2JAVoJVM6kGaxh33q8/FmrL4kuN3ut
W+MmJCVDvd4eEqPwbp7146ZWTqpIJ8lvA6wuChtqV8lhAPka2hD/LMqY8iXNmfXD
uml0obOEy+ON91k+SWTJ3ggmF/U=
-----END CERTIFICATE-----`

	// A different CA cert for testing rotation (modified version of certData from transport_test.go)
	testCACert2 = `-----BEGIN CERTIFICATE-----
MIIC6jCCAdSgAwIBAgIBCzALBgkqhkiG9w0BAQswIzEhMB8GA1UEAwwYMTAuMTMu
MTI5LjEwNkAxNDIxMzU5MDU4MB4XDTE1MDExNTIyMDEzMVoXDTE2MDExNTIyMDEz
MlowGzEZMBcGA1UEAxMQb3BlbnNoaWZ0LWNsaWVudDCCASIwDQYJKoZIhvcNAQEB
BQADggEPADCCAQoCggEBAKtdhz0+uCLXw5cSYns9rU/XifFSpb/x24WDdrm72S/v
b9BPYsAStiP148buylr1SOuNi8sTAZmlVDDIpIVwMLff+o2rKYDicn9fjbrTxTOj
lI4pHJBH+JU3AJ0tbajupioh70jwFS0oYpwtneg2zcnE2Z4l6mhrj2okrc5Q1/X2
I2HChtIU4JYTisObtin10QKJX01CLfYXJLa8upWzKZ4/GOcHG+eAV3jXWoXidtjb
1Usw70amoTZ6mIVCkiu1QwCoa8+ycojGfZhvqMsAp1536ZcCul+Na+AbCv4zKS7F
kQQaImVrXdUiFansIoofGlw/JNuoKK6ssVpS5Ic3pgcCAwEAAaM1MDMwDgYDVR0P
AQH/BAQDAgCgMBMGA1UdJQQMMAoGCCsGAQUFBwMCMAwGA1UdEwEB/wQCMAAwCwYJ
KoZIhvcNAQELA4IBAQCKLREH7bXtXtZ+8vI6cjD7W3QikiArGqbl36bAhhWsJLp/
p/ndKz39iFNaiZ3GlwIURWOOKx3y3GA0x9m8FR+Llthf0EQ8sUjnwaknWs0Y6DQ3
jjPFZOpV3KPCFrdMJ3++E3MgwFC/Ih/N2ebFX9EcV9Vcc6oVWMdwT0fsrhu683rq
6GSR/3iVX1G/pmOiuaR0fNUaCyCfYrnI4zHBDgSfnlm3vIvN2lrsR/DQBakNL8DJ
HBgKxMGeUPoneBv+c8DMXIL0EhaFXRlBv9QW45/GiAIOuyFJ0i6hCtGZpJjq4OpQ
BRjCI+izPzFTjsxD4aORE+WOkyWFCGPWKfNejfw0
-----END CERTIFICATE-----`
)

// writeCAFile writes CA data to a temporary file
func writeCAFile(t testing.TB, caData []byte) string {
	tmpDir := t.TempDir()
	caFile := filepath.Join(tmpDir, "ca.crt")

	err := os.WriteFile(caFile, caData, 0644)
	if err != nil {
		t.Fatalf("Failed to write CA file: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Remove(caFile); err != nil {
			t.Logf("unexpected error while removing file: %s - %v", caFile, err)
		}
	})
	return caFile
}

// createTestTransport creates a test transport with TLS config
func createTestTransport(t testing.TB, caData []byte) *http.Transport {
	CAs, err := rootCertPool(caData)
	if err != nil {
		t.Fatalf("Failed to parse CA certificate")
	}
	return &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: CAs,
		},
	}
}

func TestCheckCAFileAndRotate(t *testing.T) {
	tests := []struct {
		name           string
		setupCA        []byte
		updateCA       []byte
		caFileOverride string
		expectRotation bool
	}{
		{
			name:           "no change",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte(testCACert1), // Same CA
			expectRotation: false,
		},
		{
			name:           "CA changed",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte(testCACert2), // Different CA
			expectRotation: true,
		},
		{
			name:           "CA changed to invalid",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte("panda"), // invalid CA
			expectRotation: false,
		},
		{
			name:           "file error",
			setupCA:        []byte(testCACert1),
			caFileOverride: "/nonexistent/ca.crt", // Non-existent file
			expectRotation: false,
		},
		{
			name:           "empty file content",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte{}, // Empty file
			expectRotation: false,
		},
		{
			name:           "initially empty CA file updated to valid CA",
			setupCA:        []byte{},            // Starts with empty CA
			updateCA:       []byte(testCACert1), // Populated with a valid CA
			expectRotation: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caFile := writeCAFile(t, tt.setupCA)
			if len(tt.caFileOverride) > 0 {
				caFile = tt.caFileOverride
			}

			transport := createTestTransport(t, tt.setupCA)
			setupRoots := transport.TLSClientConfig.RootCAs.Clone()

			expectedRoots := setupRoots
			if tt.expectRotation {
				var err error
				expectedRoots, err = rootCertPool(tt.updateCA)
				if err != nil {
					t.Fatal(err)
				}
			}

			clock := testingclock.NewFakeClock(time.Now())
			holder := newAtomicTransportHolder(caFile, tt.setupCA, transport)
			holder.clock = clock
			holder.transportLastChecked = clock.Now()

			if tt.updateCA != nil {
				// Update the file with new CA content
				err := os.WriteFile(caFile, tt.updateCA, 0644)
				if err != nil {
					t.Errorf("Failed to update CA data with file address: %s", caFile)
				}
			}

			clock.Step(holder.caRefreshDuration)

			// Check CA file rotation
			newTransport := holder.getTransport(t.Context())
			newRoots := newTransport.TLSClientConfig.RootCAs

			if newRoots == nil || !expectedRoots.Equal(newRoots) {
				t.Error("new roots did not match expected roots")
			}

			transportRotated := newTransport != transport
			if tt.expectRotation != transportRotated {
				t.Error("transport rotation did not match")
			}

		})
	}
}

func generateServerCertAndCA(t testing.TB) (servingCertPEM, servingKeyPEM, caCertPEM []byte) {
	t.Helper()
	certPEM, keyPEM, err := cert.GenerateSelfSignedCertKey("127.0.0.1", nil, nil)
	if err != nil {
		t.Fatalf("Failed to generate server cert: %v", err)
	}
	certs, err := cert.ParseCertsPEM(certPEM)
	if err != nil || len(certs) < 2 {
		t.Fatal("Expected cert chain with [leaf, CA]")
	}
	caPEM, err := cert.EncodeCertificates(certs[len(certs)-1])
	if err != nil {
		t.Fatalf("Failed to encode CA cert: %v", err)
	}
	return certPEM, keyPEM, caPEM
}

// TestCARotationConnectionBehavior tests end-to-end CA rotation:
// 1. Client trusts server CA via a CA file on disk
// 2. Server rotates to a new CA + serving cert
// 3. Client fails (doesn't trust new CA yet)
// 4. CA file updated, transport reloads, client reconnects
func TestCARotationConnectionBehavior(t *testing.T) {
	t.Log("Testing CA Rotation Connection Behavior")

	clientfeaturestesting.SetFeatureDuringTest(t, clientgofeaturegate.ClientsAllowCARotation, true)

	// Generate initial server v1 cert and CA
	servingCertPEM1, servingKeyPEM1, caCertPEM1 := generateServerCertAndCA(t)

	// Start server v1 (no client cert required - test focuses on server CA rotation)
	srv1 := newTestServer(t, servingCertPEM1, servingKeyPEM1)
	srv1.StartTLS()
	defer srv1.Close()

	// Set up the client
	clientCAFile := writeCAFile(t, caCertPEM1)
	config := &Config{
		TLS: TLSConfig{
			CAFile: clientCAFile,
		},
	}

	transport, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create transport: %v", err)
	}
	transport.(*atomicTransportHolder).caRefreshDuration = 500 * time.Millisecond

	client := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second,
	}

	// Initial connection must succeed
	t.Log("Making initial request to server v1, expecting success...")
	resp, err := client.Get(srv1.URL)
	if err != nil {
		t.Fatalf("Failed to call the server v1: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal("Failed to close the response.")
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatal("Failed to call the server successfully.")
	}

	t.Log("Initial connection successful.")

	// Rotate: new CA, new server cert, same address
	t.Log("Stopping server v1 and starting server v2 with new CA...")
	srv1Addr := srv1.Listener.Addr().String()
	srv1.Close()

	servingCertPEM2, servingKeyPEM2, caCertPEM2 := generateServerCertAndCA(t)
	srv2 := newTestServer(t, servingCertPEM2, servingKeyPEM2)
	l, err := net.Listen("tcp", srv1Addr)
	if err != nil {
		t.Fatalf("Failed to re-claim the same server address: %v", err)
	}

	srv2.Listener = l
	srv2.StartTLS()
	defer srv2.Close()

	// Must fail - client still trusts old CA
	t.Log("Making request to server v2, expecting failure...")
	_, err = client.Get(srv2.URL)
	if err == nil {
		t.Fatal("The request should fail.")
	}
	t.Log("Request failed as expected.")

	// Update CA file to trust new CA
	t.Log("Updating client CA file on disk to trust new CA...")
	if err := os.WriteFile(clientCAFile, caCertPEM2, 0644); err != nil {
		t.Fatalf("Failed to update CA file: %v", err)
	}

	// Poll until transport reloads
	t.Log("Polling server v2 until the client's transport reloads the new CA...")
	var lastPollErr error
	ctx, ctxCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer ctxCancel()

	pollErr := wait.PollUntilContextCancel(ctx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		resp, err := client.Get(srv2.URL)
		if err != nil {
			lastPollErr = err
			t.Log("Client failed to connect before the root CAs are updated, will retry...")
			return false, nil // Error is expected, continue polling
		}
		if err := resp.Body.Close(); err != nil {
			t.Fatal("Failed to close the response.")
		}
		if resp.StatusCode == http.StatusOK {
			return true, nil // Success! Stop polling.
		}
		return false, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	})
	if pollErr != nil {
		t.Fatalf("Client failed to reconnect after CA rotation. Last error: %v. Test error: %v", lastPollErr, pollErr)
	}

	t.Log("Success! Client reconnected after CA was refreshed.")
}

func TestCARotationConnectionBehavior_Disabled(t *testing.T) {
	t.Log("Testing CA Rotation Connection Behavior (Feature Disabled)")

	clientfeaturestesting.SetFeatureDuringTest(t, clientgofeaturegate.ClientsAllowCARotation, false)

	// Generate initial server v1 cert and CA
	servingCertPEM1, servingKeyPEM1, caCertPEM1 := generateServerCertAndCA(t)

	// Start server v1 (no client cert required - test focuses on server CA rotation)
	srv1 := newTestServer(t, servingCertPEM1, servingKeyPEM1)
	srv1.StartTLS()
	defer srv1.Close()

	// Set up the client
	clientCAFile := writeCAFile(t, caCertPEM1)
	config := &Config{
		TLS: TLSConfig{
			CAFile: clientCAFile,
		},
	}

	transport, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create transport: %v", err)
	}
	if _, ok := transport.(*atomicTransportHolder); ok {
		t.Fatal("Expected plain transport when the feature gate is disabled, got atomicTransportHolder")
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second,
	}

	// Initial connection must succeed
	t.Log("Making initial request to server v1, expecting success...")
	resp, err := client.Get(srv1.URL)
	if err != nil {
		t.Fatalf("Failed to call the server v1: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal("Failed to close the response.")
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatal("Failed to call the server successfully.")
	}

	t.Log("Initial connection successful.")

	// Rotate: new CA, new server cert, same address
	t.Log("Stopping server v1 and starting server v2 with new CA...")
	srv1Addr := srv1.Listener.Addr().String()
	srv1.Close()

	servingCertPEM2, servingKeyPEM2, caCertPEM2 := generateServerCertAndCA(t)
	srv2 := newTestServer(t, servingCertPEM2, servingKeyPEM2)
	l, err := net.Listen("tcp", srv1Addr)
	if err != nil {
		t.Fatalf("Failed to re-claim the same server address: %v", err)
	}

	srv2.Listener = l
	srv2.StartTLS()
	defer srv2.Close()

	// Must fail - client still trusts old CA
	t.Log("Making request to server v2, expecting failure...")
	_, err = client.Get(srv2.URL)
	if err == nil {
		t.Fatal("The request should fail.")
	}
	t.Log("Request failed as expected.")

	// Update CA file to trust new CA
	t.Log("Updating client CA file on disk to trust new CA...")
	if err := os.WriteFile(clientCAFile, caCertPEM2, 0644); err != nil {
		t.Fatalf("Failed to update CA file: %v", err)
	}

	// Poll to ensure transport DOES NOT reload
	t.Log("Polling server v2 to verify the client DOES NOT reconnect...")
	var lastPollErr error
	ctx, ctxCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer ctxCancel()

	pollErr := wait.PollUntilContextCancel(ctx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		resp, err := client.Get(srv2.URL)
		if err != nil {
			lastPollErr = err
			t.Log("Client failed to connect (expected because feature is disabled)...")
			return false, nil // Keep polling until the timeout is reached
		}
		if err := resp.Body.Close(); err != nil {
			t.Fatal("Failed to close the response.")
		}
		if resp.StatusCode == http.StatusOK {
			return true, nil // Success! But this means the test failed.
		}
		return false, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	})

	// If pollErr is nil, it means the connection succeeded, which is wrong for this test.
	if pollErr == nil {
		t.Fatalf("Client unexpectedly reconnected after CA rotation! The feature gate is disabled, so this should have failed.")
	}

	t.Logf("Success! Client permanently failed to reconnect as expected. Last error: %v", lastPollErr)
}

// testCAReloadsMetric is a fake metric recorder that records calls to Increment.
type testCAReloadsMetric struct {
	calls []caReloadCall
}

type caReloadCall struct {
	result, reason string
}

func (m *testCAReloadsMetric) Increment(result, reason string) {
	m.calls = append(m.calls, caReloadCall{result, reason})
}

// TestCARotationMetricsEmitted verifies that ca_rotation.go emits the correct
// metrics during actual CA reload operations.
func TestCARotationMetricsEmitted(t *testing.T) {
	fakeMetricRecorder := &testCAReloadsMetric{}
	origMetric := metrics.TransportCAReloads
	metrics.TransportCAReloads = fakeMetricRecorder
	t.Cleanup(func() { metrics.TransportCAReloads = origMetric })

	caData := []byte(testCACert1)
	caFile := writeCAFile(t, caData)
	transport := createTestTransport(t, caData)

	clock := testingclock.NewFakeClock(time.Now())
	holder := newAtomicTransportHolder(caFile, caData, transport)
	holder.clock = clock
	holder.transportLastChecked = clock.Now()

	tests := []struct {
		name       string
		setup      func() error
		wantResult string
		wantReason string
	}{
		{
			name:       "unchanged CA",
			setup:      func() error { return nil },
			wantResult: "success",
			wantReason: "unchanged",
		},
		{
			name: "updated CA",
			setup: func() error {
				return os.WriteFile(caFile, []byte(testCACert2), 0644)
			},
			wantResult: "success",
			wantReason: "updated",
		},
		{
			name: "empty file",
			setup: func() error {
				return os.WriteFile(caFile, []byte{}, 0644)
			},
			wantResult: "failure",
			wantReason: "empty",
		},
		{
			name: "invalid CA data",
			setup: func() error {
				return os.WriteFile(caFile, []byte("not-a-cert"), 0644)
			},
			wantResult: "failure",
			wantReason: "ca_parse_error",
		},
		{
			name: "read error",
			setup: func() error {
				return os.Remove(caFile)
			},
			wantResult: "failure",
			wantReason: "read_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeMetricRecorder.calls = nil
			err := tt.setup()
			if err != nil {
				t.Fatalf("Setup failed: %v", err)
			}
			clock.Step(holder.caRefreshDuration)
			holder.getTransport(context.Background())

			if len(fakeMetricRecorder.calls) != 1 {
				t.Fatalf("Expected 1 metric call, got %d", len(fakeMetricRecorder.calls))
			}
			if fakeMetricRecorder.calls[0].result != tt.wantResult || fakeMetricRecorder.calls[0].reason != tt.wantReason {
				t.Errorf("Got metric(%s, %s), want (%s, %s)",
					fakeMetricRecorder.calls[0].result, fakeMetricRecorder.calls[0].reason, tt.wantResult, tt.wantReason)
			}
		})
	}
}

// helper to create a simple, non-blocking test server with a given certificate.
func newTestServer(t *testing.T, certPEM, keyPEM []byte) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := fmt.Fprint(w, "ok"); err != nil {
			t.Fatal("Failed to write to the response.")
		}
	}))
	// Configure server TLS
	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("Failed to create server cert: %v", err)
	}

	server.TLS = &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
	}
	return server
}
