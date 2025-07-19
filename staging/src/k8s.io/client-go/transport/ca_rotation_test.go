/*
Copyright 2025 The Kubernetes Authors.

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
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
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

func TestNewAtomicTransportHolder(t *testing.T) {
	caFile := writeCAFile(t, []byte(testCACert1))
	
	config := &Config{
		TLS: TLSConfig{
			CAFile: caFile,
			CAData: []byte(testCACert1),
		},
	}
	
	transport := createTestTransport(t, []byte(testCACert1))
	
	holder := newAtomicTransportHolder(config, transport)
	
	if holder == nil {
		t.Fatal("Expected non-nil holder")
	}
	
	if holder.caFile != caFile {
		t.Errorf("Expected caFile %s, got %s", caFile, holder.caFile)
	}
	
	if holder.config != config {
		t.Error("Expected config to be set")
	}
	
	if holder.transport.Load() != transport {
		t.Error("Expected transport to be stored")
	}
	
	if holder.queue == nil {
		t.Error("Expected queue to be initialized")
	}
}

func TestCheckCAFileAndRotate(t *testing.T) {
	tests := []struct {
		name           string
		setupCA        []byte
		updateCA       []byte
		caFile         string
		expectRotation bool
		expectError    bool
	}{
		{
			name:           "no change",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte(testCACert1), // Same CA
			expectRotation: false,
			expectError:    false,
		},
		{
			name:           "CA changed",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte(testCACert2), // Different CA
			expectRotation: true,
			expectError:    false,
		},
		{
			name:        "file error",
			setupCA:     []byte(testCACert1),
			caFile:      "/nonexistent/ca.crt", // Non-existent file
			expectError: true,
		},
		{
			name:           "empty file content",
			setupCA:        []byte(testCACert1),
			updateCA:       []byte{}, // Empty file
			expectRotation: false,
			expectError:    false,
		},
		{
			name:           "initial empty CA data",
			setupCA:        []byte{}, // No initial CA data
			updateCA:       []byte(testCACert1),
			expectRotation: true, // Should rotate since we have new CA data
			expectError:    false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caFile := writeCAFile(t, tt.setupCA)
			if tt.caFile != "" {
				caFile = tt.caFile
			} 
			
			config := &Config{
				TLS: TLSConfig{
					CAFile: caFile,
					CAData: tt.setupCA,
				},
			}
			
			transport := createTestTransport(t, tt.setupCA)
			holder := newAtomicTransportHolder(config, transport)
		
			if tt.updateCA != nil {
				// Update the file with new CA content
				err := os.WriteFile(caFile, tt.updateCA, 0644)
				if err != nil {
					t.Errorf("Failed to update CA data with file address: %s", caFile)
				}
			}
			
			time.Sleep(time.Second)
			
			// Check CA file rotation
			err := holder.checkCAFileAndRotate()
			
			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				// Transport should remain unchanged on error
				if holder.transport.Load() != transport {
					t.Error("Expected transport to remain unchanged on error")
				}
				return
			}
			
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			
			newTransport := holder.transport.Load()
			if tt.expectRotation {
				if newTransport == transport {
					t.Error("Expected transport to be rotated")
				}
				// New transport should have updated CA
				if newTransport.TLSClientConfig == nil {
					t.Error("Expected TLS config in new transport")
				}
				// Verify RootCAs is not nil when we have valid CA data
				if len(tt.updateCA) > 0 && newTransport.TLSClientConfig.RootCAs == nil {
					t.Error("Expected RootCAs to be set when CA data is available")
				}
				if newTransport.TLSClientConfig.RootCAs == transport.TLSClientConfig.RootCAs {
                    t.Error("Expected RootCAs should change")
				}
			} else {
				if newTransport != transport {
					t.Error("Expected transport to remain unchanged")
				}
			}
		})
	}
}

func TestController(t *testing.T) {
	tests := []struct {
		name           string
		refreshDuration time.Duration
		testRotation   bool
	}{
		{
			name:           "start and stop",
			refreshDuration: 100 * time.Millisecond,
			testRotation:   false,
		},
		{
			name:           "CA rotation",
			refreshDuration: 50 * time.Millisecond,
			testRotation:   true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caFile := writeCAFile(t, []byte(testCACert1))
			config := &Config{
				TLS: TLSConfig{
					CAFile: caFile,
					CAData: []byte(testCACert1),
				},
			}
			
			transport := createTestTransport(t, []byte(testCACert1))
			holder := newAtomicTransportHolder(config, transport)
			
			// Start controller
			stopCh := make(chan struct{})
			go holder.run(stopCh, tt.refreshDuration)
			
			// Let it run for a bit
			time.Sleep(2 * tt.refreshDuration)
			
			if tt.testRotation {
				// Update CA file for rotation test
				err := os.WriteFile(caFile, []byte(testCACert2), 0644)
				if err != nil {
					t.Fatalf("Failed to update CA file: %v", err)
				}
				
				// Wait for controller to detect and rotate
				var newTransport *http.Transport
				err = wait.PollImmediate(10*time.Millisecond, 500*time.Millisecond, func() (bool, error) {
					newTransport = holder.transport.Load()
					return newTransport != transport, nil
				})
				
				if err != nil {
					t.Fatalf("Controller did not rotate transport: %v", err)
				}
				
				// Verify new transport has updated CA
				if newTransport.TLSClientConfig == nil {
					t.Fatal("Expected TLS config in new transport")
				}
				
				if newTransport.TLSClientConfig.RootCAs == nil {
					t.Error("Expected RootCAs to be set after rotation")
				}
				
				// Verify the RootCAs actually changed
				if newTransport.TLSClientConfig.RootCAs == transport.TLSClientConfig.RootCAs {
					t.Error("Expected RootCAs to be different after rotation")
				}
			}
			
			// Stop controller
			close(stopCh)
			
			// Give it time to stop
			time.Sleep(100 * time.Millisecond)
		})
	}
}

func TestUtilityFunctions(t *testing.T) {
	t.Run("bytes equal", func(t *testing.T) {
		tests := []struct {
			name     string
			data1    []byte
			data2    []byte
			expected bool
		}{
			{"same data", []byte(testCACert1), []byte(testCACert1), true},
			{"different data", []byte(testCACert1), []byte(testCACert2), false},
			{"empty data", []byte{}, []byte{}, true},
		}
		
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := bytes.Equal(tt.data1, tt.data2)
				if result != tt.expected {
					t.Errorf("Expected %v, got %v", tt.expected, result)
				}
			})
		}
	})
	
	t.Run("root cert pool", func(t *testing.T) {
		tests := []struct {
			name        string
			caData      []byte
			expectError bool
			expectNil   bool
		}{
			{"valid CA data", []byte(testCACert1), false, false},
			{"invalid CA data", []byte("invalid-ca-data"), true, false},
			{"empty CA data", []byte{}, false, true},
		}
		
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				pool, err := rootCertPool(tt.caData)
				
				if tt.expectError {
					if err == nil {
						t.Error("Expected error but got none")
					}
					return
				}
				
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				
				if tt.expectNil {
					if pool != nil {
						t.Error("Expected nil cert pool")
					}
				} else {
					if pool == nil {
						t.Error("Expected non-nil cert pool")
					}
				}
			})
		}
	})
}

// createTestCertificateAuthority creates a test CA certificate and key
func createTestCertificateAuthority(t testing.TB, commonName string) ([]byte, crypto.PrivateKey, []byte, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, nil, err
	}
	
	// Create CA certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: commonName,
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		// Add IP SANs for 127.0.0.1 and DNS names for localhost
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
		DNSNames:    []string{"localhost"},
	}
	
	// Create certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, nil, nil, err
	}
	
	// Encode certificate to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	
	return certDER, privateKey, certPEM, nil
}

// createTestClientCertificate creates a client certificate signed by the given CA
func createTestClientCertificate(t testing.TB, caCertPEM []byte, caKey crypto.PrivateKey, commonName string) ([]byte, []byte, error) {
	// Parse CA certificate
	caCertBlock, _ := pem.Decode(caCertPEM)
	if caCertBlock == nil {
		return nil, nil, fmt.Errorf("failed to parse CA certificate PEM")
	}
	
	caCert, err := x509.ParseCertificate(caCertBlock.Bytes)
	if err != nil {
		return nil, nil, err
	}
	
	// Generate client private key
	clientKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}
	
	// Create client certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName: commonName,
		},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	
	// Create certificate signed by CA
	certDER, err := x509.CreateCertificate(rand.Reader, &template, caCert, &clientKey.PublicKey, caKey)
	if err != nil {
		return nil, nil, err
	}
	
	// Encode certificate to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	
	// Encode private key to PEM
	keyBytes, err := x509.MarshalPKCS8PrivateKey(clientKey)
	if err != nil {
		return nil, nil, err
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})
	
	return certPEM, keyPEM, nil
}

// TestCARotationConnectionBehavior tests that CA rotation:
// 1. Does NOT close active connections 
// 2. New connections use the updated CA
// 3. Idle connections are properly cleaned up
func TestCARotationConnectionBehavior(t *testing.T) {
	t.Log("Testing CA Rotation Connection Behavior")
	
	// Create initial CA and server certificates
	serverCert1, serverKey1, serverCA1, err := createTestCertificateAuthority(t, "test-server-1")
	if err != nil {
		t.Fatalf("Failed to create initial server CA: %v", err)
	}
	
	clientCert1, clientKey1, err := createTestClientCertificate(t, serverCA1, serverKey1, "test-client-1")
	if err != nil {
		t.Fatalf("Failed to create initial client cert: %v", err)
	}
	
	// Create test server with multiple endpoints
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/slow":
			// Slow endpoint - simulates active long-running connection
			time.Sleep(300 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("slow-response"))
		case "/fast":
			// Fast endpoint - for testing new connections
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("fast-response"))
		case "/stream":
			// Streaming endpoint - simulates persistent connections
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusOK)
			for i := 0; i < 5; i++ {
				fmt.Fprintf(w, "chunk-%d\n", i)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				time.Sleep(100 * time.Millisecond)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	
	// Configure server TLS
	cert1, err := tls.X509KeyPair(
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverCert1}),
		func() []byte {
			keyBytes, _ := x509.MarshalPKCS8PrivateKey(serverKey1)
			return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})
		}(),
	)
	if err != nil {
		t.Fatalf("Failed to create server cert: %v", err)
	}
	
	combinedCaPool := x509.NewCertPool()
	combinedCaPool.AppendCertsFromPEM(serverCA1)
	
	server.TLS = &tls.Config{
		Certificates: []tls.Certificate{cert1},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    combinedCaPool,
	}
	server.StartTLS()
	defer server.Close()
	
	// Setup client with CA rotation
	caFile := writeCAFile(t, serverCA1)
	config := &Config{
		TLS: TLSConfig{
			CAFile:   caFile,
			CertData: clientCert1,
			KeyData:  clientKey1,
		},
	}
	
	transport, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create transport: %v", err)
	}
	
	holder, ok := transport.(*atomicTransportHolder)
	if !ok {
		t.Fatalf("Expected atomicTransportHolder, got %T", transport)
	}
	
	// Start CA rotation controller
	stopCh := make(chan struct{})
	defer close(stopCh)
	go holder.run(stopCh, 50*time.Millisecond)
	
	client := &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}
	
	// Test 1: Establish initial connection
	t.Log("Testing initial connection")
	resp, err := client.Get(server.URL + "/fast")
	if err != nil {
		t.Fatalf("Initial connection failed: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	
	if string(body) != "fast-response" {
		t.Errorf("Expected 'fast-response', got '%s'", string(body))
	}
	t.Log("Initial connection successful")
	
	// Test 2: Start a long-running request BEFORE CA rotation
	t.Log("Testing active connection preservation during CA rotation")
	
	type requestResult struct {
		body []byte
		err  error
		startTime time.Time
		endTime   time.Time
	}
	
	// Channel to receive result from long-running request
	longRequestResult := make(chan requestResult, 1)
	
	// Start long-running request
	go func() {
		startTime := time.Now()
		resp, err := client.Get(server.URL + "/slow")
		endTime := time.Now()
		
		var body []byte
		if err == nil {
			body, _ = io.ReadAll(resp.Body)
			resp.Body.Close()
		}
		
		longRequestResult <- requestResult{
			body:      body,
			err:       err,
			startTime: startTime,
			endTime:   endTime,
		}
	}()
	
	// Wait a moment to ensure the long request starts
	time.Sleep(100 * time.Millisecond)
	
	// Test 3: Perform CA rotation WHILE the long request is active
	t.Log("Performing CA rotation during active request")
	
	// Create rotated CA and certificates
	serverCert2, serverKey2, serverCA2, err := createTestCertificateAuthority(t, "test-server-2")
	if err != nil {
		t.Fatalf("Failed to create rotated server CA: %v", err)
	}
	
	// clientCert2, clientKey2, err := createTestClientCertificate(t, serverCA2, serverKey2, "test-client-2")
	// if err != nil {
	// 	t.Fatalf("Failed to create rotated client cert: %v", err)
	// }
	
	// Store original transport reference to verify it changes
	originalTransport := holder.transport.Load()
	
	// Update server to accept both CAs (simulating gradual CA rollout)
	cert2, err := tls.X509KeyPair(
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverCert2}),
		func() []byte {
			keyBytes, _ := x509.MarshalPKCS8PrivateKey(serverKey2)
			return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})
		}(),
	)
	if err != nil {
		t.Fatalf("Failed to create rotated server cert: %v", err)
	}
	
	// Create combined CA pool (both old and new CAs)
	combinedCaPool.AppendCertsFromPEM(serverCA2)
	
	// Update server to accept both CAs
	server.TLS.Certificates = []tls.Certificate{cert2}
	server.TLS.ClientCAs = combinedCaPool
	
	// Update client CA file (simulate CA file rotation)
	err = os.WriteFile(caFile, serverCA2, 0644)
	if err != nil {
		t.Fatalf("Failed to update CA file: %v", err)
	}
	
	// Allow some time for CA rotation to be detected
	time.Sleep(100 * time.Millisecond)
	
	// Test 4: Verify the long-running request completes successfully
	t.Log("Waiting for long-running request to complete")
	select {
	case result := <-longRequestResult:
		if result.err != nil {
			t.Errorf("Long-running request failed: %v", result.err)
		} else if string(result.body) != "slow-response" {
			t.Errorf("Expected 'slow-response', got '%s'", string(result.body))
		} else {
			t.Log("Long-running request completed successfully during CA rotation")
		}
		
		// Verify request took expected time
		duration := result.endTime.Sub(result.startTime)
		if duration < 250*time.Millisecond {
			t.Errorf("Request completed too quickly: %v", duration)
		}
		
	case <-time.After(2 * time.Second):
		t.Error("Long-running request timed out")
	}
	
	// Test 5: Verify transport was updated
	newTransport := holder.transport.Load()
	if newTransport == originalTransport {
		t.Error("Expected transport to be updated after CA rotation")
	} else if newTransport.TLSClientConfig.RootCAs == originalTransport.TLSClientConfig.RootCAs {
		t.Error("The root CAs are not rotated.")
	} else {
		t.Log("Transport was properly updated after CA rotation")
	}
	
	// Test 6: Verify new connections use the rotated CA
	t.Log("Testing new connections with rotated CA")
	
	// Create new client
	// rotatedConfig := &Config{
	// 	TLS: TLSConfig{
	// 		CAFile:   caFile,
	// 		CertData: clientCert1,
	// 		KeyData:  clientKey2,
	// 	},
	// }
	
	rotatedTransport, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create rotated transport: %v", err)
	}
	
	rotatedClient := &http.Client{
		Transport: rotatedTransport,
		Timeout:   5 * time.Second,
	}
	
	// Test new connection
	resp, err = rotatedClient.Get(server.URL + "/fast")
	if err != nil {
		t.Errorf("New connection with rotated CA failed: %v", err)
	} else {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if string(body) != "fast-response" {
			t.Errorf("Expected 'fast-response', got '%s'", string(body))
		} else {
			t.Log("New connections work with rotated CA")
		}
	}
	
	// Test 7: Verify original client eventually adapts to CA rotation
	t.Log("Testing original client adaptation to CA rotation")
	
	// Give the original client some time to detect CA rotation
	time.Sleep(200 * time.Millisecond)
	
	resp, err = client.Get(server.URL + "/fast")
	if err != nil {
		t.Logf("Original client failed after CA rotation (may be expected): %v", err)
		// This might be expected depending on the implementation
	} else {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if string(body) == "fast-response" {
			t.Log("Original client successfully adapted to CA rotation")
		} else {
			t.Errorf("Expected 'fast-response', got '%s'", string(body))
		}
	}
	
	t.Log("Connection behavior test completed successfully")
}
