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

package egressselector

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/server/egressselector"
)

func generateTestCerts(t *testing.T, tempDir, serverName string) (caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath string) {
	t.Helper()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("Failed to generate CA key: %v", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "Test CA"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		IsCA:                  true,
		BasicConstraintsValid: true,
	}

	caCertDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create CA certificate: %v", err)
	}
	caCert, err := x509.ParseCertificate(caCertDER)
	if err != nil {
		t.Fatalf("Failed to parse CA certificate: %v", err)
	}

	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("Failed to generate server key: %v", err)
	}

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: serverName},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{serverName}, // SANS set here
	}

	serverCertDER, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create server certificate: %v", err)
	}

	clientKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("Failed to generate client key: %v", err)
	}

	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(3),
		Subject:      pkix.Name{CommonName: "test-client"},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	clientCertDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, caCert, &clientKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create client certificate: %v", err)
	}

	caCertPath = filepath.Join(tempDir, "ca.crt")
	caCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCertDER})
	if err := os.WriteFile(caCertPath, caCertPEM, 0600); err != nil {
		t.Fatalf("Failed to write CA cert: %v", err)
	}

	serverCertPath = filepath.Join(tempDir, "server.crt")
	serverKeyPath = filepath.Join(tempDir, "server.key")
	serverCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverCertDER})
	if err := os.WriteFile(serverCertPath, serverCertPEM, 0600); err != nil {
		t.Fatalf("Failed to write server cert: %v", err)
	}
	serverKeyBytes, _ := x509.MarshalECPrivateKey(serverKey)
	serverKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: serverKeyBytes})
	if err := os.WriteFile(serverKeyPath, serverKeyPEM, 0600); err != nil {
		t.Fatalf("Failed to write server key: %v", err)
	}

	clientCertPath = filepath.Join(tempDir, "client.crt")
	clientKeyPath = filepath.Join(tempDir, "client.key")
	clientCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: clientCertDER})
	if err := os.WriteFile(clientCertPath, clientCertPEM, 0600); err != nil {
		t.Fatalf("Failed to write client cert: %v", err)
	}
	clientKeyBytes, _ := x509.MarshalECPrivateKey(clientKey)
	clientKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: clientKeyBytes})
	if err := os.WriteFile(clientKeyPath, clientKeyPEM, 0600); err != nil {
		t.Fatalf("Failed to write client key: %v", err)
	}

	return caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath
}

// TestTLSServerName verifies that the tlsServerName field in EgressSelectorConfiguration
// correctly overrides SNI when the destination server (proxy address) doesn't match the certificate SANs.
func TestTLSServerName(t *testing.T) {
	tempDir := t.TempDir()
	// Certificate is issued for this hostname
	proxyHostname := "konnectivity-server.example.com"

	caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath := generateTestCerts(t, tempDir, proxyHostname)

	serverCert, err := tls.LoadX509KeyPair(serverCertPath, serverKeyPath)
	if err != nil {
		t.Fatalf("Failed to load server key pair: %v", err)
	}

	caCertPEM, err := os.ReadFile(caCertPath)
	if err != nil {
		t.Fatalf("Failed to read CA cert: %v", err)
	}
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caCertPEM)

	serverTLSConfig := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientCAs:    certPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}

	listener, err := tls.Listen("tcp", "127.0.0.1:0", serverTLSConfig)
	if err != nil {
		t.Fatalf("Failed to start TLS listener: %v", err)
	}
	defer func() { _ = listener.Close() }()

	// Without tlsServerName, the client would use the hostname from the URL (127.0.0.1)
	proxyAddr := listener.Addr().String()
	t.Logf("Proxy server listening on %s", proxyAddr)

	go runHTTPConnectProxy(t, listener)

	testCases := []struct {
		name           string
		tlsServerName  string
		expectTLSError bool
	}{
		{
			name:           "matching tlsServerName succeeds TLS handshake",
			tlsServerName:  proxyHostname,
			expectTLSError: false,
		},
		{
			name:           "mismatched tlsServerName fails TLS handshake",
			tlsServerName:  "wrong-hostname.example.com",
			expectTLSError: true,
		},
		{
			name:           "empty tlsServerName uses dial address",
			tlsServerName:  "",
			expectTLSError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "EgressSelectorConfiguration",
					APIVersion: "apiserver.config.k8s.io/v1beta1",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolHTTPConnect,
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://" + proxyAddr,
									TLSConfig: &apiserver.TLSConfig{
										CABundle:      caCertPath,
										ClientCert:    clientCertPath,
										ClientKey:     clientKeyPath,
										TLSServerName: tc.tlsServerName,
									},
								},
							},
						},
					},
				},
			}

			egressSelector, err := egressselector.NewEgressSelector(config)
			if err != nil {
				t.Fatalf("Failed to create egress selector: %v", err)
			}

			dialer, err := egressSelector.Lookup(egressselector.Cluster.AsNetworkContext())
			if err != nil {
				t.Fatalf("Failed to lookup dialer: %v", err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			conn, err := dialer(ctx, "tcp", "backend.example.com:80")

			if tc.expectTLSError {
				if err == nil {
					if conn != nil {
						_ = conn.Close()
					}
					t.Error("Expected TLS error due to server name mismatch, but dial succeeded")
				} else {
					if !strings.Contains(err.Error(), "certificate") && !strings.Contains(err.Error(), "x509") {
						t.Errorf("Expected certificate verification error, got: %v", err)
					}
					t.Logf("Got expected TLS error: %v", err)
				}
			} else {
				if err != nil {
					if strings.Contains(err.Error(), "certificate") || strings.Contains(err.Error(), "x509") || strings.Contains(err.Error(), "tls:") {
						t.Errorf("TLS handshake failed unexpectedly: %v", err)
					} else {
						t.Logf("TLS succeeded, got expected non-TLS error (backend unreachable): %v", err)
					}
				} else {
					t.Log("Dial succeeded completely")
					if conn != nil {
						_ = conn.Close()
					}
				}
			}
		})
	}
}

func runHTTPConnectProxy(t *testing.T, listener net.Listener) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go handleProxyConnection(t, conn)
	}
}

func handleProxyConnection(t *testing.T, conn net.Conn) {
	defer func() { _ = conn.Close() }()

	buf := make([]byte, 4096)
	n, err := conn.Read(buf)
	if err != nil {
		t.Logf("Error reading from connection: %v", err)
		return
	}

	request := string(buf[:n])
	if !strings.HasPrefix(request, "CONNECT ") {
		t.Logf("Not a CONNECT request: %s", request)
		return
	}

	lines := strings.Split(request, "\r\n")
	parts := strings.Split(lines[0], " ")
	if len(parts) < 2 {
		t.Logf("Invalid CONNECT request: %s", request)
		return
	}
	targetAddr := parts[1]
	t.Logf("CONNECT request for: %s", targetAddr)

	backendConn, err := net.Dial("tcp", targetAddr)
	if err != nil {
		_, _ = fmt.Fprintf(conn, "HTTP/1.1 502 Bad Gateway\r\n\r\n")
		t.Logf("Backend connection failed (expected in test): %v", err)
		return
	}
	defer func() { _ = backendConn.Close() }()

	_, err = conn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	if err != nil {
		t.Logf("Error sending response: %v", err)
		return
	}

	// Use channels to synchronize bidirectional copy
	done := make(chan struct{}, 2)
	go func() {
		_, _ = io.Copy(backendConn, conn)
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(conn, backendConn)
		done <- struct{}{}
	}()
	// Wait for either direction to complete, then close both connections
	<-done
	_ = conn.Close()
	_ = backendConn.Close()
	<-done // Wait for the other goroutine to finish
}

// TestTLSServerNameWithBackend verifies end-to-end data transmission through a TLS proxy
// when tlsServerName is used to override SNI for certificate validation.
func TestTLSServerNameWithBackend(t *testing.T) {
	tempDir := t.TempDir()
	proxyHostname := "konnectivity-server.example.com"

	caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath := generateTestCerts(t, tempDir, proxyHostname)

	backendListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to start backend listener: %v", err)
	}
	backendAddr := backendListener.Addr().String()
	t.Logf("Backend server listening on %s", backendAddr)

	var backendConns []net.Conn
	var backendConnsMu sync.Mutex
	var backendWg sync.WaitGroup

	// Cleanup all backend connections
	cleanupBackend := func() {
		_ = backendListener.Close()
		backendConnsMu.Lock()
		for _, c := range backendConns {
			_ = c.Close()
		}
		backendConnsMu.Unlock()
		backendWg.Wait()
	}
	defer cleanupBackend()

	go func() {
		for {
			conn, err := backendListener.Accept()
			if err != nil {
				return
			}
			backendConnsMu.Lock()
			backendConns = append(backendConns, conn)
			backendConnsMu.Unlock()

			backendWg.Add(1)
			// Create echo server
			go func(c net.Conn) {
				defer backendWg.Done()
				defer func() { _ = c.Close() }()
				buf := make([]byte, 4096)
				for {
					n, err := c.Read(buf)
					if err != nil {
						return
					}
					if _, err := c.Write(buf[:n]); err != nil {
						return
					}
				}
			}(conn)
		}
	}()

	serverCert, err := tls.LoadX509KeyPair(serverCertPath, serverKeyPath)
	if err != nil {
		t.Fatalf("Failed to load server key pair: %v", err)
	}

	caCertPEM, err := os.ReadFile(caCertPath)
	if err != nil {
		t.Fatalf("Failed to read CA cert: %v", err)
	}
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caCertPEM)

	serverTLSConfig := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientCAs:    certPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}

	proxyListener, err := tls.Listen("tcp", "127.0.0.1:0", serverTLSConfig)
	if err != nil {
		t.Fatalf("Failed to start TLS listener: %v", err)
	}
	defer func() { _ = proxyListener.Close() }()

	proxyAddr := proxyListener.Addr().String()
	t.Logf("Proxy server listening on %s", proxyAddr)

	go runHTTPConnectProxy(t, proxyListener)

	config := &apiserver.EgressSelectorConfiguration{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EgressSelectorConfiguration",
			APIVersion: "apiserver.config.k8s.io/v1beta1",
		},
		EgressSelections: []apiserver.EgressSelection{
			{
				Name: "cluster",
				Connection: apiserver.Connection{
					ProxyProtocol: apiserver.ProtocolHTTPConnect,
					Transport: &apiserver.Transport{
						TCP: &apiserver.TCPTransport{
							URL: "https://" + proxyAddr,
							TLSConfig: &apiserver.TLSConfig{
								CABundle:      caCertPath,
								ClientCert:    clientCertPath,
								ClientKey:     clientKeyPath,
								TLSServerName: proxyHostname, // Use the hostname that matches the cert
							},
						},
					},
				},
			},
		},
	}

	egressSelector, err := egressselector.NewEgressSelector(config)
	if err != nil {
		t.Fatalf("Failed to create egress selector: %v", err)
	}

	dialer, err := egressSelector.Lookup(egressselector.Cluster.AsNetworkContext())
	if err != nil {
		t.Fatalf("Failed to lookup dialer: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := dialer(ctx, "tcp", backendAddr)
	if err != nil {
		t.Fatalf("Failed to dial through proxy: %v", err)
	}
	defer func() { _ = conn.Close() }()

	testData := "hello through proxy with tlsServerName"
	_, err = conn.Write([]byte(testData))
	if err != nil {
		t.Fatalf("Failed to write to connection: %v", err)
	}

	response := make([]byte, len(testData))
	_, err = io.ReadFull(conn, response)
	if err != nil {
		t.Fatalf("Failed to read from connection: %v", err)
	}

	if string(response) != testData {
		t.Errorf("Expected echo response %q, got %q", testData, string(response))
	}

	t.Log("Successfully connected through proxy with tlsServerName and verified echo response")
}
