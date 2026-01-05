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
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/klog/v2"
)

func generateTestCA(t *testing.T, name string) (certPEM, keyPEM []byte, caCert *x509.Certificate, caKey *rsa.PrivateKey) {
	t.Helper()

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Failed to generate private key: %v", err)
	}

	template := x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: name},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create certificate: %v", err)
	}

	caCert, err = x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("Failed to parse CA certificate: %v", err)
	}

	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(caKey)})
	return certPEM, keyPEM, caCert, caKey
}

func generateServerCert(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey, serverName string) *x509.Certificate {
	t.Helper()

	serverKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Failed to generate server key: %v", err)
	}

	template := x509.Certificate{
		SerialNumber:          big.NewInt(2),
		Subject:               pkix.Name{CommonName: serverName},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{serverName},
	}

	certDER, err := x509.CreateCertificate(rand.Reader, &template, caCert, &serverKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create server certificate: %v", err)
	}

	serverCert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("Failed to parse server certificate: %v", err)
	}

	return serverCert
}

func TestDynamicCALoader_LoadCA(t *testing.T) {
	tmpDir := t.TempDir()
	caFile := filepath.Join(tmpDir, "ca.crt")

	ca1, _, _, _ := generateTestCA(t, "CA1")
	if err := os.WriteFile(caFile, ca1, 0644); err != nil {
		t.Fatalf("Failed to write CA file: %v", err)
	}

	initialPool := x509.NewCertPool()
	if !initialPool.AppendCertsFromPEM(ca1) {
		t.Fatal("Failed to parse initial CA")
	}

	loader := caRotatingDialer(klog.Background(), caFile, "test-server", initialPool, ca1,
		func(ctx context.Context, network, address string) (net.Conn, error) { return nil, nil })

	if pool := loader.CertPool(); pool == nil {
		t.Fatal("Expected non-nil cert pool")
	}

	if err := loader.loadCA(); err != nil {
		t.Fatalf("loadCA failed: %v", err)
	}

	ca2, _, _, _ := generateTestCA(t, "CA2")
	if err := os.WriteFile(caFile, ca2, 0644); err != nil {
		t.Fatalf("Failed to write updated CA file: %v", err)
	}

	if err := loader.loadCA(); err != nil {
		t.Fatalf("loadCA failed after update: %v", err)
	}

	loader.caMtx.RLock()
	defer loader.caMtx.RUnlock()
	if bytes.Equal(loader.caData, ca1) {
		t.Error("Expected CA data to be updated")
	}
	if !bytes.Equal(loader.caData, ca2) {
		t.Error("Expected CA data to match new CA")
	}
}

func TestDynamicCALoader_VerifyConnection(t *testing.T) {
	tmpDir := t.TempDir()
	caFile := filepath.Join(tmpDir, "ca.crt")

	caPEM, _, caCert, caKey := generateTestCA(t, "TestCA")
	if err := os.WriteFile(caFile, caPEM, 0644); err != nil {
		t.Fatalf("Failed to write CA file: %v", err)
	}

	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		t.Fatal("Failed to parse CA")
	}

	loader := caRotatingDialer(klog.Background(), caFile, "test-server", pool, caPEM,
		func(ctx context.Context, network, address string) (net.Conn, error) { return nil, nil })

	// Should fail with no peer certificates
	err := loader.VerifyConnection(tls.ConnectionState{
		PeerCertificates: nil,
		ServerName:       "test-server",
	})
	if err == nil {
		t.Error("Expected error when no peer certificates")
	}

	// Generate a proper server certificate signed by the CA
	serverCert := generateServerCert(t, caCert, caKey, "test-server")

	// Should succeed with valid server certificate
	err = loader.VerifyConnection(tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{serverCert, caCert},
		ServerName:       "test-server",
	})
	if err != nil {
		t.Errorf("Expected verification to succeed: %v", err)
	}
}

func TestTLSConfigKey_WithReloadCAFile(t *testing.T) {
	config1 := &Config{
		TLS: TLSConfig{
			CAFile:       "/path/to/ca.crt",
			CAData:       []byte("cert-data-1"),
			ReloadCAFile: true,
		},
	}
	config2 := &Config{
		TLS: TLSConfig{
			CAFile:       "/path/to/ca.crt",
			CAData:       []byte("cert-data-2"),
			ReloadCAFile: true,
		},
	}
	config3 := &Config{
		TLS: TLSConfig{
			CAFile:       "/path/to/other-ca.crt",
			CAData:       []byte("cert-data-1"),
			ReloadCAFile: true,
		},
	}

	key1, canCache1, err := tlsConfigKey(config1)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !canCache1 {
		t.Error("Expected canCache=true")
	}

	key2, canCache2, err := tlsConfigKey(config2)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !canCache2 {
		t.Error("Expected canCache=true")
	}

	key3, canCache3, err := tlsConfigKey(config3)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !canCache3 {
		t.Error("Expected canCache=true")
	}

	if key1 != key2 {
		t.Errorf("Expected same cache key for same CAFile with ReloadCAFile=true, got:\n\t%s\n\t%s", key1, key2)
	}
	if key1 == key3 {
		t.Errorf("Expected different cache key for different CAFile, got:\n\t%s\n\t%s", key1, key3)
	}
	if key1.caFile != "/path/to/ca.crt" {
		t.Errorf("Expected caFile to be set, got: %s", key1.caFile)
	}
	if key1.caData != "" {
		t.Errorf("Expected caData to be empty when ReloadCAFile=true, got: %s", key1.caData)
	}
}

func TestTLSConfigKey_WithoutReloadCAFile(t *testing.T) {
	config1 := &Config{
		TLS: TLSConfig{
			CAData:       []byte("cert-data-1"),
			ReloadCAFile: false,
		},
	}
	config2 := &Config{
		TLS: TLSConfig{
			CAData:       []byte("cert-data-2"),
			ReloadCAFile: false,
		},
	}

	key1, _, err := tlsConfigKey(config1)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	key2, _, err := tlsConfigKey(config2)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if key1 == key2 {
		t.Error("Expected different cache keys for different CAData")
	}
	if key1.caData != "cert-data-1" {
		t.Errorf("Expected caData to be set, got: %s", key1.caData)
	}
	if key1.caFile != "" {
		t.Errorf("Expected caFile to be empty when ReloadCAFile=false, got: %s", key1.caFile)
	}
}
