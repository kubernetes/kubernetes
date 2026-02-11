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
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiserverapptesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	oidctest "k8s.io/kubernetes/test/integration/apiserver/oidc"
	"k8s.io/kubernetes/test/integration/framework"
	utilsoidc "k8s.io/kubernetes/test/utils/oidc"
)

func generateTestCerts(t *testing.T, tempDir, serverName string) (caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath string) {
	t.Helper()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err, "Failed to generate CA key")

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
	require.NoError(t, err, "Failed to create CA certificate")

	caCert, err := x509.ParseCertificate(caCertDER)
	require.NoError(t, err, "Failed to parse CA certificate")

	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err, "Failed to generate server key")

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: serverName},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{serverName},
	}

	serverCertDER, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverKey.PublicKey, caKey)
	require.NoError(t, err, "Failed to create server certificate")

	clientKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err, "Failed to generate client key")

	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(3),
		Subject:      pkix.Name{CommonName: "test-client"},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	clientCertDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, caCert, &clientKey.PublicKey, caKey)
	require.NoError(t, err, "Failed to create client certificate")

	caCertPath = filepath.Join(tempDir, "ca.crt")
	caCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCertDER})
	require.NoError(t, os.WriteFile(caCertPath, caCertPEM, 0600), "Failed to write CA cert")

	serverCertPath = filepath.Join(tempDir, "server.crt")
	serverKeyPath = filepath.Join(tempDir, "server.key")
	serverCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverCertDER})
	require.NoError(t, os.WriteFile(serverCertPath, serverCertPEM, 0600), "Failed to write server cert")

	serverKeyBytes, err := x509.MarshalECPrivateKey(serverKey)
	require.NoError(t, err, "Failed to marshal server key")
	serverKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: serverKeyBytes})
	require.NoError(t, os.WriteFile(serverKeyPath, serverKeyPEM, 0600), "Failed to write server key")

	clientCertPath = filepath.Join(tempDir, "client.crt")
	clientKeyPath = filepath.Join(tempDir, "client.key")
	clientCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: clientCertDER})
	require.NoError(t, os.WriteFile(clientCertPath, clientCertPEM, 0600), "Failed to write client cert")

	clientKeyBytes, err := x509.MarshalECPrivateKey(clientKey)
	require.NoError(t, err, "Failed to marshal client key")
	clientKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: clientKeyBytes})
	require.NoError(t, os.WriteFile(clientKeyPath, clientKeyPEM, 0600), "Failed to write client key")

	return caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath
}

// runTLSEgressProxy runs an HTTP CONNECT proxy with TLS.
func runTLSEgressProxy(t *testing.T, serverCertPath, serverKeyPath, caCertPath string, called *atomic.Bool, ready chan<- struct{}) (string, error) {
	t.Helper()

	serverCert, err := tls.LoadX509KeyPair(serverCertPath, serverKeyPath)
	if err != nil {
		return "", fmt.Errorf("failed to load server cert: %w", err)
	}

	caCertPEM, err := os.ReadFile(caCertPath)
	if err != nil {
		return "", fmt.Errorf("failed to read CA cert: %w", err)
	}
	clientCAs := x509.NewCertPool()
	clientCAs.AppendCertsFromPEM(caCertPEM)

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    clientCAs,
	}

	listener, err := tls.Listen("tcp", "127.0.0.1:0", tlsConfig)
	if err != nil {
		return "", fmt.Errorf("failed to start TLS listener: %w", err)
	}

	proxyAddr := listener.Addr().String()

	httpConnectProxy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ready" {
			t.Log("TLS egress proxy ready")
			w.WriteHeader(http.StatusOK)
			return
		}

		called.Store(true)
		t.Logf("TLS egress proxy received connection: %s %s", r.Method, r.Host)

		if r.Method != http.MethodConnect {
			http.Error(w, "this proxy only supports CONNECT passthrough", http.StatusMethodNotAllowed)
			return
		}

		backendConn, err := (&net.Dialer{}).DialContext(r.Context(), "tcp", r.Host)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() {
			if err := backendConn.Close(); err != nil {
				t.Logf("Failed to close backend connection: %v", err)
			}
		}()

		hijacker, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijacking not supported", http.StatusInternalServerError)
			return
		}

		clientConn, _, err := hijacker.Hijack()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() {
			if err := clientConn.Close(); err != nil {
				t.Logf("Failed to close client connection: %v", err)
			}
		}()

		_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		if err != nil {
			t.Errorf("unexpected established error: %v", err)
			return
		}

		writerComplete := make(chan struct{})
		readerComplete := make(chan struct{})

		go func() {
			_, err := io.Copy(backendConn, clientConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("writer error: %v", err)
			}
			close(writerComplete)
		}()

		go func() {
			_, err := io.Copy(clientConn, backendConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("reader error: %v", err)
			}
			close(readerComplete)
		}()

		select {
		case <-writerComplete:
		case <-readerComplete:
		}
	})

	server := &http.Server{Handler: httpConnectProxy}

	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			t.Logf("TLS egress proxy serve error: %v", err)
		}
	}()

	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Logf("Failed to close server: %v", err)
		}
	})

	close(ready)

	return proxyAddr, nil
}

// TestTLSServerName verifies that the tlsServerName field in the egress selector configuration
// correctly overrides SNI for TLS certificate validation when connecting through a proxy.
func TestTLSServerName(t *testing.T) {
	tempDir := t.TempDir()

	proxyHostname := "egress-proxy.example.com"

	caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath := generateTestCerts(t, tempDir, proxyHostname)

	testCases := []struct {
		name              string
		tlsServerName     string
		expectProxyCalled bool
		description       string
	}{
		{
			name:              "matching tlsServerName succeeds TLS handshake",
			tlsServerName:     proxyHostname,
			expectProxyCalled: true,
			description:       "SNI override matches cert SANs, connection succeeds",
		},
		{
			name:              "mismatched tlsServerName fails TLS handshake",
			tlsServerName:     "wrong-hostname.example.com",
			expectProxyCalled: false,
			description:       "SNI override doesn't match cert SANs, connection fails",
		},
		{
			name:              "empty tlsServerName uses dial address and fails",
			tlsServerName:     "",
			expectProxyCalled: false,
			description:       "Without tlsServerName, uses dial address 127.0.0.1 which fails cert validation",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var proxyCalled atomic.Bool

			ready := make(chan struct{})
			proxyAddr, err := runTLSEgressProxy(t, serverCertPath, serverKeyPath, caCertPath, &proxyCalled, ready)
			require.NoError(t, err, "Failed to start TLS egress proxy")

			select {
			case <-ready:
				t.Logf("TLS egress proxy ready at %s (cert issued for: %s)", proxyAddr, proxyHostname)
			case <-time.After(10 * time.Second):
				t.Fatal("timeout waiting for TLS egress proxy to start")
			}

			caCertContent, _, caFilePath, caKeyFilePath := oidctest.GenerateCert(t)
			signingPrivateKey, publicKey := oidctest.RSAGenerateKey(t)
			oidcServer := utilsoidc.BuildAndRunTestServer(t, caFilePath, caKeyFilePath, "")

			egressConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: cluster
  connection:
    proxyProtocol: HTTPConnect
    transport:
      tcp:
        url: https://%s
        tlsConfig:
          caBundle: %s
          clientCert: %s
          clientKey: %s
          tlsServerName: %s
`, proxyAddr, caCertPath, clientCertPath, clientKeyPath, tc.tlsServerName)

			authenticationConfig := fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthenticationConfiguration
jwt:
- issuer:
    url: %s
    audiences:
    - foo
    certificateAuthority: |
        %s
    egressSelectorType: cluster
  claimMappings:
    username:
      expression: "'test-' + claims.sub"
`, oidcServer.URL(), oidctest.IndentCertificateAuthority(string(caCertContent)))

			customFlags := []string{
				fmt.Sprintf("--egress-selector-config-file=%s", oidctest.WriteTempFile(t, egressConfig)),
				fmt.Sprintf("--authentication-config=%s", oidctest.WriteTempFile(t, authenticationConfig)),
				"--authorization-mode=RBAC",
			}

			server := kubeapiserverapptesting.StartTestServerOrDie(
				t,
				kubeapiserverapptesting.NewDefaultTestServerOptions(),
				customFlags,
				framework.SharedEtcd(),
			)
			t.Cleanup(server.TearDownFn)

			oidcServer.JwksHandler().EXPECT().KeySet().RunAndReturn(utilsoidc.DefaultJwksHandlerBehavior(t, publicKey)).Maybe()

			idTokenLifetime := time.Second * 1200
			oidcServer.TokenHandler().EXPECT().Token().RunAndReturn(utilsoidc.TokenHandlerBehaviorReturningPredefinedJWT(
				t,
				signingPrivateKey,
				map[string]interface{}{
					"iss": oidcServer.URL(),
					"sub": "test-user",
					"aud": "foo",
					"exp": time.Now().Add(idTokenLifetime).Unix(),
				},
				"access-token",
				"refresh-token",
			)).Maybe()

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			client := kubernetes.NewForConfigOrDie(rest.CopyConfig(server.ClientConfig))

			// Attempt to list namespaces to trigger egress proxy usage
			// The error is expected in some test cases (e.g., when TLS handshake fails)
			_, _ = client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})

			time.Sleep(100 * time.Millisecond)

			if tc.expectProxyCalled != proxyCalled.Load() {
				t.Errorf("expected proxy called=%v, got=%v", tc.expectProxyCalled, proxyCalled.Load())
			}
		})
	}
}
