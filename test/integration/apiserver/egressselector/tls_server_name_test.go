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
	"crypto"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/wait"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	kubeapiserverapptesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	utilsoidc "k8s.io/kubernetes/test/utils/oidc"
)

func generateTestCerts(t *testing.T, tempDir, serverName string) (caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath string) {
	t.Helper()

	caKey, err := testutils.NewPrivateKey()
	require.NoError(t, err, "Failed to generate CA key")
	caCert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "Test CA"}, caKey)
	require.NoError(t, err, "Failed to create CA certificate")
	caCertPath = filepath.Join(tempDir, "ca.crt")
	require.NoError(t, os.WriteFile(caCertPath, testutils.EncodeCertPEM(caCert), 0600), "Failed to write CA cert")

	serverKey, err := testutils.NewPrivateKey()
	require.NoError(t, err, "Failed to generate server key")
	serverCert, err := testutils.NewSignedCert(&certutil.Config{
		CommonName: serverName,
		AltNames:   certutil.AltNames{DNSNames: []string{serverName}},
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}, serverKey, caCert, caKey)
	require.NoError(t, err, "Failed to create server certificate")
	serverCertPath, serverKeyPath = writeCertAndKey(t, tempDir, "server", serverCert, serverKey)

	clientKey, err := testutils.NewPrivateKey()
	require.NoError(t, err, "Failed to generate client key")
	clientCert, err := testutils.NewSignedCert(&certutil.Config{
		CommonName: "test-client",
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}, clientKey, caCert, caKey)
	require.NoError(t, err, "Failed to create client certificate")
	clientCertPath, clientKeyPath = writeCertAndKey(t, tempDir, "client", clientCert, clientKey)

	return caCertPath, serverCertPath, serverKeyPath, clientCertPath, clientKeyPath
}

func writeCertAndKey(t *testing.T, dir, name string, cert *x509.Certificate, key crypto.Signer) (certPath, keyPath string) {
	t.Helper()
	certPath = filepath.Join(dir, name+".crt")
	keyPath = filepath.Join(dir, name+".key")
	require.NoError(t, os.WriteFile(certPath, testutils.EncodeCertPEM(cert), 0600))
	keyPEM, err := keyutil.MarshalPrivateKeyToPEM(key)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(keyPath, keyPEM, 0600))
	return certPath, keyPath
}

// runTLSEgressProxy runs an HTTP CONNECT proxy with TLS.
func runTLSEgressProxy(t *testing.T, serverCertPath, serverKeyPath, caCertPath string, called *atomic.Bool, observedSNI *atomic.Pointer[string]) string {
	t.Helper()

	serverCert, err := tls.LoadX509KeyPair(serverCertPath, serverKeyPath)
	require.NoError(t, err, "Failed to load server cert")

	clientCAs, err := certutil.NewPool(caCertPath)
	require.NoError(t, err, "Failed to load CA cert pool")

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    clientCAs,
		GetConfigForClient: func(hello *tls.ClientHelloInfo) (*tls.Config, error) {
			observedSNI.Store(&hello.ServerName)
			return nil, nil
		},
	}

	listener, err := tls.Listen("tcp", "127.0.0.1:0", tlsConfig)
	require.NoError(t, err, "Failed to start TLS listener")

	proxyAddr := listener.Addr().String()

	server := &http.Server{Handler: utilsoidc.NewHTTPConnectProxyHandler(t, called)}

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

	return proxyAddr
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
		expectedSNI       string
		description       string
	}{
		{
			name:              "matching tlsServerName succeeds TLS handshake",
			tlsServerName:     proxyHostname,
			expectProxyCalled: true,
			expectedSNI:       proxyHostname,
			description:       "SNI override matches cert SANs, connection succeeds",
		},
		{
			name:              "mismatched tlsServerName fails TLS handshake",
			tlsServerName:     "wrong-hostname.example.com",
			expectProxyCalled: false,
			expectedSNI:       "wrong-hostname.example.com",
			description:       "SNI override doesn't match cert SANs, connection fails",
		},
		{
			name:              "empty tlsServerName uses dial address and fails",
			tlsServerName:     "",
			expectProxyCalled: false,
			expectedSNI:       "",
			description:       "Without tlsServerName, uses dial address 127.0.0.1 which fails cert validation",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var proxyCalled atomic.Bool
			var observedSNI atomic.Pointer[string]

			proxyAddr := runTLSEgressProxy(t, serverCertPath, serverKeyPath, caCertPath, &proxyCalled, &observedSNI)
			t.Logf("TLS egress proxy ready at %s (cert issued for: %s)", proxyAddr, proxyHostname)

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

			const authenticationConfigWithEgress = `
apiVersion: apiserver.config.k8s.io/v1
kind: AuthenticationConfiguration
jwt:
- issuer:
    url: https://panda.snorlax
    audiences:
    - foo
    egressSelectorType: cluster
  claimMappings:
    username:
      expression: "'test-' + claims.sub"
`

			server := kubeapiserverapptesting.StartTestServerOrDie(
				t,
				kubeapiserverapptesting.NewDefaultTestServerOptions(),
				[]string{
					fmt.Sprintf("--egress-selector-config-file=%s", utilsoidc.WriteTempFile(t, egressConfig)),
					fmt.Sprintf("--authentication-config=%s", utilsoidc.WriteTempFile(t, authenticationConfigWithEgress)),
				},
				framework.SharedEtcd(),
			)
			t.Cleanup(server.TearDownFn)

			var sni string
			err := wait.PollUntilContextTimeout(t.Context(), time.Second, wait.ForeverTestTimeout, true,
				func(ctx context.Context) (done bool, err error) {
					if ptr := observedSNI.Load(); ptr != nil {
						sni = *ptr
						return true, nil
					}
					return false, nil
				},
			)
			require.NoError(t, err, "SNI not observed")
			require.Equal(t, tc.expectedSNI, sni, "SNI mismatch")
			require.Equal(t, tc.expectProxyCalled, proxyCalled.Load(), "proxy called mismatch")
		})
	}
}
