/*
Copyright 2018 The Kubernetes Authors.

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

package certificate

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/mldsa"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	certapi "k8s.io/api/certificates/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate"
	"k8s.io/client-go/util/keyutil"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1"
	"k8s.io/kubernetes/pkg/controller/certificates/authority"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/certificate/keyalgorithm"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

func TestAddressesToHostnamesAndIPs(t *testing.T) {
	tests := []struct {
		name         string
		addresses    []v1.NodeAddress
		wantDNSNames []string
		wantIPs      []net.IP
	}{
		{
			name:         "empty",
			addresses:    nil,
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name:         "ignore empty values",
			addresses:    []v1.NodeAddress{{Type: v1.NodeHostName, Address: ""}},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "ignore invalid IPs",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2"},
				{Type: v1.NodeExternalIP, Address: "3.4"},
			},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "dedupe values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeExternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
		{
			name: "order values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname-2"},
				{Type: v1.NodeExternalDNS, Address: "hostname-1"},
				{Type: v1.NodeInternalDNS, Address: "hostname-3"},
				{Type: v1.NodeInternalIP, Address: "2.2.2.2"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "3.3.3.3"},
			},
			wantDNSNames: []string{"hostname-1", "hostname-2", "hostname-3"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1"), netutils.ParseIPSloppy("2.2.2.2"), netutils.ParseIPSloppy("3.3.3.3")},
		},
		{
			name: "handle IP and DNS hostnames",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeHostName, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDNSNames, gotIPs := addressesToHostnamesAndIPs(tt.addresses)
			if !reflect.DeepEqual(gotDNSNames, tt.wantDNSNames) {
				t.Errorf("addressesToHostnamesAndIPs() gotDNSNames = %v, want %v", gotDNSNames, tt.wantDNSNames)
			}
			if !reflect.DeepEqual(gotIPs, tt.wantIPs) {
				t.Errorf("addressesToHostnamesAndIPs() gotIPs = %v, want %v", gotIPs, tt.wantIPs)
			}
		})
	}
}

func removeThenCreate(name string, data []byte, perm os.FileMode) error {
	if err := os.Remove(name); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	return os.WriteFile(name, data, perm)
}

func createCertAndKeyFiles(certDir string) (string, string, error) {
	cert, key, err := cert.GenerateSelfSignedCertKey("k8s.io", nil, nil)
	if err != nil {
		return "", "", nil
	}

	certPath := filepath.Join(certDir, "kubelet.cert")
	keyPath := filepath.Join(certDir, "kubelet.key")
	if err := removeThenCreate(certPath, cert, os.FileMode(0644)); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(keyPath, key, os.FileMode(0600)); err != nil {
		return "", "", err
	}

	return certPath, keyPath, nil
}

// createCertAndKeyFilesUsingRename creates cert and key files under a parent dir `identity` as
// <certDir>/identity/kubelet.cert, <certDir>/identity/kubelet.key
func createCertAndKeyFilesUsingRename(certDir string) (string, string, error) {
	cert, key, err := cert.GenerateSelfSignedCertKey("k8s.io", nil, nil)
	if err != nil {
		return "", "", nil
	}

	var certKeyPathFn = func(dataDir string) (string, string, string) {
		outputDir := filepath.Join(certDir, dataDir)
		return outputDir, filepath.Join(outputDir, "kubelet.cert"), filepath.Join(outputDir, "kubelet.key")
	}

	writeDir, writeCertPath, writeKeyPath := certKeyPathFn("identity.tmp")
	if err := os.Mkdir(writeDir, 0777); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(writeCertPath, cert, os.FileMode(0644)); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(writeKeyPath, key, os.FileMode(0600)); err != nil {
		return "", "", err
	}

	targetDir, certPath, keyPath := certKeyPathFn("identity")
	if err := os.RemoveAll(targetDir); err != nil {
		if !os.IsNotExist(err) {
			return "", "", err
		}
	}
	if err := os.Rename(writeDir, targetDir); err != nil {
		return "", "", err
	}

	return certPath, keyPath, nil
}

func TestKubeletServerCertificateFromFiles(t *testing.T) {
	tCtx := ktesting.Init(t)
	// test two common ways of certificate file updates:
	// 1. delete and write the cert and key files directly
	// 2. create the cert and key files under a child dir and perform dir rename during update
	tests := []struct {
		name      string
		useRename bool
	}{
		{
			name:      "remove and create",
			useRename: false,
		},
		{
			name:      "rename cert dir",
			useRename: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			createFn := createCertAndKeyFiles
			if tt.useRename {
				createFn = createCertAndKeyFilesUsingRename
			}

			certDir := t.TempDir()
			certPath, keyPath, err := createFn(certDir)
			if err != nil {
				t.Fatalf("Unable to setup cert files: %v", err)
			}

			m, err := NewKubeletServerCertificateDynamicFileManager(certPath, keyPath)
			if err != nil {
				t.Fatalf("Unable to create certificte provider: %v", err)
			}

			m.Start()
			defer m.Stop()

			c := m.Current()
			if c == nil {
				t.Fatal("failed to provide valid certificate")
			}
			time.Sleep(100 * time.Millisecond)
			c2 := m.Current()
			if c2 == nil {
				t.Fatal("failed to provide valid certificate")
			}
			if c2 != c {
				t.Errorf("expected the same loaded certificate object when there is no cert file change, got different")
			}

			// simulate certificate files updated in the background
			if _, _, err := createFn(certDir); err != nil {
				t.Fatalf("got errors when rotating certificate files in the test: %v", err)
			}

			err = wait.PollUntilContextTimeout(tCtx,
				100*time.Millisecond, 10*time.Second, true,
				func(_ context.Context) (bool, error) {
					c3 := m.Current()
					if c3 == nil {
						return false, fmt.Errorf("expected valid certificate regardless of file changes, but got nil")
					}
					if bytes.Equal(c.Certificate[0], c3.Certificate[0]) {
						t.Logf("loaded certificate is not updated")
						return false, nil
					}
					return true, nil
				})
			if err != nil {
				t.Errorf("failed to provide the updated certificate after file changes: %v", err)
			}

			if err = os.Remove(certPath); err != nil {
				t.Errorf("could not delete file in order to perform test")
			}

			time.Sleep(1 * time.Second)
			if m.Current() == nil {
				t.Errorf("expected the manager still provides cached content when certificate file was not available")
			}
		})
	}
}

func TestNewCertificateManagerConfigGetTemplate(t *testing.T) {
	nodeName := "fake-node"
	nodeIP := netutils.ParseIPSloppy("192.168.1.1")
	tests := []struct {
		name          string
		nodeAddresses []v1.NodeAddress
		want          *x509.CertificateRequest
		featuregate   bool
	}{
		{
			name:        "node addresses or hostnames and gate enabled",
			featuregate: true,
		},
		{
			name:        "node addresses or hostnames and gate disabled",
			featuregate: false,
		},
		{
			name: "only hostnames and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames: []string{nodeName},
			},
			featuregate: true,
		},
		{
			name: "only hostnames and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
			},
			featuregate: false,
		},
		{
			name: "only IP addresses and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: true,
		},
		{
			name: "only IP addresses and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: false,
		},
		{
			name: "IP addresses and hostnames and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames:    []string{nodeName},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: true,
		},
		{
			name: "IP addresses and hostnames and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames:    []string{nodeName},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowDNSOnlyNodeCSR, tt.featuregate)
			getAddresses := func() []v1.NodeAddress {
				return tt.nodeAddresses
			}
			getTemplate := newGetTemplateFn(types.NodeName(nodeName), getAddresses)
			got := getTemplate()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Wrong certificate, got %v expected %v", got, tt.want)
				return
			}
		})
	}
}

func TestKeyAlgorithmCSRAndPersistence(t *testing.T) {
	tests := []struct {
		name      string
		algorithm *kubeletconfig.CertificateKeyAlgorithmType
		wantType  string
	}{
		{
			name:      "nil defaults to ECDSA P-256",
			algorithm: nil,
			wantType:  "*ecdsa.PrivateKey",
		},
		{
			name:      "ECDSA-P256",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmECDSAP256),
			wantType:  "*ecdsa.PrivateKey",
		},
		{
			name:      "ECDSA-P384",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmECDSAP384),
			wantType:  "*ecdsa.PrivateKey",
		},
		{
			name:      "RSA-2048",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmRSA2048),
			wantType:  "*rsa.PrivateKey",
		},
		{
			name:      "ML-DSA-65",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmMLDSA65),
			wantType:  "*mldsa.PrivateKey",
		},
	}

	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   "system:node:test-node",
			Organization: []string{"system:nodes"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			generateKey := keyalgorithm.KeyGeneratorFunc(tc.algorithm)

			// Generate the key (same path the manager takes).
			key, err := generateKey()
			if err != nil {
				t.Fatalf("GenerateKey failed: %v", err)
			}

			// Create a CSR from the key (same path as manager.generateCSR).
			csrPEM, err := cert.MakeCSRFromTemplate(key, template)
			if err != nil {
				t.Fatalf("MakeCSRFromTemplate failed: %v", err)
			}
			if len(csrPEM) == 0 {
				t.Fatal("expected non-empty CSR PEM")
			}

			// Parse the CSR and verify the public key type matches.
			block, _ := pem.Decode(csrPEM)
			if block == nil {
				t.Fatal("failed to decode CSR PEM")
			}
			cr, err := x509.ParseCertificateRequest(block.Bytes)
			if err != nil {
				t.Fatalf("failed to parse CSR: %v", err)
			}
			switch tc.wantType {
			case "*ecdsa.PrivateKey":
				if _, ok := cr.PublicKey.(*ecdsa.PublicKey); !ok {
					t.Errorf("expected ECDSA public key in CSR, got %T", cr.PublicKey)
				}
			case "*rsa.PrivateKey":
				if _, ok := cr.PublicKey.(*rsa.PublicKey); !ok {
					t.Errorf("expected RSA public key in CSR, got %T", cr.PublicKey)
				}
			case "*mldsa.PrivateKey":
				if _, ok := cr.PublicKey.(*mldsa.PublicKey); !ok {
					t.Errorf("expected ML-DSA public key in CSR, got %T", cr.PublicKey)
				}
			}

			// Marshal the key to PEM and parse it back (persistence round-trip).
			keyPEM, err := keyutil.MarshalPrivateKeyToPEM(key)
			if err != nil {
				t.Fatalf("MarshalPrivateKeyToPEM failed: %v", err)
			}
			parsedKey, err := keyutil.ParsePrivateKeyPEM(keyPEM)
			if err != nil {
				t.Fatalf("ParsePrivateKeyPEM failed: %v", err)
			}
			if reflect.TypeOf(parsedKey) != reflect.TypeOf(key) {
				t.Errorf("round-tripped key type mismatch: got %T, want %T", parsedKey, key)
			}
		})
	}
}

type rotater interface {
	RotateCerts() (bool, error)
}

func TestClientCertificateManagerMLDSA(t *testing.T) {
	certDir := t.TempDir()
	caKey, caCert, s, capturedCSR := setupTestCSRServer(t)
	defer s.Close()
	_ = caKey
	_ = caCert

	algoVal := kubeletconfig.CertificateKeyAlgorithmMLDSA65
	m, err := NewKubeletClientCertificateManager(
		ktesting.Init(t).Logger(),
		certDir,
		types.NodeName("test-node"),
		nil, nil, "", "",
		func(current *tls.Certificate) (clientset.Interface, error) {
			return clientset.NewForConfig(&restclient.Config{
				Host:          s.URL,
				ContentConfig: restclient.ContentConfig{ContentType: runtime.ContentTypeJSON},
			})
		},
		&algoVal,
	)
	if err != nil {
		t.Fatal(err)
	}

	r := m.(rotater)
	ok, err := r.RotateCerts()
	if !ok || err != nil {
		t.Fatalf("RotateCerts failed: ok=%v, err=%v", ok, err)
	}

	csrPEM := capturedCSR()
	verifyCSRPublicKeyType[*mldsa.PublicKey](t, csrPEM, "ML-DSA")
	verifyCertReloaded(t, m, certDir, csrPEM)
}

func TestServerCertificateManagerMLDSA(t *testing.T) {
	certDir := t.TempDir()
	_, _, s, capturedCSR := setupTestCSRServer(t)
	defer s.Close()

	kubeClient, err := clientset.NewForConfig(&restclient.Config{
		Host:          s.URL,
		ContentConfig: restclient.ContentConfig{ContentType: runtime.ContentTypeJSON},
	})
	if err != nil {
		t.Fatal(err)
	}

	algoVal := kubeletconfig.CertificateKeyAlgorithmMLDSA65
	kubeCfg := &kubeletconfig.KubeletConfiguration{
		ServerCertificateKeyAlgorithm: &algoVal,
	}

	m, err := NewKubeletServerCertificateManager(
		ktesting.Init(t).Logger(),
		kubeClient,
		kubeCfg,
		types.NodeName("test-node"),
		func() []v1.NodeAddress {
			return []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.0.1"}}
		},
		certDir,
	)
	if err != nil {
		t.Fatal(err)
	}

	r := m.(rotater)
	ok, err := r.RotateCerts()
	if !ok || err != nil {
		t.Fatalf("RotateCerts failed: ok=%v, err=%v", ok, err)
	}

	csrPEM := capturedCSR()
	verifyCSRPublicKeyType[*mldsa.PublicKey](t, csrPEM, "ML-DSA")
	verifyCertReloaded(t, m, certDir, csrPEM)
}

func setupTestCSRServer(t *testing.T) (*ecdsa.PrivateKey, *x509.Certificate, *httptest.Server, func() []byte) {
	t.Helper()
	serverPrivateKey, err := ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	serverCA, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "test-ca"}, serverPrivateKey)
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var csrPEM []byte
	srv := &testCSRServer{
		t:                t,
		serverPrivateKey: serverPrivateKey,
		serverCA:         serverCA,
		onCSR:            func(pem []byte) { mu.Lock(); csrPEM = pem; mu.Unlock() },
	}
	return serverPrivateKey, serverCA, httptest.NewServer(srv), func() []byte {
		mu.Lock()
		defer mu.Unlock()
		return csrPEM
	}
}

func verifyCSRPublicKeyType[T any](t *testing.T, csrPEM []byte, name string) {
	t.Helper()
	if len(csrPEM) == 0 {
		t.Fatal("no CSR was captured")
	}
	block, _ := pem.Decode(csrPEM)
	if block == nil {
		t.Fatal("failed to decode CSR PEM")
	}
	cr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		t.Fatalf("failed to parse CSR: %v", err)
	}
	if _, ok := cr.PublicKey.(T); !ok {
		t.Errorf("expected %s public key in CSR, got %T", name, cr.PublicKey)
	}
}

func verifyCertReloaded(t *testing.T, m certificate.Manager, certDir string, csrPEM []byte) {
	t.Helper()
	current := m.Current()
	if current == nil {
		t.Fatal("expected current certificate after rotation")
	}

	// Find the current PEM file on disk and reload it independently.
	matches, err := filepath.Glob(filepath.Join(certDir, "*-current.pem"))
	if err != nil {
		t.Fatalf("failed to glob for current PEM: %v", err)
	}
	if len(matches) == 0 {
		t.Fatal("no *-current.pem file found on disk")
	}

	pemData, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatalf("failed to read PEM file: %v", err)
	}

	// Parse the stored key and verify it matches the CSR's public key type.
	parsedKey, err := keyutil.ParsePrivateKeyPEM(pemData)
	if err != nil {
		t.Fatalf("failed to parse private key from stored PEM: %v", err)
	}

	// Parse the CSR to get the expected public key type.
	block, _ := pem.Decode(csrPEM)
	if block == nil {
		t.Fatal("failed to decode CSR PEM for comparison")
	}
	cr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		t.Fatalf("failed to parse CSR for comparison: %v", err)
	}

	// Verify the reloaded private key's public component matches the CSR's public key.
	switch pk := parsedKey.(type) {
	case *ecdsa.PrivateKey:
		csrPub, ok := cr.PublicKey.(*ecdsa.PublicKey)
		if !ok {
			t.Errorf("key type mismatch: stored key is ECDSA but CSR public key is %T", cr.PublicKey)
		} else if !pk.PublicKey.Equal(csrPub) {
			t.Error("stored ECDSA key does not match CSR public key")
		}
	case *rsa.PrivateKey:
		csrPub, ok := cr.PublicKey.(*rsa.PublicKey)
		if !ok {
			t.Errorf("key type mismatch: stored key is RSA but CSR public key is %T", cr.PublicKey)
		} else if !pk.PublicKey.Equal(csrPub) {
			t.Error("stored RSA key does not match CSR public key")
		}
	case *mldsa.PrivateKey:
		csrPub, ok := cr.PublicKey.(*mldsa.PublicKey)
		if !ok {
			t.Errorf("key type mismatch: stored key is ML-DSA but CSR public key is %T", cr.PublicKey)
		} else if !pk.PublicKey().Equal(csrPub) {
			t.Error("stored ML-DSA key does not match CSR public key")
		}
	default:
		t.Errorf("unexpected stored key type: %T", parsedKey)
	}
}

// testCSRServer is a minimal HTTP server that accepts CSR creation requests,
// signs them, and returns the signed certificate for watch/list.
type testCSRServer struct {
	t                *testing.T
	serverPrivateKey *ecdsa.PrivateKey
	serverCA         *x509.Certificate
	onCSR            func(csrPEM []byte)

	lock sync.Mutex
	csr  *certapi.CertificateSigningRequest
}

func (s *testCSRServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	s.lock.Lock()
	defer s.lock.Unlock()
	t := s.t

	q := req.URL.Query()
	q.Del("timeout")
	q.Del("timeoutSeconds")
	q.Del("allowWatchBookmarks")
	req.URL.RawQuery = q.Encode()

	switch {
	case req.Method == http.MethodPost && req.URL.Path == "/apis/certificates.k8s.io/v1/certificatesigningrequests":
		body, err := io.ReadAll(req.Body)
		if err != nil {
			t.Fatal(err)
		}
		csr := &certapi.CertificateSigningRequest{}
		if err := json.Unmarshal(body, csr); err != nil {
			t.Fatal(err)
		}
		if csr.Name == "" {
			csr.Name = "test-csr"
		}
		csr.UID = types.UID("1")
		csr.ResourceVersion = "1"

		if s.onCSR != nil {
			s.onCSR(csr.Spec.Request)
		}

		w.Header().Set("Content-Type", "application/json")
		data, _ := json.Marshal(csr)
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}

		csr = csr.DeepCopy()
		csr.ResourceVersion = "2"
		ca := &authority.CertificateAuthority{
			Certificate: s.serverCA,
			PrivateKey:  s.serverPrivateKey,
		}
		cr, err := capihelper.ParseCSR(csr.Spec.Request)
		if err != nil {
			t.Fatal(err)
		}
		der, err := ca.Sign(cr.Raw, authority.PermissiveSigningPolicy{
			TTL: time.Hour,
		})
		if err != nil {
			t.Fatal(err)
		}
		csr.Status.Certificate = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
		csr.Status.Conditions = []certapi.CertificateSigningRequestCondition{
			{Type: certapi.CertificateApproved},
		}
		s.csr = csr

	case req.Method == http.MethodGet && req.URL.Path == "/apis/certificates.k8s.io/v1/certificatesigningrequests" && !q.Has("watch"):
		if s.csr == nil {
			t.Fatalf("no csr")
		}
		data, _ := json.Marshal(&certapi.CertificateSigningRequestList{
			ListMeta: metav1.ListMeta{ResourceVersion: "2"},
			Items:    []certapi.CertificateSigningRequest{*s.csr.DeepCopy()},
		})
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}

	case req.Method == http.MethodGet && req.URL.Path == "/apis/certificates.k8s.io/v1/certificatesigningrequests" && q.Has("watch"):
		if s.csr == nil {
			t.Fatalf("no csr")
		}

		if q.Has("sendInitialEvents") {
			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("ResponseWriter doesn't support Flusher")
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			evt, _ := json.Marshal(&metav1.WatchEvent{
				Type:   string(watch.Added),
				Object: runtime.RawExtension{Raw: mustMarshalJSON(s.csr.DeepCopy())},
			})
			if _, err := w.Write(evt); err != nil {
				t.Fatal(err)
			}
			bookmark, _ := json.Marshal(&metav1.WatchEvent{
				Type: string(watch.Bookmark),
				Object: runtime.RawExtension{Raw: mustMarshalJSON(&certapi.CertificateSigningRequest{
					TypeMeta:   metav1.TypeMeta{Kind: "CertificateSigningRequest", APIVersion: "certificates.k8s.io/v1"},
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{metav1.InitialEventsAnnotationKey: "true"}},
				})},
			})
			if _, err := w.Write(bookmark); err != nil {
				t.Fatal(err)
			}
			flusher.Flush()
		} else {
			data, _ := json.Marshal(&metav1.WatchEvent{
				Type:   "ADDED",
				Object: runtime.RawExtension{Raw: mustMarshalJSON(s.csr.DeepCopy())},
			})
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write(data); err != nil {
				t.Fatal(err)
			}
		}

	default:
		t.Fatalf("unexpected request: %s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
	}
}

func mustMarshalJSON(obj interface{}) []byte {
	data, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	return data
}
