/*
Copyright 2020 The Kubernetes Authors.

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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io/ioutil"
	"math"
	"math/big"
	"os"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/cert"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
)

func TestCertRotation(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	clientSigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}

	transport.CertCallbackRefreshDuration = 1 * time.Second

	certDir := os.TempDir()
	clientCAFilename := path.Join(certDir, "ca.crt")

	if err := ioutil.WriteFile(clientCAFilename, utils.EncodeCertPEM(clientSigningCert), 0644); err != nil {
		t.Fatal(err)
	}

	server := apiservertesting.StartTestServerOrDie(t, apiservertesting.NewDefaultTestServerOptions(), []string{
		"--client-ca-file=" + clientCAFilename,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	writeCerts(t, clientSigningCert, clientSigningKey, certDir, 30*time.Second)

	kubeconfig := server.ClientConfig
	kubeconfig.CertFile = path.Join(certDir, "client.crt")
	kubeconfig.KeyFile = path.Join(certDir, "client.key")
	kubeconfig.BearerToken = ""

	client := clientset.NewForConfigOrDie(kubeconfig)
	ctx := context.Background()

	w, err := client.CoreV1().ServiceAccounts("default").Watch(ctx, v1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-w.ResultChan():
		t.Fatal("Watch closed before rotation")
	default:
	}

	writeCerts(t, clientSigningCert, clientSigningKey, certDir, 5*time.Minute)

	time.Sleep(10 * time.Second)

	// Should have had a rotation; connections will have been closed
	select {
	case _, ok := <-w.ResultChan():
		assert.Equal(t, false, ok)
	default:
		t.Fatal("Watch wasn't closed despite rotation")
	}

	// Wait for old cert to expire (30s)
	time.Sleep(30 * time.Second)

	// Ensure we make requests with the new cert
	_, err = client.CoreV1().ServiceAccounts("default").List(ctx, v1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCertRotationContinuousRequests(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	clientSigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}

	transport.CertCallbackRefreshDuration = 1 * time.Second

	certDir := os.TempDir()
	clientCAFilename := path.Join(certDir, "ca.crt")

	if err := ioutil.WriteFile(clientCAFilename, utils.EncodeCertPEM(clientSigningCert), 0644); err != nil {
		t.Fatal(err)
	}

	server := apiservertesting.StartTestServerOrDie(t, apiservertesting.NewDefaultTestServerOptions(), []string{
		"--client-ca-file=" + clientCAFilename,
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	writeCerts(t, clientSigningCert, clientSigningKey, certDir, 30*time.Second)

	kubeconfig := server.ClientConfig
	kubeconfig.CertFile = path.Join(certDir, "client.crt")
	kubeconfig.KeyFile = path.Join(certDir, "client.key")
	kubeconfig.BearerToken = ""

	client := clientset.NewForConfigOrDie(kubeconfig)

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(10 * time.Second)

		writeCerts(t, clientSigningCert, clientSigningKey, certDir, 5*time.Minute)

		// Wait for old cert to expire (30s)
		time.Sleep(30 * time.Second)
		cancel()
	}()

	for range time.Tick(time.Second) {
		_, err := client.CoreV1().ServiceAccounts("default").List(ctx, v1.ListOptions{})
		if err != nil {
			if err == ctx.Err() {
				return
			}

			t.Fatal(err)
		}
	}
}

func writeCerts(t *testing.T, clientSigningCert *x509.Certificate, clientSigningKey *rsa.PrivateKey, certDir string, duration time.Duration) {
	clientKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	privBytes, err := x509.MarshalPKCS8PrivateKey(clientKey)
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(path.Join(certDir, "client.key"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes}), 0666); err != nil {
		t.Fatal(err)
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64))
	if err != nil {
		t.Fatal(err)
	}

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName:   "foo",
			Organization: []string{"system:masters"},
		},
		SerialNumber: serial,
		NotBefore:    clientSigningCert.NotBefore,
		NotAfter:     time.Now().Add(duration).UTC(),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	certDERBytes, err := x509.CreateCertificate(rand.Reader, &certTmpl, clientSigningCert, clientKey.Public(), clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(path.Join(certDir, "client.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDERBytes}), 0666); err != nil {
		t.Fatal(err)
	}
}
