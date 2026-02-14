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

package e2enode

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
)

func createCAWithCertAndKeyFiles(certDir string) (string, string, string, error) {
	ca, cert, key, err := cert.GenerateCAAndCertKeyWithOptions(
		"localhost",
		[]net.IP{{127, 0, 0, 1}},
		[]string{"localhost"},
	)
	if err != nil {
		return "", "", "", nil
	}

	caPath := filepath.Join(certDir, "kubelet-client-ca.cert")
	certPath := filepath.Join(certDir, "kubelet.cert")
	keyPath := filepath.Join(certDir, "kubelet.key")

	if err = os.WriteFile(caPath, ca, os.FileMode(0644)); err != nil {
		return "", "", "", err
	}

	if err = os.WriteFile(certPath, cert, os.FileMode(0644)); err != nil {
		return "", "", "", err
	}

	if err = os.WriteFile(keyPath, key, os.FileMode(0600)); err != nil {
		return "", "", "", err
	}

	return caPath, certPath, keyPath, nil
}

func callKubeletWithClientCert(addr string, caPath, certPath, keyPath string) (*x509.Certificate, error) {
	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, err
	}

	caCert, err := os.ReadFile(caPath)
	if err != nil {
		return nil, err
	}
	caCertPool := x509.NewCertPool()
	caCertPool.AppendCertsFromPEM(caCert)

	conn, err := tls.Dial("tcp", addr, &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = conn.Close() }()

	if !conn.ConnectionState().HandshakeComplete {
		return nil, fmt.Errorf("handshake failed")
	}

	return conn.ConnectionState().PeerCertificates[0], nil
}

/*
Command to run the test locally:

	make test-e2e-node FOCUS="Feature:KubeletServerCAAndCertReload" SKIP="nothing" TEST_ARGS='--ginkgo.v --kubelet-flags="--fail-swap-on=false"'
*/
var _ = SIGDescribe("kubeletServerCAAndCertReloadTest", feature.KubeletServerCAAndCertReload, framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("kubelet-ca-and-cert-reload-test")

	ginkgo.Context("certificate reload", func() {
		var tempDir string

		t := ginkgo.GinkgoT()
		ginkgo.BeforeEach(func(ctx context.Context) {
			tempDir = t.TempDir()
			cfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func(ctx context.Context, cfg *kubeletconfig.KubeletConfiguration) {
				updateKubeletConfig(ctx, f, cfg, true)
			}, cfg.DeepCopy())

			caPath, certPath, keyPath, err := createCAWithCertAndKeyFiles(tempDir)
			framework.ExpectNoError(err)

			cfg.TLSCertFile = certPath
			cfg.TLSPrivateKeyFile = keyPath
			cfg.Authentication.X509.ClientCAFile = caPath
			cfg.FeatureGates = map[string]bool{
				"KubeletServerCAAndCertReload":       true,
				"ReloadKubeletServerCertificateFile": false,
				"RotateKubeletServerCertificate":     false,
			}
			updateKubeletConfig(ctx, f, cfg, true)
		})

		ginkgo.It("should reload certificates from disk", func(ctx context.Context) {
			cfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			addr := fmt.Sprintf("127.0.0.1:%d", ports.KubeletPort)
			oldServingCert, err := callKubeletWithClientCert(addr, cfg.Authentication.X509.ClientCAFile, cfg.TLSCertFile, cfg.TLSPrivateKeyFile)
			framework.ExpectNoError(err)

			caPath, certPath, keyPath, err := createCAWithCertAndKeyFiles(tempDir)
			framework.ExpectNoError(err)

			checkTimeout := time.Minute * 1
			checkFreq := time.Second * 10
			gomega.Eventually(ctx, func(ctx context.Context) (bool, error) {
				newServingCert, err := callKubeletWithClientCert(addr, caPath, certPath, keyPath)
				if err != nil {
					return true, err
				}

				return newServingCert.Equal(oldServingCert), nil
			}, checkTimeout, checkFreq).Should(gomega.BeFalseBecause("new certificate should be loaded"))
		})
	})
})
