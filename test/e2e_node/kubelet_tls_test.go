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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
)

func createCertAndKeyFiles(certDir string) (string, string, error) {
	cert, key, err := cert.GenerateSelfSignedCertKey(
		"localhost",
		[]net.IP{{127, 0, 0, 1}},
		[]string{"localhost"},
	)
	if err != nil {
		return "", "", nil
	}

	certPath := filepath.Join(certDir, "kubelet.cert")
	keyPath := filepath.Join(certDir, "kubelet.key")
	if err = os.WriteFile(certPath, cert, os.FileMode(0644)); err != nil {
		return "", "", err
	}

	if err = os.WriteFile(keyPath, key, os.FileMode(0600)); err != nil {
		return "", "", err
	}

	return certPath, keyPath, nil
}

func getCert(addr string) (*x509.Certificate, error) {
	conn, err := tls.Dial("tcp", addr, &tls.Config{InsecureSkipVerify: true})
	if err != nil {
		return nil, err
	}
	defer func() { _ = conn.Close() }()

	return conn.ConnectionState().PeerCertificates[0], nil
}

var _ = SIGDescribe("KubletTLS", framework.WithSerial(), framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("kubelet-tls-test")

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

			certPath, keyPath, err := createCertAndKeyFiles(tempDir)
			framework.ExpectNoError(err)

			cfg.TLSCertFile = certPath
			cfg.TLSPrivateKeyFile = keyPath
			updateKubeletConfig(ctx, f, cfg, true)
		})

		ginkgo.It("should reload certificates from disk", func(ctx context.Context) {
			addr := fmt.Sprintf("127.0.0.1:%d", ports.KubeletPort)
			oldCert, err := getCert(addr)
			framework.ExpectNoError(err)

			_, _, err = createCertAndKeyFiles(tempDir)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(ctx context.Context) (bool, error) {
				newCert, err := getCert(addr)
				if err != nil {
					return true, err
				}

				return newCert.Equal(oldCert), nil
			}).Should(gomega.BeFalseBecause("new certificate should be loaded"))
		})
	})
})
