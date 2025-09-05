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
	defer conn.Close()

	return conn.ConnectionState().PeerCertificates[0], nil
}

var _ = SIGDescribe("KubletTLS", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("kubelet-tls-test")

	ginkgo.Context("certificate reload", func() {
		var tempDir string
		var oldCfg *kubeletconfig.KubeletConfiguration

		t := ginkgo.GinkgoT()
		ginkgo.BeforeEach(func(ctx context.Context) {
			tempDir = t.TempDir()
			oldCfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			certPath, keyPath, err := createCertAndKeyFiles(tempDir)
			framework.ExpectNoError(err)

			cfg := oldCfg.DeepCopy()
			cfg.TLSCertFile = certPath
			cfg.TLSPrivateKeyFile = keyPath

			updateKubeletConfig(ctx, f, cfg, true)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if oldCfg != nil {
				updateKubeletConfig(ctx, f, oldCfg, true)
			}
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
			}).Should(gomega.BeFalse())
		})
	})
})
