/*
Copyright 2023 The Kubernetes Authors.

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

package auth

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"regexp"

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe(feature.ClusterTrustBundle, feature.ClusterTrustBundleProjection, func() {
	f := framework.NewDefaultFramework("projected-clustertrustbundle")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	goodCert1 := mustMakeCertificate(&x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root1",
		},
		IsCA:                  true,
		BasicConstraintsValid: true,
	})

	goodCert1Block := string(mustMakePEMBlock("CERTIFICATE", nil, goodCert1))

	ginkgo.It("should be able to mount a single ClusterTrustBundle by name", func(ctx context.Context) {

		ctb1 := &certificatesv1alpha1.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
				TrustBundle: goodCert1Block,
			},
		}

		if _, err := f.ClientSet.CertificatesV1alpha1().ClusterTrustBundles().Create(ctx, ctb1, metav1.CreateOptions{}); err != nil {
			framework.Failf("Error while creating ClusterTrustBundle: %v", err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-ctb-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:  "projected-ctb-volume-test",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args: []string{
							"mounttest",
							"--file_content=/var/run/ctbtest/trust-anchors.pem",
							"--file_mode=/var/run/ctbtest/trust-anchors.pem",
						},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "ctb-volume",
								MountPath: "/var/run/ctbtest",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "ctb-volume",
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ClusterTrustBundle: &v1.ClusterTrustBundleProjection{
											Name: ptr.To("ctb1"),
											Path: "trust-anchors.pem",
										},
									},
								},
							},
						},
					},
				},
			},
		}

		fileModeRegexp := getFileModeRegex("/var/run/ctbtest/trust-anchors.pem", nil)
		expectedOutput := []string{
			regexp.QuoteMeta(goodCert1Block),
			fileModeRegexp,
		}

		e2epodoutput.TestContainerOutputRegexp(ctx, f, "project cluster trust bundle", pod, 0, expectedOutput)
	})
})

func mustMakeCertificate(template *x509.Certificate) []byte {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		framework.Failf("Error while generating key: %v", err)
	}

	cert, err := x509.CreateCertificate(rand.Reader, template, template, pub, priv)
	if err != nil {
		framework.Failf("Error while making certificate: %v", err)
	}

	return cert
}

func mustMakePEMBlock(blockType string, headers map[string]string, data []byte) string {
	return string(pem.EncodeToMemory(&pem.Block{
		Type:    blockType,
		Headers: headers,
		Bytes:   data,
	}))
}

// getFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func getFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}
