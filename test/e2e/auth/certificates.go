/*
Copyright 2017 The Kubernetes Authors.

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
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"time"

	"k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	v1beta1client "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Certificates API", func() {
	f := framework.NewDefaultFramework("certificates")

	It("should support building a client with a CSR", func() {
		const commonName = "tester-csr"

		pk, err := pkiutil.NewPrivateKey()
		framework.ExpectNoError(err)

		pkder := x509.MarshalPKCS1PrivateKey(pk)
		pkpem := pem.EncodeToMemory(&pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: pkder,
		})

		csrb, err := cert.MakeCSR(pk, &pkix.Name{CommonName: commonName, Organization: []string{"system:masters"}}, nil, nil)
		framework.ExpectNoError(err)

		csr := &v1beta1.CertificateSigningRequest{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: commonName + "-",
			},
			Spec: v1beta1.CertificateSigningRequestSpec{
				Request: csrb,
				Usages: []v1beta1.KeyUsage{
					v1beta1.UsageSigning,
					v1beta1.UsageKeyEncipherment,
					v1beta1.UsageClientAuth,
				},
			},
		}
		csrs := f.ClientSet.CertificatesV1beta1().CertificateSigningRequests()

		framework.Logf("creating CSR")
		csr, err = csrs.Create(csr)
		framework.ExpectNoError(err)

		csrName := csr.Name

		framework.Logf("approving CSR")
		framework.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr.Status.Conditions = []v1beta1.CertificateSigningRequestCondition{
				{
					Type:    v1beta1.CertificateApproved,
					Reason:  "E2E",
					Message: "Set from an e2e test",
				},
			}
			csr, err = csrs.UpdateApproval(csr)
			if err != nil {
				csr, _ = csrs.Get(csrName, metav1.GetOptions{})
				framework.Logf("err updating approval: %v", err)
				return false, nil
			}
			return true, nil
		}))

		framework.Logf("waiting for CSR to be signed")
		framework.ExpectNoError(wait.Poll(5*time.Second, time.Minute, func() (bool, error) {
			csr, err = csrs.Get(csrName, metav1.GetOptions{})
			if err != nil {
				framework.Logf("error getting csr: %v", err)
				return false, nil
			}
			if len(csr.Status.Certificate) == 0 {
				framework.Logf("csr not signed yet")
				return false, nil
			}
			return true, nil
		}))

		framework.Logf("testing the client")
		rcfg, err := framework.LoadConfig()
		framework.ExpectNoError(err)

		rcfg.TLSClientConfig.CertData = csr.Status.Certificate
		rcfg.TLSClientConfig.KeyData = pkpem
		rcfg.TLSClientConfig.CertFile = ""
		rcfg.BearerToken = ""
		rcfg.AuthProvider = nil
		rcfg.Username = ""
		rcfg.Password = ""

		newClient, err := v1beta1client.NewForConfig(rcfg)
		framework.ExpectNoError(err)
		framework.ExpectNoError(newClient.CertificateSigningRequests().Delete(csrName, nil))
	})
})
