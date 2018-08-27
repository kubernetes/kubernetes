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

package renewal

import (
	"crypto/rsa"
	"crypto/x509"
	"testing"
	"time"

	certsapi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	fakecerts "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake"
	k8stesting "k8s.io/client-go/testing"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

func TestRenewImplementations(t *testing.T) {
	caCertCfg := &certutil.Config{CommonName: "kubernetes"}
	caCert, caKey, err := certs.NewCACertAndKey(caCertCfg)
	if err != nil {
		t.Fatalf("couldn't create CA: %v", err)
	}

	client := &fakecerts.FakeCertificatesV1beta1{
		Fake: &k8stesting.Fake{},
	}
	certReq := getCertReq(t, caCert, caKey)
	client.AddReactor("get", "certificatesigningrequests", defaultReactionFunc(certReq))
	watcher := watch.NewFakeWithChanSize(1, false)
	watcher.Modify(certReq)
	client.AddWatchReactor("certificatesigningrequests", k8stesting.DefaultWatchReactor(watcher, nil))

	// override the timeout so tests are faster
	watchTimeout = time.Second

	tests := []struct {
		name string
		impl Interface
	}{
		{
			name: "filerenewal",
			impl: NewFileRenewal(caCert, caKey),
		},
		{
			name: "certs api",
			impl: &CertsAPIRenewal{
				client: client,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			certCfg := &certutil.Config{
				CommonName: "test-certs",
				AltNames: certutil.AltNames{
					DNSNames: []string{"test-domain.space"},
				},
				Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			}

			cert, _, err := test.impl.Renew(certCfg)
			if err != nil {
				t.Fatalf("unexpected error renewing cert: %v", err)
			}

			pool := x509.NewCertPool()
			pool.AddCert(caCert)

			_, err = cert.Verify(x509.VerifyOptions{
				DNSName:   "test-domain.space",
				Roots:     pool,
				KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			})
			if err != nil {
				t.Errorf("couldn't verify new cert: %v", err)
			}
		})
	}
}

func defaultReactionFunc(obj runtime.Object) k8stesting.ReactionFunc {
	return func(act k8stesting.Action) (bool, runtime.Object, error) {
		return true, obj, nil
	}
}

func getCertReq(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey) *certsapi.CertificateSigningRequest {
	cert, _, err := pkiutil.NewCertAndKey(caCert, caKey, &certutil.Config{
		CommonName: "testcert",
		AltNames: certutil.AltNames{
			DNSNames: []string{"test-domain.space"},
		},
		Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	})
	if err != nil {
		t.Fatalf("couldn't generate cert: %v", err)
	}

	return &certsapi.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "testcert",
		},
		Status: certsapi.CertificateSigningRequestStatus{
			Conditions: []certsapi.CertificateSigningRequestCondition{
				certsapi.CertificateSigningRequestCondition{
					Type: certsapi.CertificateApproved,
				},
			},
			Certificate: cert.Raw,
		},
	}
}
