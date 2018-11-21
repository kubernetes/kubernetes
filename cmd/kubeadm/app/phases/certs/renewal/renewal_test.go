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
	"crypto/x509/pkix"
	"net"
	"os"
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
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	certtestutil "k8s.io/kubernetes/cmd/kubeadm/test/certs"
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
	certReqNoCert := certReq.DeepCopy()
	certReqNoCert.Status.Certificate = nil
	client.AddReactor("get", "certificatesigningrequests", defaultReactionFunc(certReq))
	watcher := watch.NewFakeWithChanSize(3, false)
	watcher.Add(certReqNoCert)
	watcher.Modify(certReqNoCert)
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
				{
					Type: certsapi.CertificateApproved,
				},
			},
			Certificate: certutil.EncodeCertPEM(cert),
		},
	}
}

func TestCertToConfig(t *testing.T) {
	expectedConfig := &certutil.Config{
		CommonName:   "test-common-name",
		Organization: []string{"sig-cluster-lifecycle"},
		AltNames: certutil.AltNames{
			IPs:      []net.IP{net.ParseIP("10.100.0.1")},
			DNSNames: []string{"test-domain.space"},
		},
		Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	cert := &x509.Certificate{
		Subject: pkix.Name{
			CommonName:   "test-common-name",
			Organization: []string{"sig-cluster-lifecycle"},
		},
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		DNSNames:    []string{"test-domain.space"},
		IPAddresses: []net.IP{net.ParseIP("10.100.0.1")},
	}

	cfg := certToConfig(cert)

	if cfg.CommonName != expectedConfig.CommonName {
		t.Errorf("expected common name %q, got %q", expectedConfig.CommonName, cfg.CommonName)
	}

	if len(cfg.Organization) != 1 || cfg.Organization[0] != expectedConfig.Organization[0] {
		t.Errorf("expected organization %v, got %v", expectedConfig.Organization, cfg.Organization)

	}

	if len(cfg.Usages) != 1 || cfg.Usages[0] != expectedConfig.Usages[0] {
		t.Errorf("expected ext key usage %v, got %v", expectedConfig.Usages, cfg.Usages)
	}

	if len(cfg.AltNames.IPs) != 1 || cfg.AltNames.IPs[0].String() != expectedConfig.AltNames.IPs[0].String() {
		t.Errorf("expected SAN IPs %v, got %v", expectedConfig.AltNames.IPs, cfg.AltNames.IPs)
	}

	if len(cfg.AltNames.DNSNames) != 1 || cfg.AltNames.DNSNames[0] != expectedConfig.AltNames.DNSNames[0] {
		t.Errorf("expected SAN DNSNames %v, got %v", expectedConfig.AltNames.DNSNames, cfg.AltNames.DNSNames)
	}
}

func TestRenewExistingCert(t *testing.T) {
	cfg := &certutil.Config{
		CommonName:   "test-common-name",
		Organization: []string{"sig-cluster-lifecycle"},
		AltNames: certutil.AltNames{
			IPs:      []net.IP{net.ParseIP("10.100.0.1")},
			DNSNames: []string{"test-domain.space"},
		},
		Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	caCertCfg := &certutil.Config{CommonName: "kubernetes"}
	caCert, caKey, err := certs.NewCACertAndKey(caCertCfg)
	if err != nil {
		t.Fatalf("couldn't create CA: %v", err)
	}

	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, cfg)
	if err != nil {
		t.Fatalf("couldn't generate certificate: %v", err)
	}

	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	if err := pkiutil.WriteCertAndKey(dir, "server", cert, key); err != nil {
		t.Fatalf("couldn't write out certificate")
	}

	renewer := NewFileRenewal(caCert, caKey)

	if err := RenewExistingCert(dir, "server", renewer); err != nil {
		t.Fatalf("couldn't renew certificate: %v", err)
	}

	newCert, err := pkiutil.TryLoadCertFromDisk(dir, "server")
	if err != nil {
		t.Fatalf("couldn't load created certificate: %v", err)
	}

	if newCert.SerialNumber.Cmp(cert.SerialNumber) == 0 {
		t.Fatal("expected new certificate, but renewed certificate has same serial number")
	}

	certtestutil.AssertCertificateIsSignedByCa(t, newCert, caCert)
	certtestutil.AssertCertificateHasClientAuthUsage(t, newCert)
	certtestutil.AssertCertificateHasOrganizations(t, newCert, cfg.Organization...)
	certtestutil.AssertCertificateHasCommonName(t, newCert, cfg.CommonName)
	certtestutil.AssertCertificateHasDNSNames(t, newCert, cfg.AltNames.DNSNames...)
	certtestutil.AssertCertificateHasIPAddresses(t, newCert, cfg.AltNames.IPs...)
}
