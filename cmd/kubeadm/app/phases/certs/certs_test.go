/*
Copyright 2016 The Kubernetes Authors.

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

package certs

import (
	"crypto/x509"
	"net"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestNewCACertAndKey(t *testing.T) {
	caCert, _, err := NewCACertAndKey()
	if err != nil {
		t.Fatalf("failed call NewCACertAndKey: %v", err)
	}

	assertIsCa(t, caCert)
}

func TestNewAPIServerCertAndKey(t *testing.T) {
	hostname := "valid-hostname"

	advertiseAddresses := []string{"1.2.3.4", "1:2:3::4"}
	for _, addr := range advertiseAddresses {
		cfg := &kubeadmapi.MasterConfiguration{
			API:        kubeadmapi.API{AdvertiseAddress: addr},
			Networking: kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
			NodeName:   "valid-hostname",
		}
		caCert, caKey, err := NewCACertAndKey()

		apiServerCert, _, err := NewAPIServerCertAndKey(cfg, caCert, caKey)
		if err != nil {
			t.Fatalf("failed creation of cert and key: %v", err)
		}

		assertIsSignedByCa(t, apiServerCert, caCert)
		assertHasServerAuth(t, apiServerCert)

		for _, DNSName := range []string{hostname, "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local"} {
			assertHasDNSNames(t, apiServerCert, DNSName)
		}
		for _, IPAddress := range []string{"10.96.0.1", addr} {
			assertHasIPAddresses(t, apiServerCert, net.ParseIP(IPAddress))
		}
	}
}

func TestNewAPIServerKubeletClientCertAndKey(t *testing.T) {
	caCert, caKey, err := NewCACertAndKey()

	apiClientCert, _, err := NewAPIServerKubeletClientCertAndKey(caCert, caKey)
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	assertIsSignedByCa(t, apiClientCert, caCert)
	assertHasClientAuth(t, apiClientCert)
	assertHasOrganization(t, apiClientCert, constants.MastersGroup)
}

func TestNewNewServiceAccountSigningKey(t *testing.T) {

	key, err := NewServiceAccountSigningKey()
	if err != nil {
		t.Fatalf("failed creation of key: %v", err)
	}

	if key.N.BitLen() < 2048 {
		t.Error("Service account signing key has less than 2048 bits size")
	}
}

func TestNewFrontProxyCACertAndKey(t *testing.T) {
	frontProxyCACert, _, err := NewFrontProxyCACertAndKey()
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	assertIsCa(t, frontProxyCACert)
}

func TestNewFrontProxyClientCertAndKey(t *testing.T) {
	frontProxyCACert, frontProxyCAKey, err := NewFrontProxyCACertAndKey()

	frontProxyClientCert, _, err := NewFrontProxyClientCertAndKey(frontProxyCACert, frontProxyCAKey)
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	assertIsSignedByCa(t, frontProxyClientCert, frontProxyCACert)
	assertHasClientAuth(t, frontProxyClientCert)
}

func assertIsCa(t *testing.T, cert *x509.Certificate) {
	if !cert.IsCA {
		t.Error("cert is not a valida CA")
	}
}

func assertIsSignedByCa(t *testing.T, cert *x509.Certificate, ca *x509.Certificate) {
	if err := cert.CheckSignatureFrom(ca); err != nil {
		t.Error("cert is not signed by ca")
	}
}

func assertHasClientAuth(t *testing.T, cert *x509.Certificate) {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageClientAuth {
			return
		}
	}
	t.Error("cert is not a ClientAuth")
}

func assertHasServerAuth(t *testing.T, cert *x509.Certificate) {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageServerAuth {
			return
		}
	}
	t.Error("cert is not a ServerAuth")
}

func assertHasOrganization(t *testing.T, cert *x509.Certificate, OU string) {
	for i := range cert.Subject.Organization {
		if cert.Subject.Organization[i] == OU {
			return
		}
	}
	t.Errorf("cert does not contain OU %s", OU)
}

func assertHasDNSNames(t *testing.T, cert *x509.Certificate, DNSName string) {
	for i := range cert.DNSNames {
		if cert.DNSNames[i] == DNSName {
			return
		}
	}
	t.Errorf("cert does not contain DNSName %s", DNSName)
}

func assertHasIPAddresses(t *testing.T, cert *x509.Certificate, IPAddress net.IP) {
	for i := range cert.IPAddresses {
		if cert.IPAddresses[i].Equal(IPAddress) {
			return
		}
	}
	t.Errorf("cert does not contain IPAddress %s", IPAddress)
}
