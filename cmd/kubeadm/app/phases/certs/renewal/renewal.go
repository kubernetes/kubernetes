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
	"crypto/x509"
	"net"

	"github.com/pkg/errors"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// RenewExistingCert loads a certificate file, uses the renew interface to renew it,
// and saves the resulting certificate and key over the old one.
func RenewExistingCert(certsDir, baseName string, impl Interface) error {
	certificatePath, _ := pkiutil.PathsForCertAndKey(certsDir, baseName)
	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return errors.Wrapf(err, "failed to load existing certificate %s", baseName)
	}

	if len(certs) != 1 {
		return errors.Errorf("wanted exactly one certificate, got %d", len(certs))
	}

	cfg := certToConfig(certs[0])
	newCert, newKey, err := impl.Renew(cfg)
	if err != nil {
		return errors.Wrapf(err, "failed to renew certificate %s", baseName)
	}

	if err := pkiutil.WriteCertAndKey(certsDir, baseName, newCert, newKey); err != nil {
		return errors.Wrapf(err, "failed to write new certificate %s", baseName)
	}
	return nil
}

func certToConfig(cert *x509.Certificate) *certutil.Config {
	return &certutil.Config{
		CommonName:   cert.Subject.CommonName,
		Organization: cert.Subject.Organization,
		AltNames: certutil.AltNames{
			IPs:      cert.IPAddresses,
			DNSNames: cert.DNSNames,
		},
		Usages: cert.ExtKeyUsage,
	}
}

// RenewAndMutateExistingEtcdServerCert loads a certificate file, uses the renew interface to renew it,
// and saves the resulting certificate and key over the old one.
// This method differs from usual RenewExistingCert because it checks if the etcd server certificate
// includes the advertiseAddress in the SANS list; if not, the certificate is mutated in order to include it.
// N.B. this code is necessary only in v1.14; starting from v1.15 all the etcd manifests should have 2 endpoints
func RenewAndMutateExistingEtcdServerCert(certsDir, baseName string, advertiseAddress net.IP, impl Interface) error {
	certificatePath, _ := pkiutil.PathsForCertAndKey(certsDir, baseName)
	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return errors.Wrapf(err, "failed to load existing certificate %s", baseName)
	}

	if len(certs) != 1 {
		return errors.Errorf("wanted exactly one certificate from %s, got %d", baseName, len(certs))
	}

	cfg := certToConfig(certs[0])

	hasAdvertiseAddress := false
	for _, val := range cfg.AltNames.IPs {
		if val.Equal(advertiseAddress) {
			hasAdvertiseAddress = true
			break
		}
	}
	if !hasAdvertiseAddress {
		cfg.AltNames.IPs = append(cfg.AltNames.IPs, advertiseAddress)
	}

	newCert, newKey, err := impl.Renew(cfg)
	if err != nil {
		return errors.Wrapf(err, "failed to renew certificate %s", baseName)
	}

	if err := pkiutil.WriteCertAndKey(certsDir, baseName, newCert, newKey); err != nil {
		return errors.Wrapf(err, "failed to write new certificate %s", baseName)
	}
	return nil
}
