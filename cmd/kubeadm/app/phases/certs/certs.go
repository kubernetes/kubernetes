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
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"

	"github.com/golang/glog"

	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// If the PKI assets already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
func CreatePKIAssets(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating PKI assets")

	// This structure cannot handle multilevel CA hierarchies.
	// This isn't a problem right now, but may become one in the future.

	var certList Certificates

	if cfg.Etcd.Local == nil {
		certList = GetCertsWithoutEtcd()
	} else {
		certList = GetDefaultCertList()
	}

	certTree, err := certList.AsMap().CertTree()
	if err != nil {
		return err
	}

	if err := certTree.CreateTree(cfg); err != nil {
		return fmt.Errorf("Error creating PKI assets: %v", err)
	}

	fmt.Printf("[certificates] valid certificates and keys now exist in %q\n", cfg.CertificatesDir)

	// Service accounts are not x509 certs, so handled separately
	if err := CreateServiceAccountKeyAndPublicKeyFiles(cfg); err != nil {
		return err
	}

	return nil
}

// CreateServiceAccountKeyAndPublicKeyFiles create a new public/private key files for signing service account users.
// If the sa public/private key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateServiceAccountKeyAndPublicKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new public/private key files for signing service account users")
	saSigningKey, err := NewServiceAccountSigningKey()
	if err != nil {
		return err
	}

	return writeKeyFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.ServiceAccountKeyBaseName,
		saSigningKey,
	)
}

// NewServiceAccountSigningKey generate public/private key pairs for signing service account tokens.
func NewServiceAccountSigningKey() (*rsa.PrivateKey, error) {
	// The key does NOT exist, let's generate it now
	saSigningKey, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, fmt.Errorf("failure while creating service account token signing key: %v", err)
	}

	return saSigningKey, nil
}

// NewCACertAndKey will generate a self signed CA.
func NewCACertAndKey(certSpec *certutil.Config) (*x509.Certificate, *rsa.PrivateKey, error) {

	caCert, caKey, err := pkiutil.NewCertificateAuthority(certSpec)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating CA certificate and key: %v", err)
	}

	return caCert, caKey, nil
}

// CreateCACertAndKeyFiles generates and writes out a given certificate authority.
// The certSpec should be one of the variables from this package.
func CreateCACertAndKeyFiles(certSpec *KubeadmCert, cfg *kubeadmapi.InitConfiguration) error {
	if certSpec.CAName != "" {
		return fmt.Errorf("This function should only be used for CAs, but cert %s has CA %s", certSpec.Name, certSpec.CAName)
	}
	glog.V(1).Infoln("creating a new certificate authority for %s", certSpec.Name)

	certConfig, err := certSpec.GetConfig(cfg)
	if err != nil {
		return err
	}

	caCert, caKey, err := NewCACertAndKey(certConfig)
	if err != nil {
		return err
	}

	return writeCertificateAuthorithyFilesIfNotExist(
		cfg.CertificatesDir,
		certSpec.BaseName,
		caCert,
		caKey,
	)
}

// CreateCertAndKeyFilesWithCA loads the given certificate authority from disk, then generates and writes out the given certificate and key.
// The certSpec and caCertSpec should both be one of the variables from this package.
func CreateCertAndKeyFilesWithCA(certSpec *KubeadmCert, caCertSpec *KubeadmCert, cfg *kubeadmapi.InitConfiguration) error {
	if certSpec.CAName != caCertSpec.Name {
		return fmt.Errorf("Expected CAname for %s to be %q, but was %s", certSpec.Name, certSpec.CAName, caCertSpec.Name)
	}

	caCert, caKey, err := LoadCertificateAuthority(cfg.CertificatesDir, caCertSpec.BaseName)
	if err != nil {
		return fmt.Errorf("Couldn't load CA certificate %s: %v", caCertSpec.Name, err)
	}

	return certSpec.CreateFromCA(cfg, caCert, caKey)
}

func newCertAndKeyFromSpec(certSpec *KubeadmCert, cfg *kubeadmapi.InitConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {
	certConfig, err := certSpec.GetConfig(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating certificate %s: %v", certSpec.Name, err)
	}
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, certConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating %s key and certificate: %v", certSpec.Name, err)
	}

	return cert, key, err
}

// LoadCertificateAuthority tries to load a CA in the given directory with the given name.
func LoadCertificateAuthority(pkiDir string, baseName string) (*x509.Certificate, *rsa.PrivateKey, error) {
	// Checks if certificate authority exists in the PKI directory
	if !pkiutil.CertOrKeyExist(pkiDir, baseName) {
		return nil, nil, fmt.Errorf("couldn't load %s certificate authority from %s", baseName, pkiDir)
	}

	// Try to load certificate authority .crt and .key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
	if err != nil {
		return nil, nil, fmt.Errorf("failure loading %s certificate authority: %v", baseName, err)
	}

	// Make sure the loaded CA cert actually is a CA
	if !caCert.IsCA {
		return nil, nil, fmt.Errorf("%s certificate is not a certificate authority", baseName)
	}

	return caCert, caKey, nil
}

// writeCertificateAuthorithyFilesIfNotExist write a new certificate Authority to the given path.
// If there already is a certificate file at the given path; kubeadm tries to load it and check if the values in the
// existing and the eexpected certificate equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// otherwise this function returns an error.
func writeCertificateAuthorithyFilesIfNotExist(pkiDir string, baseName string, caCert *x509.Certificate, caKey *rsa.PrivateKey) error {

	// If cert or key exists, we should try to load them
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {

		// Try to load .crt and .key from the PKI directory
		caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("failure loading %s certificate: %v", baseName, err)
		}

		// Check if the existing cert is a CA
		if !caCert.IsCA {
			return fmt.Errorf("certificate %s is not a CA", baseName)
		}

		// kubeadm doesn't validate the existing certificate Authority more than this;
		// Basically, if we find a certificate file with the same path; and it is a CA
		// kubeadm thinks those files are equal and doesn't bother writing a new file
		fmt.Printf("[certificates] Using the existing %s certificate and key.\n", baseName)
	} else {

		// Write .crt and .key files to disk
		if err := pkiutil.WriteCertAndKey(pkiDir, baseName, caCert, caKey); err != nil {
			return fmt.Errorf("failure while saving %s certificate and key: %v", baseName, err)
		}

		fmt.Printf("[certificates] Generated %s certificate and key.\n", baseName)
	}
	return nil
}

// writeCertificateFilesIfNotExist write a new certificate to the given path.
// If there already is a certificate file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected certificate equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// otherwise this function returns an error.
func writeCertificateFilesIfNotExist(pkiDir string, baseName string, signingCert *x509.Certificate, cert *x509.Certificate, key *rsa.PrivateKey) error {

	// Checks if the signed certificate exists in the PKI directory
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {
		// Try to load signed certificate .crt and .key from the PKI directory
		signedCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("failure loading %s certificate: %v", baseName, err)
		}

		// Check if the existing cert is signed by the given CA
		if err := signedCert.CheckSignatureFrom(signingCert); err != nil {
			return fmt.Errorf("certificate %s is not signed by corresponding CA", baseName)
		}

		// kubeadm doesn't validate the existing certificate more than this;
		// Basically, if we find a certificate file with the same path; and it is signed by
		// the expected certificate authority, kubeadm thinks those files are equal and
		// doesn't bother writing a new file
		fmt.Printf("[certificates] Using the existing %s certificate and key.\n", baseName)
	} else {

		// Write .crt and .key files to disk
		if err := pkiutil.WriteCertAndKey(pkiDir, baseName, cert, key); err != nil {
			return fmt.Errorf("failure while saving %s certificate and key: %v", baseName, err)
		}

		fmt.Printf("[certificates] Generated %s certificate and key.\n", baseName)
		if pkiutil.HasServerAuth(cert) {
			fmt.Printf("[certificates] %s serving cert is signed for DNS names %v and IPs %v\n", baseName, cert.DNSNames, cert.IPAddresses)
		}
	}

	return nil
}

// writeKeyFilesIfNotExist write a new key to the given path.
// If there already is a key file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected key equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// otherwise this function returns an error.
func writeKeyFilesIfNotExist(pkiDir string, baseName string, key *rsa.PrivateKey) error {

	// Checks if the key exists in the PKI directory
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {

		// Try to load .key from the PKI directory
		_, err := pkiutil.TryLoadKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("%s key existed but it could not be loaded properly: %v", baseName, err)
		}

		// kubeadm doesn't validate the existing certificate key more than this;
		// Basically, if we find a key file with the same path kubeadm thinks those files
		// are equal and doesn't bother writing a new file
		fmt.Printf("[certificates] Using the existing %s key.\n", baseName)
	} else {

		// Write .key and .pub files to disk
		if err := pkiutil.WriteKey(pkiDir, baseName, key); err != nil {
			return fmt.Errorf("failure while saving %s key: %v", baseName, err)
		}

		if err := pkiutil.WritePublicKey(pkiDir, baseName, &key.PublicKey); err != nil {
			return fmt.Errorf("failure while saving %s public key: %v", baseName, err)
		}
		fmt.Printf("[certificates] Generated %s key and public key.\n", baseName)
	}

	return nil
}

type certKeyLocation struct {
	pkiDir     string
	caBaseName string
	baseName   string
	uxName     string
}

// SharedCertificateExists verifies if the shared certificates - the certificates that must be
// equal across masters: ca.key, ca.crt, sa.key, sa.pub
func SharedCertificateExists(cfg *kubeadmapi.InitConfiguration) (bool, error) {

	if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, "", "CA"}); err != nil {
		return false, err
	}

	if err := validatePrivatePublicKey(certKeyLocation{cfg.CertificatesDir, "", kubeadmconstants.ServiceAccountKeyBaseName, "service account"}); err != nil {
		return false, err
	}

	if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, "", "front-proxy CA"}); err != nil {
		return false, err
	}

	return true, nil
}

// UsingExternalCA determines whether the user is relying on an external CA.  We currently implicitly determine this is the case
// when both the CA Cert and the front proxy CA Cert are present but the CA Key and front proxy CA Key are not.
// This allows us to, e.g., skip generating certs or not start the csr signing controller.
func UsingExternalCA(cfg *kubeadmapi.InitConfiguration) (bool, error) {

	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, "", "CA"}); err != nil {
		return false, err
	}

	caKeyPath := filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)
	if _, err := os.Stat(caKeyPath); !os.IsNotExist(err) {
		return false, fmt.Errorf("%s exists", kubeadmconstants.CAKeyName)
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, kubeadmconstants.APIServerCertAndKeyBaseName, "API server"}); err != nil {
		return false, err
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, "API server kubelet client"}); err != nil {
		return false, err
	}

	if err := validatePrivatePublicKey(certKeyLocation{cfg.CertificatesDir, "", kubeadmconstants.ServiceAccountKeyBaseName, "service account"}); err != nil {
		return false, err
	}

	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, "", "front-proxy CA"}); err != nil {
		return false, err
	}

	frontProxyCAKeyPath := filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCAKeyName)
	if _, err := os.Stat(frontProxyCAKeyPath); !os.IsNotExist(err) {
		return false, fmt.Errorf("%s exists", kubeadmconstants.FrontProxyCAKeyName)
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, kubeadmconstants.FrontProxyClientCertAndKeyBaseName, "front-proxy client"}); err != nil {
		return false, err
	}

	return true, nil
}

// validateCACert tries to load a x509 certificate from pkiDir and validates that it is a CA
func validateCACert(l certKeyLocation) error {
	// Check CA Cert
	caCert, err := pkiutil.TryLoadCertFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		return fmt.Errorf("failure loading certificate for %s: %v", l.uxName, err)
	}

	// Check if cert is a CA
	if !caCert.IsCA {
		return fmt.Errorf("certificate %s is not a CA", l.uxName)
	}
	return nil
}

// validateCACertAndKey tries to load a x509 certificate and private key from pkiDir,
// and validates that the cert is a CA
func validateCACertAndKey(l certKeyLocation) error {
	if err := validateCACert(l); err != nil {
		return err
	}

	_, err := pkiutil.TryLoadKeyFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		return fmt.Errorf("failure loading key for %s: %v", l.uxName, err)
	}
	return nil
}

// validateSignedCert tries to load a x509 certificate and private key from pkiDir and validates
// that the cert is signed by a given CA
func validateSignedCert(l certKeyLocation) error {
	// Try to load CA
	caCert, err := pkiutil.TryLoadCertFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		return fmt.Errorf("failure loading certificate authority for %s: %v", l.uxName, err)
	}

	return validateSignedCertWithCA(l, caCert)
}

// validateSignedCertWithCA tries to load a certificate and validate it with the given caCert
func validateSignedCertWithCA(l certKeyLocation, caCert *x509.Certificate) error {
	// Try to load key and signed certificate
	signedCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(l.pkiDir, l.baseName)
	if err != nil {
		return fmt.Errorf("failure loading certificate for %s: %v", l.uxName, err)
	}

	// Check if the cert is signed by the CA
	if err := signedCert.CheckSignatureFrom(caCert); err != nil {
		return fmt.Errorf("certificate %s is not signed by corresponding CA", l.uxName)
	}
	return nil
}

// validatePrivatePublicKey tries to load a private key from pkiDir
func validatePrivatePublicKey(l certKeyLocation) error {
	// Try to load key
	_, _, err := pkiutil.TryLoadPrivatePublicKeyFromDisk(l.pkiDir, l.baseName)
	if err != nil {
		return fmt.Errorf("failure loading key for %s: %v", l.uxName, err)
	}
	return nil
}
