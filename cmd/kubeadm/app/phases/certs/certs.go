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
	"crypto"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

var (
	// certPeriodValidation is used to store if period validation was done for a certificate
	certPeriodValidationMutex sync.Mutex
	certPeriodValidation      = map[string]struct{}{}
)

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// If the PKI assets already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
func CreatePKIAssets(cfg *kubeadmapi.InitConfiguration) error {
	klog.V(1).Infoln("creating PKI assets")

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
		return errors.Wrap(err, "error creating PKI assets")
	}

	fmt.Printf("[certs] Valid certificates and keys now exist in %q\n", cfg.CertificatesDir)

	// Service accounts are not x509 certs, so handled separately
	return CreateServiceAccountKeyAndPublicKeyFiles(cfg.CertificatesDir, cfg.ClusterConfiguration.EncryptionAlgorithmType())
}

// CreateServiceAccountKeyAndPublicKeyFiles creates new public/private key files for signing service account users.
// If the sa public/private key files already exist in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateServiceAccountKeyAndPublicKeyFiles(certsDir string, keyType kubeadmapi.EncryptionAlgorithmType) error {
	klog.V(1).Infoln("creating new public/private key files for signing service account users")
	_, err := keyutil.PrivateKeyFromFile(filepath.Join(certsDir, kubeadmconstants.ServiceAccountPrivateKeyName))
	if err == nil {
		// kubeadm doesn't validate the existing certificate key more than this;
		// Basically, if we find a key file with the same path kubeadm thinks those files
		// are equal and doesn't bother writing a new file
		fmt.Printf("[certs] Using the existing %q key\n", kubeadmconstants.ServiceAccountKeyBaseName)
		return nil
	} else if !os.IsNotExist(err) {
		return errors.Wrapf(err, "file %s existed but it could not be loaded properly", kubeadmconstants.ServiceAccountPrivateKeyName)
	}

	// The key does NOT exist, let's generate it now
	key, err := pkiutil.NewPrivateKey(keyType)
	if err != nil {
		return err
	}

	// Write .key and .pub files to disk
	fmt.Printf("[certs] Generating %q key and public key\n", kubeadmconstants.ServiceAccountKeyBaseName)

	if err := pkiutil.WriteKey(certsDir, kubeadmconstants.ServiceAccountKeyBaseName, key); err != nil {
		return err
	}

	return pkiutil.WritePublicKey(certsDir, kubeadmconstants.ServiceAccountKeyBaseName, key.Public())
}

// CreateCACertAndKeyFiles generates and writes out a given certificate authority.
// The certSpec should be one of the variables from this package.
func CreateCACertAndKeyFiles(certSpec *KubeadmCert, cfg *kubeadmapi.InitConfiguration) error {
	if certSpec.CAName != "" {
		return errors.Errorf("this function should only be used for CAs, but cert %s has CA %s", certSpec.Name, certSpec.CAName)
	}
	klog.V(1).Infof("creating a new certificate authority for %s", certSpec.Name)

	certConfig, err := certSpec.GetConfig(cfg)
	if err != nil {
		return err
	}

	caCert, caKey, err := pkiutil.NewCertificateAuthority(certConfig)
	if err != nil {
		return err
	}

	return writeCertificateAuthorityFilesIfNotExist(
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
		return errors.Errorf("expected CAname for %s to be %q, but was %s", certSpec.Name, certSpec.CAName, caCertSpec.Name)
	}

	caCert, caKey, err := LoadCertificateAuthority(cfg.CertificatesDir, caCertSpec.BaseName)
	if err != nil {
		return errors.Wrapf(err, "couldn't load CA certificate %s", caCertSpec.Name)
	}

	return certSpec.CreateFromCA(cfg, caCert, caKey)
}

// LoadCertificateAuthority tries to load a CA in the given directory with the given name.
func LoadCertificateAuthority(pkiDir string, baseName string) (*x509.Certificate, crypto.Signer, error) {
	// Checks if certificate authority exists in the PKI directory
	if !pkiutil.CertOrKeyExist(pkiDir, baseName) {
		return nil, nil, errors.Errorf("couldn't load %s certificate authority from %s", baseName, pkiDir)
	}

	// Try to load certificate authority .crt and .key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "failure loading %s certificate authority", baseName)
	}
	// Validate period
	CheckCertificatePeriodValidity(baseName, caCert)

	// Make sure the loaded CA cert actually is a CA
	if !caCert.IsCA {
		return nil, nil, errors.Errorf("%s certificate is not a certificate authority", baseName)
	}

	return caCert, caKey, nil
}

// writeCertificateAuthorityFilesIfNotExist write a new certificate Authority to the given path.
// If there already is a certificate file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected certificate equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// otherwise this function returns an error.
func writeCertificateAuthorityFilesIfNotExist(pkiDir string, baseName string, caCert *x509.Certificate, caKey crypto.Signer) error {

	// If cert or key exists, we should try to load them
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {

		// Try to load .crt and .key from the PKI directory
		caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return errors.Wrapf(err, "failure loading %s certificate", baseName)
		}
		// Validate period
		CheckCertificatePeriodValidity(baseName, caCert)

		// Check if the existing cert is a CA
		if !caCert.IsCA {
			return errors.Errorf("certificate %s is not a CA", baseName)
		}

		// kubeadm doesn't validate the existing certificate Authority more than this;
		// Basically, if we find a certificate file with the same path; and it is a CA
		// kubeadm thinks those files are equal and doesn't bother writing a new file
		fmt.Printf("[certs] Using the existing %q certificate and key\n", baseName)
	} else {
		// Write .crt and .key files to disk
		fmt.Printf("[certs] Generating %q certificate and key\n", baseName)

		if err := pkiutil.WriteCertAndKey(pkiDir, baseName, caCert, caKey); err != nil {
			return errors.Wrapf(err, "failure while saving %s certificate and key", baseName)
		}
	}
	return nil
}

// writeCertificateFilesIfNotExist write a new certificate to the given path.
// If there already is a certificate file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected certificate equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// otherwise this function returns an error.
func writeCertificateFilesIfNotExist(pkiDir string, baseName string, signingCert *x509.Certificate, cert *x509.Certificate, key crypto.Signer, cfg *pkiutil.CertConfig) error {

	// Checks if the signed certificate exists in the PKI directory
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {
		// Try to load key from the PKI directory
		_, err := pkiutil.TryLoadKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return errors.Wrapf(err, "failure loading %s key", baseName)
		}

		// Try to load certificate from the PKI directory
		signedCert, intermediates, err := pkiutil.TryLoadCertChainFromDisk(pkiDir, baseName)
		if err != nil {
			return errors.Wrapf(err, "failure loading %s certificate", baseName)
		}
		// Validate period
		CheckCertificatePeriodValidity(baseName, signedCert)

		// Check if the existing cert is signed by the given CA
		if err := pkiutil.VerifyCertChain(signedCert, intermediates, signingCert); err != nil {
			return errors.Errorf("certificate %s is not signed by corresponding CA", baseName)
		}

		// Check if the certificate has the correct attributes
		if err := validateCertificateWithConfig(signedCert, baseName, cfg); err != nil {
			return err
		}

		fmt.Printf("[certs] Using the existing %q certificate and key\n", baseName)
	} else {
		// Write .crt and .key files to disk
		fmt.Printf("[certs] Generating %q certificate and key\n", baseName)

		if err := pkiutil.WriteCertAndKey(pkiDir, baseName, cert, key); err != nil {
			return errors.Wrapf(err, "failure while saving %s certificate and key", baseName)
		}
		if pkiutil.HasServerAuth(cert) {
			fmt.Printf("[certs] %s serving cert is signed for DNS names %v and IPs %v\n", baseName, cert.DNSNames, cert.IPAddresses)
		}
	}

	return nil
}

type certKeyLocation struct {
	pkiDir     string
	caBaseName string
	baseName   string
	uxName     string
}

// SharedCertificateExists verifies if the shared certificates exist and are still valid - the certificates must be
// equal across control-plane nodes: ca.key, ca.crt, sa.key, sa.pub, front-proxy-ca.key, front-proxy-ca.crt and etcd/ca.key, etcd/ca.crt if local/stacked etcd
// Missing private keys of CA are non-fatal and produce warnings.
func SharedCertificateExists(cfg *kubeadmapi.ClusterConfiguration) (bool, error) {
	var errs []error
	if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, "", "CA"}); err != nil {
		errs = append(errs, err)
	}

	if err := validatePrivatePublicKey(certKeyLocation{cfg.CertificatesDir, "", kubeadmconstants.ServiceAccountKeyBaseName, "service account"}); err != nil {
		errs = append(errs, err)
	}

	if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, "", "front-proxy CA"}); err != nil {
		errs = append(errs, err)
	}

	// in case of local/stacked etcd
	if cfg.Etcd.External == nil {
		if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, "", "etcd CA"}); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) != 0 {
		return false, utilerrors.NewAggregate(errs)
	}
	return true, nil
}

// UsingExternalCA determines whether the user is relying on an external CA.  We currently implicitly determine this is the case
// when the CA Cert is present but the CA Key is not.
// This allows us to, e.g., skip generating certs or not start the csr signing controller.
// In case we are using an external front-proxy CA, the function validates the certificates signed by front-proxy CA that should be provided by the user.
func UsingExternalCA(cfg *kubeadmapi.ClusterConfiguration) (bool, error) {

	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, "", "CA"}); err != nil {
		return false, err
	}

	caKeyPath := filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)
	if _, err := os.Stat(caKeyPath); !os.IsNotExist(err) {
		return false, nil
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, kubeadmconstants.APIServerCertAndKeyBaseName, "API server"}); err != nil {
		return true, err
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName, "API server kubelet client"}); err != nil {
		return true, err
	}

	return true, nil
}

// UsingExternalFrontProxyCA determines whether the user is relying on an external front-proxy CA.  We currently implicitly determine this is the case
// when the front proxy CA Cert is present but the front proxy CA Key is not.
// In case we are using an external front-proxy CA, the function validates the certificates signed by front-proxy CA that should be provided by the user.
func UsingExternalFrontProxyCA(cfg *kubeadmapi.ClusterConfiguration) (bool, error) {

	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, "", "front-proxy CA"}); err != nil {
		return false, err
	}

	frontProxyCAKeyPath := filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCAKeyName)
	if _, err := os.Stat(frontProxyCAKeyPath); !os.IsNotExist(err) {
		return false, nil
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, kubeadmconstants.FrontProxyClientCertAndKeyBaseName, "front-proxy client"}); err != nil {
		return true, err
	}

	return true, nil
}

// UsingExternalEtcdCA determines whether the user is relying on an external etcd CA. We currently implicitly determine this is the case
// when the etcd CA Cert is present but the etcd CA Key is not.
// In case we are using an external etcd CA, the function validates the certificates signed by etcd CA that should be provided by the user.
func UsingExternalEtcdCA(cfg *kubeadmapi.ClusterConfiguration) (bool, error) {
	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, "", "etcd CA"}); err != nil {
		return false, err
	}

	path := filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCAKeyName)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return false, nil
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName, "apiserver etcd client"}); err != nil {
		return true, err
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, kubeadmconstants.EtcdServerCertAndKeyBaseName, "etcd server"}); err != nil {
		return true, err
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, kubeadmconstants.EtcdPeerCertAndKeyBaseName, "etcd peer"}); err != nil {
		return true, err
	}

	if err := validateSignedCert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName, kubeadmconstants.EtcdHealthcheckClientCertAndKeyBaseName, "etcd health-check client"}); err != nil {
		return true, err
	}

	return true, nil
}

// validateCACert tries to load a x509 certificate from pkiDir and validates that it is a CA
func validateCACert(l certKeyLocation) error {
	// Check CA Cert
	caCert, err := pkiutil.TryLoadCertFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		return errors.Wrapf(err, "failure loading certificate for %s", l.uxName)
	}
	// Validate period
	CheckCertificatePeriodValidity(l.uxName, caCert)

	// Check if cert is a CA
	if !caCert.IsCA {
		return errors.Errorf("certificate %s is not a CA", l.uxName)
	}
	return nil
}

// validateCACertAndKey tries to load a x509 certificate and private key from pkiDir,
// and validates that the cert is a CA. Failure to load the key produces a warning.
func validateCACertAndKey(l certKeyLocation) error {
	if err := validateCACert(l); err != nil {
		return err
	}

	_, err := pkiutil.TryLoadKeyFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		klog.Warningf("assuming external key for %s: %v", l.uxName, err)
	}
	return nil
}

// validateSignedCert tries to load a x509 certificate and private key from pkiDir and validates
// that the cert is signed by a given CA
func validateSignedCert(l certKeyLocation) error {
	// Try to load CA
	caCert, err := pkiutil.TryLoadCertFromDisk(l.pkiDir, l.caBaseName)
	if err != nil {
		return errors.Wrapf(err, "failure loading certificate authority for %s", l.uxName)
	}
	// Validate period
	CheckCertificatePeriodValidity(l.uxName, caCert)

	return validateSignedCertWithCA(l, caCert)
}

// validateSignedCertWithCA tries to load a certificate and private key and
// validates that the cert is signed by the given caCert
func validateSignedCertWithCA(l certKeyLocation, caCert *x509.Certificate) error {
	// Try to load key from the PKI directory
	_, err := pkiutil.TryLoadKeyFromDisk(l.pkiDir, l.baseName)
	if err != nil {
		return errors.Wrapf(err, "failure loading key for %s", l.baseName)
	}

	// Try to load certificate from the PKI directory
	signedCert, intermediates, err := pkiutil.TryLoadCertChainFromDisk(l.pkiDir, l.baseName)
	if err != nil {
		return errors.Wrapf(err, "failure loading certificate for %s", l.uxName)
	}
	// Validate period
	CheckCertificatePeriodValidity(l.uxName, signedCert)

	// Check if the cert is signed by the CA
	if err := pkiutil.VerifyCertChain(signedCert, intermediates, caCert); err != nil {
		return errors.Wrapf(err, "certificate %s is not signed by corresponding CA", l.uxName)
	}
	return nil
}

// validatePrivatePublicKey tries to load a private key from pkiDir
func validatePrivatePublicKey(l certKeyLocation) error {
	// Try to load key
	_, _, err := pkiutil.TryLoadPrivatePublicKeyFromDisk(l.pkiDir, l.baseName)
	return errors.Wrapf(err, "failure loading key for %s", l.uxName)
}

// validateCertificateWithConfig makes sure that a given certificate is valid at
// least for the SANs defined in the configuration.
func validateCertificateWithConfig(cert *x509.Certificate, baseName string, cfg *pkiutil.CertConfig) error {
	for _, dnsName := range cfg.AltNames.DNSNames {
		if err := cert.VerifyHostname(dnsName); err != nil {
			return errors.Wrapf(err, "certificate %s is invalid", baseName)
		}
	}
	for _, ipAddress := range cfg.AltNames.IPs {
		if err := cert.VerifyHostname(ipAddress.String()); err != nil {
			return errors.Wrapf(err, "certificate %s is invalid", baseName)
		}
	}
	return nil
}

// CheckCertificatePeriodValidity takes a certificate and prints a warning if its period
// is not valid related to the current time. It does so only if the certificate was not validated already
// by keeping track with a cache.
func CheckCertificatePeriodValidity(baseName string, cert *x509.Certificate) {
	certPeriodValidationMutex.Lock()
	defer certPeriodValidationMutex.Unlock()
	if _, exists := certPeriodValidation[baseName]; exists {
		return
	}
	certPeriodValidation[baseName] = struct{}{}

	klog.V(5).Infof("validating certificate period for %s certificate", baseName)
	if err := pkiutil.ValidateCertPeriod(cert, 0); err != nil {
		klog.Warningf("WARNING: could not validate bounds for certificate %s: %v", baseName, err)
	}
}
