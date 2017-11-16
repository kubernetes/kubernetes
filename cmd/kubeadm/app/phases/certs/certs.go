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
	"net"
	"os"
	"path/filepath"

	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// If the PKI assets already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration) error {

	certActions := []func(cfg *kubeadmapi.MasterConfiguration) error{
		CreateCACertAndKeyfiles,
		CreateAPIServerCertAndKeyFiles,
		CreateAPIServerKubeletClientCertAndKeyFiles,
		CreateServiceAccountKeyAndPublicKeyFiles,
		CreateFrontProxyCACertAndKeyFiles,
		CreateFrontProxyClientCertAndKeyFiles,
	}

	for _, action := range certActions {
		err := action(cfg)
		if err != nil {
			return err
		}
	}

	fmt.Printf("[certificates] Valid certificates and keys now exist in %q\n", cfg.CertificatesDir)

	return nil
}

// CreateCACertAndKeyfiles create a new self signed CA certificate and key files.
// If the CA certificate and key files already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
func CreateCACertAndKeyfiles(cfg *kubeadmapi.MasterConfiguration) error {

	caCert, caKey, err := NewCACertAndKey()
	if err != nil {
		return err
	}

	return writeCertificateAuthorithyFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.CACertAndKeyBaseName,
		caCert,
		caKey,
	)
}

// CreateAPIServerCertAndKeyFiles create a new certificate and key files for the apiserver.
// If the apiserver certificate and key files already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
// It assumes the cluster CA certificate and key files should exists into the CertificatesDir
func CreateAPIServerCertAndKeyFiles(cfg *kubeadmapi.MasterConfiguration) error {

	caCert, caKey, err := loadCertificateAuthorithy(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return err
	}

	apiCert, apiKey, err := NewAPIServerCertAndKey(cfg, caCert, caKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.APIServerCertAndKeyBaseName,
		caCert,
		apiCert,
		apiKey,
	)
}

// CreateAPIServerKubeletClientCertAndKeyFiles create a new CA certificate for kubelets calling apiserver
// If the apiserver-kubelet-client certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
// It assumes the cluster CA certificate and key files should exists into the CertificatesDir
func CreateAPIServerKubeletClientCertAndKeyFiles(cfg *kubeadmapi.MasterConfiguration) error {

	caCert, caKey, err := loadCertificateAuthorithy(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return err
	}

	apiClientCert, apiClientKey, err := NewAPIServerKubeletClientCertAndKey(caCert, caKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
		caCert,
		apiClientCert,
		apiClientKey,
	)
}

// CreateServiceAccountKeyAndPublicKeyFiles create a new public/private key files for signing service account users.
// If the sa public/private key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateServiceAccountKeyAndPublicKeyFiles(cfg *kubeadmapi.MasterConfiguration) error {

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

// CreateFrontProxyCACertAndKeyFiles create a self signed front proxy CA certificate and key files.
// Front proxy CA and client certs are used to secure a front proxy authenticator which is used to assert identity
// without the client cert; This is a separte CA, so that front proxy identities cannot hit the API and normal client certs cannot be used
// as front proxies.
// If the front proxy CA certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateFrontProxyCACertAndKeyFiles(cfg *kubeadmapi.MasterConfiguration) error {

	frontProxyCACert, frontProxyCAKey, err := NewFrontProxyCACertAndKey()
	if err != nil {
		return err
	}

	return writeCertificateAuthorithyFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		frontProxyCACert,
		frontProxyCAKey,
	)
}

// CreateFrontProxyClientCertAndKeyFiles create a new certificate for proxy server client.
// If the front-proxy-client certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
// It assumes the front proxy CAA certificate and key files should exists into the CertificatesDir
func CreateFrontProxyClientCertAndKeyFiles(cfg *kubeadmapi.MasterConfiguration) error {

	frontProxyCACert, frontProxyCAKey, err := loadCertificateAuthorithy(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName)
	if err != nil {
		return err
	}

	frontProxyClientCert, frontProxyClientKey, err := NewFrontProxyClientCertAndKey(frontProxyCACert, frontProxyCAKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
		frontProxyCACert,
		frontProxyClientCert,
		frontProxyClientKey,
	)
}

// NewCACertAndKey will generate a self signed CA.
func NewCACertAndKey() (*x509.Certificate, *rsa.PrivateKey, error) {

	caCert, caKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating CA certificate and key: %v", err)
	}

	return caCert, caKey, nil
}

// NewAPIServerCertAndKey generate CA certificate for apiserver, signed by the given CA.
func NewAPIServerCertAndKey(cfg *kubeadmapi.MasterConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	altNames, err := getAltNames(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while composing altnames for API server: %v", err)
	}

	config := certutil.Config{
		CommonName: kubeadmconstants.APIServerCertCommonName,
		AltNames:   *altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	apiCert, apiKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating API server key and certificate: %v", err)
	}

	return apiCert, apiKey, nil
}

// NewAPIServerKubeletClientCertAndKey generate CA certificate for the apiservers to connect to the kubelets securely, signed by the given CA.
func NewAPIServerKubeletClientCertAndKey(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName:   kubeadmconstants.APIServerKubeletClientCertCommonName,
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating API server kubelet client key and certificate: %v", err)
	}

	return apiClientCert, apiClientKey, nil
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

// NewFrontProxyCACertAndKey generate a self signed front proxy CA.
func NewFrontProxyCACertAndKey() (*x509.Certificate, *rsa.PrivateKey, error) {

	frontProxyCACert, frontProxyCAKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating front-proxy CA certificate and key: %v", err)
	}

	return frontProxyCACert, frontProxyCAKey, nil
}

// NewFrontProxyClientCertAndKey generate CA certificate for proxy server client, signed by the given front proxy CA.
func NewFrontProxyClientCertAndKey(frontProxyCACert *x509.Certificate, frontProxyCAKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName: kubeadmconstants.FrontProxyClientCertCommonName,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	frontProxyClientCert, frontProxyClientKey, err := pkiutil.NewCertAndKey(frontProxyCACert, frontProxyCAKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating front-proxy client key and certificate: %v", err)
	}

	return frontProxyClientCert, frontProxyClientKey, nil
}

// loadCertificateAuthorithy loads certificate authorithy
func loadCertificateAuthorithy(pkiDir string, baseName string) (*x509.Certificate, *rsa.PrivateKey, error) {
	// Checks if certificate authorithy exists in the PKI directory
	if !pkiutil.CertOrKeyExist(pkiDir, baseName) {
		return nil, nil, fmt.Errorf("couldn't load %s certificate authorithy from %s", baseName, pkiDir)
	}

	// Try to load certificate authorithy .crt and .key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
	if err != nil {
		return nil, nil, fmt.Errorf("failure loading %s certificate authorithy: %v", baseName, err)
	}

	// Make sure the loaded CA cert actually is a CA
	if !caCert.IsCA {
		return nil, nil, fmt.Errorf("%s certificate is not a certificate authorithy", baseName)
	}

	return caCert, caKey, nil
}

// writeCertificateAuthorithyFilesIfNotExist write a new certificate Authorithy to the given path.
// If there already is a certificate file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected certificate equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
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

		// kubeadm doesn't validate the existing certificate Authorithy more than this;
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
		// the expected certificate authorithy, kubeadm thinks those files are equal and
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

// UsingExternalCA determines whether the user is relying on an external CA.  We currently implicitly determine this is the case when the CA Cert
// is present but the CA Key is not. This allows us to, e.g., skip generating certs or not start the csr signing controller.
func UsingExternalCA(cfg *kubeadmapi.MasterConfiguration) (bool, error) {

	if err := validateCACert(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName, "", "CA"}); err != nil {
		return false, err
	}

	caKeyPath := filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)
	if _, err := os.Stat(caKeyPath); !os.IsNotExist(err) {
		return false, fmt.Errorf("ca.key exists")
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

	if err := validateCACertAndKey(certKeyLocation{cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName, "", "front-proxy CA"}); err != nil {
		return false, err
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
		return fmt.Errorf("failure loading certificate authorithy for %s: %v", l.uxName, err)
	}

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

// getAltNames builds an AltNames object for to be used when generating apiserver certificate
func getAltNames(cfg *kubeadmapi.MasterConfiguration) (*certutil.AltNames, error) {

	// advertise address
	advertiseAddress := net.ParseIP(cfg.API.AdvertiseAddress)
	if advertiseAddress == nil {
		return nil, fmt.Errorf("error parsing API AdvertiseAddress %v: is not a valid textual representation of an IP address", cfg.API.AdvertiseAddress)
	}

	// internal IP address for the API server
	_, svcSubnet, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("error parsing CIDR %q: %v", cfg.Networking.ServiceSubnet, err)
	}

	internalAPIServerVirtualIP, err := ipallocator.GetIndexedIP(svcSubnet, 1)
	if err != nil {
		return nil, fmt.Errorf("unable to get first IP address from the given CIDR (%s): %v", svcSubnet.String(), err)
	}

	// create AltNames with defaults DNSNames/IPs
	altNames := &certutil.AltNames{
		DNSNames: []string{
			cfg.NodeName,
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			fmt.Sprintf("kubernetes.default.svc.%s", cfg.Networking.DNSDomain),
		},
		IPs: []net.IP{
			internalAPIServerVirtualIP,
			advertiseAddress,
		},
	}

	// adds additional SAN
	for _, altname := range cfg.APIServerCertSANs {
		if ip := net.ParseIP(altname); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		}
	}

	return altNames, nil
}
