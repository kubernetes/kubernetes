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
	certActions := []func(cfg *kubeadmapi.InitConfiguration) error{
		CreateCACertAndKeyFiles,
		CreateAPIServerCertAndKeyFiles,
		CreateAPIServerKubeletClientCertAndKeyFiles,
		CreateServiceAccountKeyAndPublicKeyFiles,
		CreateFrontProxyCACertAndKeyFiles,
		CreateFrontProxyClientCertAndKeyFiles,
	}
	etcdCertActions := []func(cfg *kubeadmapi.InitConfiguration) error{
		CreateEtcdCACertAndKeyFiles,
		CreateEtcdServerCertAndKeyFiles,
		CreateEtcdPeerCertAndKeyFiles,
		CreateEtcdHealthcheckClientCertAndKeyFiles,
		CreateAPIServerEtcdClientCertAndKeyFiles,
	}

	if cfg.Etcd.Local != nil {
		certActions = append(certActions, etcdCertActions...)
	}

	for _, action := range certActions {
		err := action(cfg)
		if err != nil {
			return err
		}
	}

	fmt.Printf("[certificates] valid certificates and keys now exist in %q\n", cfg.CertificatesDir)

	return nil
}

// CreateCACertAndKeyFiles create a new self signed cluster CA certificate and key files.
// If the CA certificate and key files already exists in the target folder, they are used only if evaluated equal; otherwise an error is returned.
func CreateCACertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("create a new self signed cluster CA certificate and key files")
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
// It assumes the cluster CA certificate and key files exist in the CertificatesDir.
func CreateAPIServerCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new certificate and key files for the apiserver")
	caCert, caKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
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

// CreateAPIServerKubeletClientCertAndKeyFiles create a new certificate for kubelets calling apiserver.
// If the apiserver-kubelet-client certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
// It assumes the cluster CA certificate and key files exist in the CertificatesDir.
func CreateAPIServerKubeletClientCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new certificate for kubelets calling apiserver")
	caCert, caKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return err
	}

	apiKubeletClientCert, apiKubeletClientKey, err := NewAPIServerKubeletClientCertAndKey(caCert, caKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
		caCert,
		apiKubeletClientCert,
		apiKubeletClientKey,
	)
}

// CreateEtcdCACertAndKeyFiles create a self signed etcd CA certificate and key files.
// The etcd CA and client certs are used to secure communication between etcd peers and connections to etcd from the API server.
// This is a separate CA, so that kubernetes client identities cannot connect to etcd directly or peer with the etcd cluster.
// If the etcd CA certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateEtcdCACertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a self signed etcd CA certificate and key files")
	etcdCACert, etcdCAKey, err := NewEtcdCACertAndKey()
	if err != nil {
		return err
	}

	return writeCertificateAuthorithyFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.EtcdCACertAndKeyBaseName,
		etcdCACert,
		etcdCAKey,
	)
}

// CreateEtcdServerCertAndKeyFiles create a new certificate and key file for etcd.
// If the etcd serving certificate and key file already exist in the target folder, they are used only if evaluated equal; otherwise an error is returned.
// It assumes the etcd CA certificate and key file exist in the CertificatesDir
func CreateEtcdServerCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new server certificate and key files for etcd")
	etcdCACert, etcdCAKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName)
	if err != nil {
		return err
	}

	etcdServerCert, etcdServerKey, err := NewEtcdServerCertAndKey(cfg, etcdCACert, etcdCAKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.EtcdServerCertAndKeyBaseName,
		etcdCACert,
		etcdServerCert,
		etcdServerKey,
	)
}

// CreateEtcdPeerCertAndKeyFiles create a new certificate and key file for etcd peering.
// If the etcd peer certificate and key file already exist in the target folder, they are used only if evaluated equal; otherwise an error is returned.
// It assumes the etcd CA certificate and key file exist in the CertificatesDir
func CreateEtcdPeerCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new certificate and key files for etcd peering")
	etcdCACert, etcdCAKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName)
	if err != nil {
		return err
	}

	etcdPeerCert, etcdPeerKey, err := NewEtcdPeerCertAndKey(cfg, etcdCACert, etcdCAKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.EtcdPeerCertAndKeyBaseName,
		etcdCACert,
		etcdPeerCert,
		etcdPeerKey,
	)
}

// CreateEtcdHealthcheckClientCertAndKeyFiles create a new client certificate for liveness probes to healthcheck etcd
// If the etcd-healthcheck-client certificate and key file already exist in the target folder, they are used only if evaluated equal; otherwise an error is returned.
// It assumes the etcd CA certificate and key file exist in the CertificatesDir
func CreateEtcdHealthcheckClientCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {

	etcdCACert, etcdCAKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName)
	if err != nil {
		return err
	}

	etcdHealthcheckClientCert, etcdHealthcheckClientKey, err := NewEtcdHealthcheckClientCertAndKey(etcdCACert, etcdCAKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.EtcdHealthcheckClientCertAndKeyBaseName,
		etcdCACert,
		etcdHealthcheckClientCert,
		etcdHealthcheckClientKey,
	)
}

// CreateAPIServerEtcdClientCertAndKeyFiles create a new client certificate for the apiserver calling etcd
// If the apiserver-etcd-client certificate and key file already exist in the target folder, they are used only if evaluated equal; otherwise an error is returned.
// It assumes the etcd CA certificate and key file exist in the CertificatesDir
func CreateAPIServerEtcdClientCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new client certificate for the apiserver calling etcd")
	etcdCACert, etcdCAKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.EtcdCACertAndKeyBaseName)
	if err != nil {
		return err
	}

	apiEtcdClientCert, apiEtcdClientKey, err := NewAPIServerEtcdClientCertAndKey(etcdCACert, etcdCAKey)
	if err != nil {
		return err
	}

	return writeCertificateFilesIfNotExist(
		cfg.CertificatesDir,
		kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName,
		etcdCACert,
		apiEtcdClientCert,
		apiEtcdClientKey,
	)
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

// CreateFrontProxyCACertAndKeyFiles create a self signed front proxy CA certificate and key files.
// Front proxy CA and client certs are used to secure a front proxy authenticator which is used to assert identity
// without the client cert; This is a separate CA, so that front proxy identities cannot hit the API and normal client certs cannot be used
// as front proxies.
// If the front proxy CA certificate and key files already exists in the target folder, they are used only if evaluated equals; otherwise an error is returned.
func CreateFrontProxyCACertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a self signed front proxy CA certificate and key files")
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
// It assumes the front proxy CA certificate and key files exist in the CertificatesDir.
func CreateFrontProxyClientCertAndKeyFiles(cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating a new certificate for proxy server client")
	frontProxyCACert, frontProxyCAKey, err := loadCertificateAuthority(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertAndKeyBaseName)
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

// NewAPIServerCertAndKey generate certificate for apiserver, signed by the given CA.
func NewAPIServerCertAndKey(cfg *kubeadmapi.InitConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	altNames, err := pkiutil.GetAPIServerAltNames(cfg)
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

// NewAPIServerKubeletClientCertAndKey generate certificate for the apiservers to connect to the kubelets securely, signed by the given CA.
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

// NewEtcdCACertAndKey generate a self signed etcd CA.
func NewEtcdCACertAndKey() (*x509.Certificate, *rsa.PrivateKey, error) {

	etcdCACert, etcdCAKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		return nil, nil, fmt.Errorf("failure while generating etcd CA certificate and key: %v", err)
	}

	return etcdCACert, etcdCAKey, nil
}

// NewEtcdServerCertAndKey generate certificate for etcd, signed by the given CA.
func NewEtcdServerCertAndKey(cfg *kubeadmapi.InitConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	altNames, err := pkiutil.GetEtcdAltNames(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while composing altnames for etcd: %v", err)
	}

	// TODO: etcd 3.2 introduced an undocumented requirement for ClientAuth usage on the
	// server cert: https://github.com/coreos/etcd/issues/9785#issuecomment-396715692
	// Once the upstream issue is resolved, this should be returned to only allowing
	// ServerAuth usage.
	config := certutil.Config{
		CommonName: cfg.NodeRegistration.Name,
		AltNames:   *altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	etcdServerCert, etcdServerKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating etcd key and certificate: %v", err)
	}

	return etcdServerCert, etcdServerKey, nil
}

// NewEtcdPeerCertAndKey generate certificate for etcd peering, signed by the given CA.
func NewEtcdPeerCertAndKey(cfg *kubeadmapi.InitConfiguration, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	altNames, err := pkiutil.GetEtcdPeerAltNames(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while composing altnames for etcd peering: %v", err)
	}

	config := certutil.Config{
		CommonName: cfg.NodeRegistration.Name,
		AltNames:   *altNames,
		Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	etcdPeerCert, etcdPeerKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating etcd peer key and certificate: %v", err)
	}

	return etcdPeerCert, etcdPeerKey, nil
}

// NewEtcdHealthcheckClientCertAndKey generate certificate for liveness probes to healthcheck etcd, signed by the given CA.
func NewEtcdHealthcheckClientCertAndKey(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName:   kubeadmconstants.EtcdHealthcheckClientCertCommonName,
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	etcdHealcheckClientCert, etcdHealcheckClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating etcd healthcheck client key and certificate: %v", err)
	}

	return etcdHealcheckClientCert, etcdHealcheckClientKey, nil
}

// NewAPIServerEtcdClientCertAndKey generate certificate for the apiservers to connect to etcd securely, signed by the given CA.
func NewAPIServerEtcdClientCertAndKey(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	config := certutil.Config{
		CommonName:   kubeadmconstants.APIServerEtcdClientCertCommonName,
		Organization: []string{kubeadmconstants.MastersGroup},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	apiClientCert, apiClientKey, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return nil, nil, fmt.Errorf("failure while creating API server etcd client key and certificate: %v", err)
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

// NewFrontProxyClientCertAndKey generate certificate for proxy server client, signed by the given front proxy CA.
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

// loadCertificateAuthority loads certificate authority
func loadCertificateAuthority(pkiDir string, baseName string) (*x509.Certificate, *rsa.PrivateKey, error) {
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
