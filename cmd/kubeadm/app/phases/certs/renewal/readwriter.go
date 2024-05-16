/*
Copyright 2019 The Kubernetes Authors.

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
	"crypto"
	"crypto/x509"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// certificateReadWriter defines the behavior of a component that
// read or write a certificate stored/embedded in a file
type certificateReadWriter interface {
	//Exists return true if the certificate exists
	Exists() (bool, error)

	// Read a certificate stored/embedded in a file
	Read() (*x509.Certificate, error)

	// Write (update) a certificate stored/embedded in a file
	Write(*x509.Certificate, crypto.Signer) error
}

// pkiCertificateReadWriter defines a certificateReadWriter for certificate files
// in the K8s pki managed by kubeadm
type pkiCertificateReadWriter struct {
	baseName       string
	certificateDir string
}

// newPKICertificateReadWriter return a new pkiCertificateReadWriter
func newPKICertificateReadWriter(certificateDir string, baseName string) *pkiCertificateReadWriter {
	return &pkiCertificateReadWriter{
		baseName:       baseName,
		certificateDir: certificateDir,
	}
}

// Exists checks if a certificate exist
func (rw *pkiCertificateReadWriter) Exists() (bool, error) {
	certificatePath, _ := pkiutil.PathsForCertAndKey(rw.certificateDir, rw.baseName)
	return fileExists(certificatePath)
}

func fileExists(filename string) (bool, error) {
	info, err := os.Stat(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return !info.IsDir(), nil
}

// Read a certificate from a file the K8s pki managed by kubeadm
func (rw *pkiCertificateReadWriter) Read() (*x509.Certificate, error) {
	certificatePath, _ := pkiutil.PathsForCertAndKey(rw.certificateDir, rw.baseName)
	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load existing certificate %s", rw.baseName)
	}

	// Safely pick the first one because the sender's certificate must come first in the list.
	// For details, see: https://www.rfc-editor.org/rfc/rfc4346#section-7.4.2
	return certs[0], nil
}

// Write a certificate to files in the K8s pki managed by kubeadm
func (rw *pkiCertificateReadWriter) Write(newCert *x509.Certificate, newKey crypto.Signer) error {
	if err := pkiutil.WriteCertAndKey(rw.certificateDir, rw.baseName, newCert, newKey); err != nil {
		return errors.Wrapf(err, "failed to write new certificate %s", rw.baseName)
	}
	return nil
}

// kubeConfigReadWriter defines a certificateReadWriter for certificate files
// embedded in the kubeConfig files managed by kubeadm, and more specifically
// for the client certificate of the AuthInfo
type kubeConfigReadWriter struct {
	kubernetesDir      string
	kubeConfigFileName string
	kubeConfigFilePath string
	kubeConfig         *clientcmdapi.Config
	baseName           string
	certificateDir     string
	caCert             *x509.Certificate
}

// newKubeconfigReadWriter return a new kubeConfigReadWriter
func newKubeconfigReadWriter(kubernetesDir string, kubeConfigFileName string, certificateDir, baseName string) *kubeConfigReadWriter {
	return &kubeConfigReadWriter{
		kubernetesDir:      kubernetesDir,
		kubeConfigFileName: kubeConfigFileName,
		kubeConfigFilePath: filepath.Join(kubernetesDir, kubeConfigFileName),
		certificateDir:     certificateDir,
		baseName:           baseName,
	}
}

// Exists checks if a certificate embedded in kubeConfig file exists
func (rw *kubeConfigReadWriter) Exists() (bool, error) {
	return fileExists(rw.kubeConfigFilePath)
}

// Read a certificate embedded in kubeConfig file managed by kubeadm.
// Please note that the kubeConfig file itself is kept in the ReadWriter state thus allowing
// to preserve the attributes (Context, Servers, AuthInfo etc.)
func (rw *kubeConfigReadWriter) Read() (*x509.Certificate, error) {
	// try to load the kubeConfig file
	kubeConfig, err := clientcmd.LoadFromFile(rw.kubeConfigFilePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load kubeConfig file %s", rw.kubeConfigFilePath)
	}

	// The CA cert is required for updating kubeconfig files.
	// For local CA renewal, the local CA on disk could have changed, thus a reload is needed.
	// For CSR renewal we assume the same CA on disk is mounted for usage with KCM's
	// '--cluster-signing-cert-file' flag.
	certificatePath, _ := pkiutil.PathsForCertAndKey(rw.certificateDir, rw.baseName)
	caCerts, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load existing certificate %s", rw.baseName)
	}

	// Safely pick the first one because the sender's certificate must come first in the list.
	// For details, see: https://www.rfc-editor.org/rfc/rfc4346#section-7.4.2
	rw.caCert = caCerts[0]

	// get current context
	if _, ok := kubeConfig.Contexts[kubeConfig.CurrentContext]; !ok {
		return nil, errors.Errorf("invalid kubeConfig file %s: missing context %s", rw.kubeConfigFilePath, kubeConfig.CurrentContext)
	}

	// get cluster info for current context and ensure a server certificate is embedded in it
	clusterName := kubeConfig.Contexts[kubeConfig.CurrentContext].Cluster
	if _, ok := kubeConfig.Clusters[clusterName]; !ok {
		return nil, errors.Errorf("invalid kubeConfig file %s: missing cluster %s", rw.kubeConfigFilePath, clusterName)
	}

	cluster := kubeConfig.Clusters[clusterName]
	if len(cluster.CertificateAuthorityData) == 0 {
		return nil, errors.Errorf("kubeConfig file %s does not have an embedded server certificate", rw.kubeConfigFilePath)
	}

	// get auth info for current context and ensure a client certificate is embedded in it
	authInfoName := kubeConfig.Contexts[kubeConfig.CurrentContext].AuthInfo
	if _, ok := kubeConfig.AuthInfos[authInfoName]; !ok {
		return nil, errors.Errorf("invalid kubeConfig file %s: missing authInfo %s", rw.kubeConfigFilePath, authInfoName)
	}

	authInfo := kubeConfig.AuthInfos[authInfoName]
	if len(authInfo.ClientCertificateData) == 0 {
		return nil, errors.Errorf("kubeConfig file %s does not have an embedded client certificate", rw.kubeConfigFilePath)
	}

	// parse the client certificate, retrieve the cert config and then renew it
	certs, err := certutil.ParseCertsPEM(authInfo.ClientCertificateData)
	if err != nil {
		return nil, errors.Wrapf(err, "kubeConfig file %s does not contain a valid client certificate", rw.kubeConfigFilePath)
	}

	rw.kubeConfig = kubeConfig

	return certs[0], nil
}

// Write a certificate embedded in kubeConfig file managed by kubeadm
// Please note that all the other attribute of the kubeConfig file are preserved, but this
// requires to call Read before Write
func (rw *kubeConfigReadWriter) Write(newCert *x509.Certificate, newKey crypto.Signer) error {
	// check if Read was called before Write
	if rw.kubeConfig == nil {
		return errors.Errorf("failed to Write kubeConfig file with renewed certs. It is necessary to call Read before Write")
	}

	// encodes the new key
	encodedClientKey, err := keyutil.MarshalPrivateKeyToPEM(newKey)
	if err != nil {
		return errors.Wrapf(err, "failed to marshal private key to PEM")
	}

	// Update the embedded CA in the kubeconfig file.
	// This assumes that the user has kept the current context to the desired one.
	clusterName := rw.kubeConfig.Contexts[rw.kubeConfig.CurrentContext].Cluster
	cluster := rw.kubeConfig.Clusters[clusterName]
	cluster.CertificateAuthorityData = pkiutil.EncodeCertPEM(rw.caCert)

	// get auth info for current context and ensure a client certificate is embedded in it
	authInfoName := rw.kubeConfig.Contexts[rw.kubeConfig.CurrentContext].AuthInfo

	// create a kubeConfig copy with the new client certs
	newConfig := rw.kubeConfig.DeepCopy()
	newConfig.AuthInfos[authInfoName].ClientKeyData = encodedClientKey
	newConfig.AuthInfos[authInfoName].ClientCertificateData = pkiutil.EncodeCertPEM(newCert)

	// writes the kubeConfig to disk
	return clientcmd.WriteToFile(*newConfig, rw.kubeConfigFilePath)
}
