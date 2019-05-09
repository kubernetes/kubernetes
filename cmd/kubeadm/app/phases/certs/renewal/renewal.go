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
	"path/filepath"

	"github.com/pkg/errors"
	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
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

// RenewEmbeddedClientCert loads a kubeconfig file, uses the renew interface to renew the client certificate
// embedded in it, and then saves the resulting kubeconfig and key over the old one.
func RenewEmbeddedClientCert(kubeConfigFileDir, kubeConfigFileName string, impl Interface) error {
	kubeConfigFilePath := filepath.Join(kubeConfigFileDir, kubeConfigFileName)

	// try to load the kubeconfig file
	kubeconfig, err := clientcmd.LoadFromFile(kubeConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to load kubeconfig file %s", kubeConfigFilePath)
	}

	// get current context
	if _, ok := kubeconfig.Contexts[kubeconfig.CurrentContext]; !ok {
		return errors.Errorf("invalid kubeconfig file %s: missing context %s", kubeConfigFilePath, kubeconfig.CurrentContext)
	}

	// get cluster info for current context and ensure a server certificate is embedded in it
	clusterName := kubeconfig.Contexts[kubeconfig.CurrentContext].Cluster
	if _, ok := kubeconfig.Clusters[clusterName]; !ok {
		return errors.Errorf("invalid kubeconfig file %s: missing cluster %s", kubeConfigFilePath, clusterName)
	}

	cluster := kubeconfig.Clusters[clusterName]
	if len(cluster.CertificateAuthorityData) == 0 {
		return errors.Errorf("kubeconfig file %s does not have and embedded server certificate", kubeConfigFilePath)
	}

	// get auth info for current context and ensure a client certificate is embedded in it
	authInfoName := kubeconfig.Contexts[kubeconfig.CurrentContext].AuthInfo
	if _, ok := kubeconfig.AuthInfos[authInfoName]; !ok {
		return errors.Errorf("invalid kubeconfig file %s: missing authInfo %s", kubeConfigFilePath, authInfoName)
	}

	authInfo := kubeconfig.AuthInfos[authInfoName]
	if len(authInfo.ClientCertificateData) == 0 {
		return errors.Errorf("kubeconfig file %s does not have and embedded client certificate", kubeConfigFilePath)
	}

	// parse the client certificate, retrive the cert config and then renew it
	certs, err := certutil.ParseCertsPEM(authInfo.ClientCertificateData)
	if err != nil {
		return errors.Wrapf(err, "kubeconfig file %s does not contain a valid client certificate", kubeConfigFilePath)
	}

	cfg := certToConfig(certs[0])

	newCert, newKey, err := impl.Renew(cfg)
	if err != nil {
		return errors.Wrapf(err, "failed to renew certificate embedded in %s", kubeConfigFilePath)
	}

	// encodes the new key
	encodedClientKey, err := keyutil.MarshalPrivateKeyToPEM(newKey)
	if err != nil {
		return errors.Wrapf(err, "failed to marshal private key to PEM")
	}

	// create a kubeconfig copy with the new client certs
	newConfig := kubeconfig.DeepCopy()
	newConfig.AuthInfos[authInfoName].ClientKeyData = encodedClientKey
	newConfig.AuthInfos[authInfoName].ClientCertificateData = pkiutil.EncodeCertPEM(newCert)

	// writes the kubeconfig to disk
	return clientcmd.WriteToFile(*newConfig, kubeConfigFilePath)
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
