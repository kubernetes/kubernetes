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

package kubeconfig

import (
	"bytes"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// TODO: Make an integration test for this function that runs after the certificates phase
// and makes sure that those two phases work well together...

// TODO: Integration test cases:
// /etc/kubernetes/{admin,kubelet}.conf don't exist => generate kubeconfig files
// /etc/kubernetes/{admin,kubelet}.conf and certs in /etc/kubernetes/pki exist => don't touch anything as long as everything's valid
// /etc/kubernetes/{admin,kubelet}.conf exist but the server URL is invalid in those files => error
// /etc/kubernetes/{admin,kubelet}.conf exist but the CA cert doesn't match what's in the pki dir => error
// /etc/kubernetes/{admin,kubelet}.conf exist but not certs => certs will be generated and conflict with the kubeconfig files => error

// CreateAdminAndKubeletKubeConfig is called from the main init and does the work for the default phase behaviour
func CreateAdminAndKubeletKubeConfig(masterEndpoint, pkiDir, outDir string) error {

	// Try to load ca.crt and ca.key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v", err)
	}

	// User admin should have full access to the cluster
	// TODO: Add test case that make sure this cert has the x509.ExtKeyUsageClientAuth flag
	adminCertConfig := certutil.Config{
		CommonName:   "kubernetes-admin",
		Organization: []string{"system:masters"},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	adminKubeConfigFilePath := filepath.Join(outDir, kubeadmconstants.AdminKubeConfigFileName)
	if err := createKubeConfigFileForClient(masterEndpoint, adminKubeConfigFilePath, adminCertConfig, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create config for the admin: %v", err)
	}

	// TODO: The kubelet should have limited access to the cluster. Right now, this gives kubelet basically root access
	// and we do need that in the bootstrap phase, but we should swap it out after the control plane is up
	// TODO: Add test case that make sure this cert has the x509.ExtKeyUsageClientAuth flag
	kubeletCertConfig := certutil.Config{
		CommonName:   "kubelet",
		Organization: []string{"system:nodes"},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	kubeletKubeConfigFilePath := filepath.Join(outDir, kubeadmconstants.KubeletKubeConfigFileName)
	if err := createKubeConfigFileForClient(masterEndpoint, kubeletKubeConfigFilePath, kubeletCertConfig, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create a kubeconfig file for the kubelet: %v", err)
	}

	// TODO make credentials for the controller-manager and scheduler
	return nil
}

func createKubeConfigFileForClient(masterEndpoint, kubeConfigFilePath string, config certutil.Config, caCert *x509.Certificate, caKey *rsa.PrivateKey) error {
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, config)
	if err != nil {
		return fmt.Errorf("failure while creating %s client certificate [%v]", config.CommonName, err)
	}

	kubeconfig := kubeconfigutil.CreateWithCerts(
		masterEndpoint,
		"kubernetes",
		config.CommonName,
		certutil.EncodeCertPEM(caCert),
		certutil.EncodePrivateKeyPEM(key),
		certutil.EncodeCertPEM(cert),
	)

	// Write it now to a file if there already isn't a valid one
	return writeKubeconfigToDiskIfNotExists(kubeConfigFilePath, kubeconfig)
}

// writeKubeconfigToDiskIfNotExists saves the KubeConfig struct to disk if there isn't any file at the given path
// If there already is a KubeConfig file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected config equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// but if a file exists but has old content or isn't a kubeconfig file, this function returns an error.
func writeKubeconfigToDiskIfNotExists(filename string, expectedConfig *clientcmdapi.Config) error {
	// Check if the file exist, and if it doesn't, just write it to disk
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return kubeconfigutil.WriteToDisk(filename, expectedConfig)
	}

	// The kubeconfig already exists, let's check if it has got the same CA and server URL
	currentConfig, err := clientcmd.LoadFromFile(filename)
	if err != nil {
		return fmt.Errorf("failed to load kubeconfig that already exists on disk [%v]", err)
	}

	expectedCtx := expectedConfig.CurrentContext
	expectedCluster := expectedConfig.Contexts[expectedCtx].Cluster
	currentCtx := currentConfig.CurrentContext
	currentCluster := currentConfig.Contexts[currentCtx].Cluster
	// If the current CA cert on disk doesn't match the expected CA cert, error out because we have a file, but it's stale
	if !bytes.Equal(currentConfig.Clusters[currentCluster].CertificateAuthorityData, expectedConfig.Clusters[expectedCluster].CertificateAuthorityData) {
		return fmt.Errorf("a kubeconfig file %q exists already but has got the wrong CA cert", filename)
	}
	// If the current API Server location on disk doesn't match the expected API server, error out because we have a file, but it's stale
	if currentConfig.Clusters[currentCluster].Server != expectedConfig.Clusters[expectedCluster].Server {
		return fmt.Errorf("a kubeconfig file %q exists already but has got the wrong API Server URL", filename)
	}

	// kubeadm doesn't validate the existing kubeconfig file more than this (kubeadm trusts the client certs to be valid)
	// Basically, if we find a kubeconfig file with the same path; the same CA cert and the same server URL;
	// kubeadm thinks those files are equal and doesn't bother writing a new file
	fmt.Printf("[kubeconfig] Using existing up-to-date KubeConfig file: %q\n", filename)

	return nil
}
