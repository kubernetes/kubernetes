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

// BuildConfigProperties holds some simple information about how this phase should build the KubeConfig object
type BuildConfigProperties struct {
	CertDir         string
	ClientName      string
	Organization    []string
	APIServer       string
	Token           string
	MakeClientCerts bool
}

// TODO: Make an integration test for this function that runs after the certificates phase
// and makes sure that those two phases work well together...

// TODO: Integration test cases:
// /etc/kubernetes/{admin,kubelet}.conf don't exist => generate kubeconfig files
// /etc/kubernetes/{admin,kubelet}.conf and certs in /etc/kubernetes/pki exist => don't touch anything as long as everything's valid
// /etc/kubernetes/{admin,kubelet}.conf exist but the server URL is invalid in those files => error
// /etc/kubernetes/{admin,kubelet}.conf exist but the CA cert doesn't match what's in the pki dir => error
// /etc/kubernetes/{admin,kubelet}.conf exist but not certs => certs will be generated and conflict with the kubeconfig files => error

// CreateInitKubeConfigFiles is called from the main init and does the work for the default phase behaviour
func CreateInitKubeConfigFiles(masterEndpoint, pkiDir, outDir string) error {

	hostname, err := os.Hostname()
	if err != nil {
		return err
	}

	// Create a lightweight specification for what the files should look like
	filesToCreateFromSpec := map[string]BuildConfigProperties{
		kubeadmconstants.AdminKubeConfigFileName: {
			ClientName:      "kubernetes-admin",
			APIServer:       masterEndpoint,
			CertDir:         pkiDir,
			Organization:    []string{kubeadmconstants.MastersGroup},
			MakeClientCerts: true,
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			ClientName:      fmt.Sprintf("system:node:%s", hostname),
			APIServer:       masterEndpoint,
			CertDir:         pkiDir,
			Organization:    []string{kubeadmconstants.NodesGroup},
			MakeClientCerts: true,
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			ClientName:      kubeadmconstants.ControllerManagerUser,
			APIServer:       masterEndpoint,
			CertDir:         pkiDir,
			MakeClientCerts: true,
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			ClientName:      kubeadmconstants.SchedulerUser,
			APIServer:       masterEndpoint,
			CertDir:         pkiDir,
			MakeClientCerts: true,
		},
	}

	// Loop through all specs for kubeconfig files and create them if necessary
	for filename, config := range filesToCreateFromSpec {
		kubeconfig, err := buildKubeConfig(config)
		if err != nil {
			return err
		}

		kubeConfigFilePath := filepath.Join(outDir, filename)
		err = writeKubeconfigToDiskIfNotExists(kubeConfigFilePath, kubeconfig)
		if err != nil {
			return err
		}
	}

	return nil
}

// GetKubeConfigBytesFromSpec takes properties how to build a KubeConfig file and then returns the bytes of that file
func GetKubeConfigBytesFromSpec(config BuildConfigProperties) ([]byte, error) {
	kubeconfig, err := buildKubeConfig(config)
	if err != nil {
		return []byte{}, err
	}

	kubeConfigBytes, err := clientcmd.Write(*kubeconfig)
	if err != nil {
		return []byte{}, err
	}
	return kubeConfigBytes, nil
}

// buildKubeConfig creates a kubeconfig object from some commonly specified properties in the struct above
func buildKubeConfig(config BuildConfigProperties) (*clientcmdapi.Config, error) {

	// Try to load ca.crt and ca.key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(config.CertDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return nil, fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v", err)
	}

	// If this file should have client certs, generate one from the spec
	if config.MakeClientCerts {
		certConfig := certutil.Config{
			CommonName:   config.ClientName,
			Organization: config.Organization,
			Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}
		cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, certConfig)
		if err != nil {
			return nil, fmt.Errorf("failure while creating %s client certificate [%v]", certConfig.CommonName, err)
		}
		return kubeconfigutil.CreateWithCerts(
			config.APIServer,
			"kubernetes",
			config.ClientName,
			certutil.EncodeCertPEM(caCert),
			certutil.EncodePrivateKeyPEM(key),
			certutil.EncodeCertPEM(cert),
		), nil
	}

	// otherwise, create a kubeconfig with a token
	return kubeconfigutil.CreateWithToken(
		config.APIServer,
		"kubernetes",
		config.ClientName,
		certutil.EncodeCertPEM(caCert),
		config.Token,
	), nil
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
