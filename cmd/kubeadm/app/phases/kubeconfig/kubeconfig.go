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
	"io"
	"os"
	"path/filepath"

	"crypto/rsa"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// clientCertAuth struct holds info required to build a client certificate to provide authentication info in a kubeconfig object
type clientCertAuth struct {
	CaKey         *rsa.PrivateKey
	Organizations []string
}

// tokenAuth struct holds info required to use a token to provide authentication info in a kubeconfig object
type tokenAuth struct {
	Token string
}

// kubeConfigSpec struct holds info required to build a KubeConfig object
type kubeConfigSpec struct {
	CaCert         *x509.Certificate
	APIServer      string
	ClientName     string
	TokenAuth      *tokenAuth
	ClientCertAuth *clientCertAuth
}

// CreateInitKubeConfigFiles will create and write to disk all kubeconfig files necessary in the kubeadm init phase
// to establish the control plane, including also the admin kubeconfig file.
// If kubeconfig files already exists, they are used only if evaluated equal; otherwise an error is returned.
func CreateInitKubeConfigFiles(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createKubeConfigFiles(
		outDir,
		cfg,
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.KubeletKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
	)
}

// CreateAdminKubeConfigFile create a kubeconfig file for the admin to use and for kubeadm itself.
// If the kubeconfig file already exists, it is used only if evaluated equal; otherwise an error is returned.
func CreateAdminKubeConfigFile(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createKubeConfigFiles(outDir, cfg, kubeadmconstants.AdminKubeConfigFileName)
}

// CreateKubeletKubeConfigFile create a kubeconfig file for the Kubelet to use.
// If the kubeconfig file already exists, it is used only if evaluated equal; otherwise an error is returned.
func CreateKubeletKubeConfigFile(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createKubeConfigFiles(outDir, cfg, kubeadmconstants.KubeletKubeConfigFileName)
}

// CreateControllerManagerKubeConfigFile create a kubeconfig file for the ControllerManager to use.
// If the kubeconfig file already exists, it is used only if evaluated equal; otherwise an error is returned.
func CreateControllerManagerKubeConfigFile(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createKubeConfigFiles(outDir, cfg, kubeadmconstants.ControllerManagerKubeConfigFileName)
}

// CreateSchedulerKubeConfigFile create a create a kubeconfig file for the Scheduler to use.
// If the kubeconfig file already exists, it is used only if evaluated equal; otherwise an error is returned.
func CreateSchedulerKubeConfigFile(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createKubeConfigFiles(outDir, cfg, kubeadmconstants.SchedulerKubeConfigFileName)
}

// createKubeConfigFiles creates all the requested kubeconfig files.
// If kubeconfig files already exists, they are used only if evaluated equal; otherwise an error is returned.
func createKubeConfigFiles(outDir string, cfg *kubeadmapi.MasterConfiguration, kubeConfigFileNames ...string) error {

	// gets the KubeConfigSpecs, actualized for the current MasterConfiguration
	specs, err := getKubeConfigSpecs(cfg)
	if err != nil {
		return err
	}

	for _, kubeConfigFileName := range kubeConfigFileNames {
		// retrives the KubeConfigSpec for given kubeConfigFileName
		spec, exists := specs[kubeConfigFileName]
		if !exists {
			return fmt.Errorf("couldn't retrive KubeConfigSpec for %s", kubeConfigFileName)
		}

		// builds the KubeConfig object
		config, err := buildKubeConfigFromSpec(spec)
		if err != nil {
			return err
		}

		// writes the KubeConfig to disk if it not exists
		err = createKubeConfigFileIfNotExists(outDir, kubeConfigFileName, config)
		if err != nil {
			return err
		}
	}

	return nil
}

// getKubeConfigSpecs returns all KubeConfigSpecs actualized to the context of the current MasterConfiguration
// NB. this methods holds the information about how kubeadm creates kubeconfig files.
func getKubeConfigSpecs(cfg *kubeadmapi.MasterConfiguration) (map[string]*kubeConfigSpec, error) {

	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return nil, fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v", err)
	}

	var kubeConfigSpec = map[string]*kubeConfigSpec{
		kubeadmconstants.AdminKubeConfigFileName: {
			CaCert:     caCert,
			APIServer:  cfg.GetMasterEndpoint(),
			ClientName: "kubernetes-admin",
			ClientCertAuth: &clientCertAuth{
				CaKey:         caKey,
				Organizations: []string{kubeadmconstants.MastersGroup},
			},
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			CaCert:     caCert,
			APIServer:  cfg.GetMasterEndpoint(),
			ClientName: fmt.Sprintf("system:node:%s", cfg.NodeName),
			ClientCertAuth: &clientCertAuth{
				CaKey:         caKey,
				Organizations: []string{kubeadmconstants.NodesGroup},
			},
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			CaCert:     caCert,
			APIServer:  cfg.GetMasterEndpoint(),
			ClientName: kubeadmconstants.ControllerManagerUser,
			ClientCertAuth: &clientCertAuth{
				CaKey: caKey,
			},
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			CaCert:     caCert,
			APIServer:  cfg.GetMasterEndpoint(),
			ClientName: kubeadmconstants.SchedulerUser,
			ClientCertAuth: &clientCertAuth{
				CaKey: caKey,
			},
		},
	}

	return kubeConfigSpec, nil
}

// buildKubeConfigFromSpec creates a kubeconfig object for the given kubeConfigSpec
func buildKubeConfigFromSpec(spec *kubeConfigSpec) (*clientcmdapi.Config, error) {

	// If this kubeconfing should use token
	if spec.TokenAuth != nil {
		// create a kubeconfig with a token
		return kubeconfigutil.CreateWithToken(
			spec.APIServer,
			"kubernetes",
			spec.ClientName,
			certutil.EncodeCertPEM(spec.CaCert),
			spec.TokenAuth.Token,
		), nil
	}

	// otherwise, create a client certs
	clientCertConfig := certutil.Config{
		CommonName:   spec.ClientName,
		Organization: spec.ClientCertAuth.Organizations,
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	clientCert, clientKey, err := pkiutil.NewCertAndKey(spec.CaCert, spec.ClientCertAuth.CaKey, clientCertConfig)
	if err != nil {
		return nil, fmt.Errorf("failure while creating %s client certificate: %v", spec.ClientName, err)
	}

	// create a kubeconfig with the client certs
	return kubeconfigutil.CreateWithCerts(
		spec.APIServer,
		"kubernetes",
		spec.ClientName,
		certutil.EncodeCertPEM(spec.CaCert),
		certutil.EncodePrivateKeyPEM(clientKey),
		certutil.EncodeCertPEM(clientCert),
	), nil
}

// createKubeConfigFileIfNotExists saves the KubeConfig object into a file if there isn't any file at the given path.
// If there already is a KubeConfig file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected config equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// but if a file exists but has old content or isn't a kubeconfig file, this function returns an error.
func createKubeConfigFileIfNotExists(outDir, filename string, config *clientcmdapi.Config) error {
	kubeConfigFilePath := filepath.Join(outDir, filename)

	// Check if the file exist, and if it doesn't, just write it to disk
	if _, err := os.Stat(kubeConfigFilePath); os.IsNotExist(err) {
		err = kubeconfigutil.WriteToDisk(kubeConfigFilePath, config)
		if err != nil {
			return fmt.Errorf("failed to save kubeconfig file %s on disk: %v", kubeConfigFilePath, err)
		}

		fmt.Printf("[kubeconfig] Wrote KubeConfig file to disk: %q\n", filename)
		return nil
	}

	// The kubeconfig already exists, let's check if it has got the same CA and server URL
	currentConfig, err := clientcmd.LoadFromFile(kubeConfigFilePath)
	if err != nil {
		return fmt.Errorf("failed to load kubeconfig file %s that already exists on disk: %v", kubeConfigFilePath, err)
	}

	expectedCtx := config.CurrentContext
	expectedCluster := config.Contexts[expectedCtx].Cluster
	currentCtx := currentConfig.CurrentContext
	currentCluster := currentConfig.Contexts[currentCtx].Cluster

	// If the current CA cert on disk doesn't match the expected CA cert, error out because we have a file, but it's stale
	if !bytes.Equal(currentConfig.Clusters[currentCluster].CertificateAuthorityData, config.Clusters[expectedCluster].CertificateAuthorityData) {
		return fmt.Errorf("a kubeconfig file %q exists already but has got the wrong CA cert", kubeConfigFilePath)
	}
	// If the current API Server location on disk doesn't match the expected API server, error out because we have a file, but it's stale
	if currentConfig.Clusters[currentCluster].Server != config.Clusters[expectedCluster].Server {
		return fmt.Errorf("a kubeconfig file %q exists already but has got the wrong API Server URL", kubeConfigFilePath)
	}

	// kubeadm doesn't validate the existing kubeconfig file more than this (kubeadm trusts the client certs to be valid)
	// Basically, if we find a kubeconfig file with the same path; the same CA cert and the same server URL;
	// kubeadm thinks those files are equal and doesn't bother writing a new file
	fmt.Printf("[kubeconfig] Using existing up-to-date KubeConfig file: %q\n", filename)

	return nil
}

// WriteKubeConfigWithClientCert writes a kubeconfig file - with a client certificate as authentication info  - to the given writer.
func WriteKubeConfigWithClientCert(out io.Writer, cfg *kubeadmapi.MasterConfiguration, clientName string) error {

	// creates the KubeConfigSpecs, actualized for the current MasterConfiguration
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v", err)
	}

	spec := &kubeConfigSpec{
		ClientName: clientName,
		APIServer:  cfg.GetMasterEndpoint(),
		CaCert:     caCert,
		ClientCertAuth: &clientCertAuth{
			CaKey: caKey,
		},
	}

	return writeKubeConfigFromSpec(out, spec)
}

// WriteKubeConfigWithToken writes a kubeconfig file - with a token as client authentication info - to the given writer.
func WriteKubeConfigWithToken(out io.Writer, cfg *kubeadmapi.MasterConfiguration, clientName, token string) error {

	// creates the KubeConfigSpecs, actualized for the current MasterConfiguration
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v", err)
	}

	spec := &kubeConfigSpec{
		ClientName: clientName,
		APIServer:  cfg.GetMasterEndpoint(),
		CaCert:     caCert,
		TokenAuth: &tokenAuth{
			Token: token,
		},
	}

	return writeKubeConfigFromSpec(out, spec)
}

// writeKubeConfigFromSpec creates a kubeconfig object from a kubeConfigSpec and writes it to the given writer.
func writeKubeConfigFromSpec(out io.Writer, spec *kubeConfigSpec) error {

	// builds the KubeConfig object
	config, err := buildKubeConfigFromSpec(spec)
	if err != nil {
		return err
	}

	// writes the KubeConfig to disk if it not exists
	configBytes, err := clientcmd.Write(*config)
	if err != nil {
		return fmt.Errorf("failure while serializing admin kubeconfig: %v", err)
	}

	fmt.Fprintln(out, string(configBytes))
	return nil
}
