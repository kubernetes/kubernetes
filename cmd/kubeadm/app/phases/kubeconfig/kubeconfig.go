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

package kubeconfig

import (
	"bytes"
	"crypto"
	"crypto/x509"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/klog/v2"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	pkiutil "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

const (
	errInvalid = "invalid argument"
	errExist   = "file already exists"
)

// clientCertAuth struct holds info required to build a client certificate to provide authentication info in a kubeconfig object
type clientCertAuth struct {
	CAKey         crypto.Signer
	Organizations []string
}

// tokenAuth struct holds info required to use a token to provide authentication info in a kubeconfig object
type tokenAuth struct {
	Token string
}

// kubeConfigSpec struct holds info required to build a KubeConfig object
type kubeConfigSpec struct {
	CACert         *x509.Certificate
	APIServer      string
	ClientName     string
	TokenAuth      *tokenAuth
	ClientCertAuth *clientCertAuth
}

// CreateJoinControlPlaneKubeConfigFiles will create and write to disk the kubeconfig files required by kubeadm
// join --control-plane workflow, plus the admin kubeconfig file used by the administrator and kubeadm itself; the
// kubelet.conf file must not be created because it will be created and signed by the kubelet TLS bootstrap process.
// If any kubeconfig files already exists, it used only if evaluated equal; otherwise an error is returned.
func CreateJoinControlPlaneKubeConfigFiles(outDir string, cfg *kubeadmapi.InitConfiguration) error {
	return createKubeConfigFiles(
		outDir,
		cfg,
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
	)
}

// CreateKubeConfigFile creates a kubeconfig file.
// If the kubeconfig file already exists, it is used only if evaluated equal; otherwise an error is returned.
func CreateKubeConfigFile(kubeConfigFileName string, outDir string, cfg *kubeadmapi.InitConfiguration) error {
	klog.V(1).Infof("creating kubeconfig file for %s", kubeConfigFileName)
	return createKubeConfigFiles(outDir, cfg, kubeConfigFileName)
}

// createKubeConfigFiles creates all the requested kubeconfig files.
// If kubeconfig files already exists, they are used only if evaluated equal; otherwise an error is returned.
func createKubeConfigFiles(outDir string, cfg *kubeadmapi.InitConfiguration, kubeConfigFileNames ...string) error {

	// gets the KubeConfigSpecs, actualized for the current InitConfiguration
	specs, err := getKubeConfigSpecs(cfg)
	if err != nil {
		return err
	}

	for _, kubeConfigFileName := range kubeConfigFileNames {
		// retrieves the KubeConfigSpec for given kubeConfigFileName
		spec, exists := specs[kubeConfigFileName]
		if !exists {
			return errors.Errorf("couldn't retrieve KubeConfigSpec for %s", kubeConfigFileName)
		}

		// builds the KubeConfig object
		config, err := buildKubeConfigFromSpec(spec, cfg.ClusterName)
		if err != nil {
			return err
		}

		// writes the kubeconfig to disk if it does not exist
		if err = createKubeConfigFileIfNotExists(outDir, kubeConfigFileName, config); err != nil {
			return err
		}
	}

	return nil
}

// getKubeConfigSpecs returns all KubeConfigSpecs actualized to the context of the current InitConfiguration
// NB. this method holds the information about how kubeadm creates kubeconfig files.
func getKubeConfigSpecs(cfg *kubeadmapi.InitConfiguration) (map[string]*kubeConfigSpec, error) {
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return nil, errors.Wrap(err, "couldn't create a kubeconfig; the CA files couldn't be loaded")
	}
	configs, err := getKubeConfigSpecsBase(cfg)
	if err != nil {
		return nil, err
	}
	for _, spec := range configs {
		spec.CACert = caCert
		spec.ClientCertAuth.CAKey = caKey
	}
	return configs, nil
}

// buildKubeConfigFromSpec creates a kubeconfig object for the given kubeConfigSpec
func buildKubeConfigFromSpec(spec *kubeConfigSpec, clustername string) (*clientcmdapi.Config, error) {

	// If this kubeconfig should use token
	if spec.TokenAuth != nil {
		// create a kubeconfig with a token
		return kubeconfigutil.CreateWithToken(
			spec.APIServer,
			clustername,
			spec.ClientName,
			pkiutil.EncodeCertPEM(spec.CACert),
			spec.TokenAuth.Token,
		), nil
	}

	// otherwise, create a client certs
	clientCertConfig := newClientCertConfigFromKubeConfigSpec(spec)

	clientCert, clientKey, err := pkiutil.NewCertAndKey(spec.CACert, spec.ClientCertAuth.CAKey, &clientCertConfig)
	if err != nil {
		return nil, errors.Wrapf(err, "failure while creating %s client certificate", spec.ClientName)
	}

	encodedClientKey, err := keyutil.MarshalPrivateKeyToPEM(clientKey)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal private key to PEM")
	}
	// create a kubeconfig with the client certs
	return kubeconfigutil.CreateWithCerts(
		spec.APIServer,
		clustername,
		spec.ClientName,
		pkiutil.EncodeCertPEM(spec.CACert),
		encodedClientKey,
		pkiutil.EncodeCertPEM(clientCert),
	), nil
}

func newClientCertConfigFromKubeConfigSpec(spec *kubeConfigSpec) pkiutil.CertConfig {
	return pkiutil.CertConfig{
		Config: certutil.Config{
			CommonName:   spec.ClientName,
			Organization: spec.ClientCertAuth.Organizations,
			Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		},
	}
}

// validateKubeConfig check if the kubeconfig file exist and has the expected CA and server URL
func validateKubeConfig(outDir, filename string, config *clientcmdapi.Config) error {
	kubeConfigFilePath := filepath.Join(outDir, filename)

	if _, err := os.Stat(kubeConfigFilePath); err != nil {
		return err
	}

	// The kubeconfig already exists, let's check if it has got the same CA and server URL
	currentConfig, err := clientcmd.LoadFromFile(kubeConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to load kubeconfig file %s that already exists on disk", kubeConfigFilePath)
	}

	expectedCtx, exists := config.Contexts[config.CurrentContext]
	if !exists {
		return errors.Errorf("failed to find expected context %s", config.CurrentContext)
	}
	expectedCluster := expectedCtx.Cluster
	currentCtx, exists := currentConfig.Contexts[currentConfig.CurrentContext]
	if !exists {
		return errors.Errorf("failed to find CurrentContext in Contexts of the kubeconfig file %s", kubeConfigFilePath)
	}
	currentCluster := currentCtx.Cluster
	if currentConfig.Clusters[currentCluster] == nil {
		return errors.Errorf("failed to find the given CurrentContext Cluster in Clusters of the kubeconfig file %s", kubeConfigFilePath)
	}

	// Make sure the compared CAs are whitespace-trimmed. The function clientcmd.LoadFromFile() just decodes
	// the base64 CA and places it raw in the v1.Config object. In case the user has extra whitespace
	// in the CA they used to create a kubeconfig this comparison to a generated v1.Config will otherwise fail.
	caCurrent := bytes.TrimSpace(currentConfig.Clusters[currentCluster].CertificateAuthorityData)
	caExpected := bytes.TrimSpace(config.Clusters[expectedCluster].CertificateAuthorityData)

	// If the current CA cert on disk doesn't match the expected CA cert, error out because we have a file, but it's stale
	if !bytes.Equal(caCurrent, caExpected) {
		return errors.Errorf("a kubeconfig file %q exists already but has got the wrong CA cert", kubeConfigFilePath)
	}
	// If the current API Server location on disk doesn't match the expected API server, error out because we have a file, but it's stale
	if currentConfig.Clusters[currentCluster].Server != config.Clusters[expectedCluster].Server {
		return errors.Errorf("a kubeconfig file %q exists already but has got the wrong API Server URL", kubeConfigFilePath)
	}

	return nil
}

// createKubeConfigFileIfNotExists saves the KubeConfig object into a file if there isn't any file at the given path.
// If there already is a kubeconfig file at the given path; kubeadm tries to load it and check if the values in the
// existing and the expected config equals. If they do; kubeadm will just skip writing the file as it's up-to-date,
// but if a file exists but has old content or isn't a kubeconfig file, this function returns an error.
func createKubeConfigFileIfNotExists(outDir, filename string, config *clientcmdapi.Config) error {
	kubeConfigFilePath := filepath.Join(outDir, filename)

	err := validateKubeConfig(outDir, filename, config)
	if err != nil {
		// Check if the file exist, and if it doesn't, just write it to disk
		if !os.IsNotExist(err) {
			return err
		}
		fmt.Printf("[kubeconfig] Writing %q kubeconfig file\n", filename)
		err = kubeconfigutil.WriteToDisk(kubeConfigFilePath, config)
		if err != nil {
			return errors.Wrapf(err, "failed to save kubeconfig file %q on disk", kubeConfigFilePath)
		}
		return nil
	}
	// kubeadm doesn't validate the existing kubeconfig file more than this (kubeadm trusts the client certs to be valid)
	// Basically, if we find a kubeconfig file with the same path; the same CA cert and the same server URL;
	// kubeadm thinks those files are equal and doesn't bother writing a new file
	fmt.Printf("[kubeconfig] Using existing kubeconfig file: %q\n", kubeConfigFilePath)

	return nil
}

// WriteKubeConfigWithClientCert writes a kubeconfig file - with a client certificate as authentication info  - to the given writer.
func WriteKubeConfigWithClientCert(out io.Writer, cfg *kubeadmapi.InitConfiguration, clientName string, organizations []string) error {

	// creates the KubeConfigSpecs, actualized for the current InitConfiguration
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return errors.Wrap(err, "couldn't create a kubeconfig; the CA files couldn't be loaded")
	}

	controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
	if err != nil {
		return err
	}

	spec := &kubeConfigSpec{
		ClientName: clientName,
		APIServer:  controlPlaneEndpoint,
		CACert:     caCert,
		ClientCertAuth: &clientCertAuth{
			CAKey:         caKey,
			Organizations: organizations,
		},
	}

	return writeKubeConfigFromSpec(out, spec, cfg.ClusterName)
}

// WriteKubeConfigWithToken writes a kubeconfig file - with a token as client authentication info - to the given writer.
func WriteKubeConfigWithToken(out io.Writer, cfg *kubeadmapi.InitConfiguration, clientName, token string) error {

	// creates the KubeConfigSpecs, actualized for the current InitConfiguration
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return errors.Wrap(err, "couldn't create a kubeconfig; the CA files couldn't be loaded")
	}

	controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
	if err != nil {
		return err
	}

	spec := &kubeConfigSpec{
		ClientName: clientName,
		APIServer:  controlPlaneEndpoint,
		CACert:     caCert,
		TokenAuth: &tokenAuth{
			Token: token,
		},
	}

	return writeKubeConfigFromSpec(out, spec, cfg.ClusterName)
}

// writeKubeConfigFromSpec creates a kubeconfig object from a kubeConfigSpec and writes it to the given writer.
func writeKubeConfigFromSpec(out io.Writer, spec *kubeConfigSpec, clustername string) error {

	// builds the KubeConfig object
	config, err := buildKubeConfigFromSpec(spec, clustername)
	if err != nil {
		return err
	}

	// writes the kubeconfig to disk if it not exists
	configBytes, err := clientcmd.Write(*config)
	if err != nil {
		return errors.Wrap(err, "failure while serializing admin kubeconfig")
	}

	fmt.Fprintln(out, string(configBytes))
	return nil
}

// ValidateKubeconfigsForExternalCA check if the kubeconfig file exist and has the expected CA and server URL using kubeadmapi.InitConfiguration.
func ValidateKubeconfigsForExternalCA(outDir string, cfg *kubeadmapi.InitConfiguration) error {
	kubeConfigFileNames := []string{
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.KubeletKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
	}

	// Creates a kubeconfig file with the target CA and server URL
	// to be used as a input for validating user provided kubeconfig files
	caCert, err := pkiutil.TryLoadCertFromDisk(cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return errors.Wrapf(err, "the CA file couldn't be loaded")
	}

	controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
	if err != nil {
		return err
	}

	validationConfig := kubeconfigutil.CreateBasic(controlPlaneEndpoint, "dummy", "dummy", pkiutil.EncodeCertPEM(caCert))

	// validate user provided kubeconfig files
	for _, kubeConfigFileName := range kubeConfigFileNames {
		if err = validateKubeConfig(outDir, kubeConfigFileName, validationConfig); err != nil {
			return errors.Wrapf(err, "the %s file does not exists or it is not valid", kubeConfigFileName)
		}
	}
	return nil
}

func getKubeConfigSpecsBase(cfg *kubeadmapi.InitConfiguration) (map[string]*kubeConfigSpec, error) {
	controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
	if err != nil {
		return nil, err
	}
	localAPIEndpoint, err := kubeadmutil.GetLocalAPIEndpoint(&cfg.LocalAPIEndpoint)
	if err != nil {
		return nil, err
	}

	return map[string]*kubeConfigSpec{
		kubeadmconstants.AdminKubeConfigFileName: {
			APIServer:  controlPlaneEndpoint,
			ClientName: "kubernetes-admin",
			ClientCertAuth: &clientCertAuth{
				Organizations: []string{kubeadmconstants.SystemPrivilegedGroup},
			},
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			APIServer:  controlPlaneEndpoint,
			ClientName: fmt.Sprintf("%s%s", kubeadmconstants.NodesUserPrefix, cfg.NodeRegistration.Name),
			ClientCertAuth: &clientCertAuth{
				Organizations: []string{kubeadmconstants.NodesGroup},
			},
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			APIServer:      localAPIEndpoint,
			ClientName:     kubeadmconstants.ControllerManagerUser,
			ClientCertAuth: &clientCertAuth{},
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			APIServer:      localAPIEndpoint,
			ClientName:     kubeadmconstants.SchedulerUser,
			ClientCertAuth: &clientCertAuth{},
		},
	}, nil
}

func createKubeConfigAndCSR(kubeConfigDir string, kubeadmConfig *kubeadmapi.InitConfiguration, name string, spec *kubeConfigSpec) error {
	if kubeConfigDir == "" {
		return errors.Errorf("%s: kubeConfigDir was empty", errInvalid)
	}
	if kubeadmConfig == nil {
		return errors.Errorf("%s: kubeadmConfig was nil", errInvalid)
	}
	if name == "" {
		return errors.Errorf("%s: name was empty", errInvalid)
	}
	if spec == nil {
		return errors.Errorf("%s: spec was nil", errInvalid)
	}
	kubeConfigPath := filepath.Join(kubeConfigDir, name)
	if _, err := os.Stat(kubeConfigPath); err == nil {
		return errors.Errorf("%s: kube config: %s", errExist, kubeConfigPath)
	} else if !os.IsNotExist(err) {
		return errors.Wrapf(err, "unexpected error while checking if file exists: %s", kubeConfigPath)
	}
	if pkiutil.CSROrKeyExist(kubeConfigDir, name) {
		return errors.Errorf("%s: csr: %s", errExist, kubeConfigPath)
	}

	clientCertConfig := newClientCertConfigFromKubeConfigSpec(spec)

	clientKey, err := pkiutil.NewPrivateKey(clientCertConfig.PublicKeyAlgorithm)
	if err != nil {
		return err
	}
	clientCSR, err := pkiutil.NewCSR(clientCertConfig, clientKey)
	if err != nil {
		return err
	}

	encodedClientKey, err := keyutil.MarshalPrivateKeyToPEM(clientKey)
	if err != nil {
		return err
	}

	var (
		emptyCACert     []byte
		emptyClientCert []byte
	)

	// create a kubeconfig with the client certs
	config := kubeconfigutil.CreateWithCerts(
		spec.APIServer,
		kubeadmConfig.ClusterName,
		spec.ClientName,
		emptyCACert,
		encodedClientKey,
		emptyClientCert,
	)

	if err := kubeconfigutil.WriteToDisk(kubeConfigPath, config); err != nil {
		return err
	}
	// Write CSR to disk
	if err := pkiutil.WriteCSR(kubeConfigDir, name, clientCSR); err != nil {
		return err
	}
	return nil
}

// CreateDefaultKubeConfigsAndCSRFiles is used in ExternalCA mode to create
// kubeconfig files and adjacent CSR files.
func CreateDefaultKubeConfigsAndCSRFiles(kubeConfigDir string, kubeadmConfig *kubeadmapi.InitConfiguration) error {
	kubeConfigs, err := getKubeConfigSpecsBase(kubeadmConfig)
	if err != nil {
		return err
	}
	for name, spec := range kubeConfigs {
		if err := createKubeConfigAndCSR(kubeConfigDir, kubeadmConfig, name, spec); err != nil {
			return err
		}
	}
	return nil
}
