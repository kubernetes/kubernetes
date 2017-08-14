/*
Copyright 2017 The Kubernetes Authors.

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

package phases

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdCerts return main command for certs phase
func NewCmdCerts() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Generate certificates for a Kubernetes cluster.",
		RunE:    subCmdRunE("certs"),
	}

	cmd.AddCommand(getCertsSubCommands()...)
	return cmd
}

// getCertsSubCommands returns sub commands for certs phase
func getCertsSubCommands() []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all PKI assets necessary to establish the control plane",
			cmdFunc: CreatePKIAssets,
		},
		{
			use:     "ca",
			short:   "Generate CA certificate and key for a Kubernetes cluster.",
			cmdFunc: createOrUseCACertAndKey,
		},
		{
			use:     "apiserver",
			short:   "Generate API Server serving certificate and key.",
			cmdFunc: createOrUseAPIServerCertAndKey,
		},
		{
			use:     "apiserver-kubelet-client",
			short:   "Generate a client certificate for the API Server to connect to the kubelets securely.",
			cmdFunc: createOrUseAPIServerKubeletClientCertAndKey,
		},
		{
			use:     "sa",
			short:   "Generate a private key for signing service account tokens along with its public key.",
			cmdFunc: createOrUseServiceAccountKeyAndPublicKey,
		},
		{
			use:     "front-proxy-ca",
			short:   "Generate front proxy CA certificate and key for a Kubernetes cluster.",
			cmdFunc: createOrUseFrontProxyCACertAndKey,
		},
		{
			use:     "front-proxy-client",
			short:   "Generate front proxy CA client certificate and key for a Kubernetes cluster.",
			cmdFunc: createOrUseFrontProxyClientCertAndKey,
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdFunc(properties.cmdFunc, &cfgPath, cfg),
		}

		// Add flags to the command
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where to save and store the certificates")
		if properties.use == "all" || properties.use == "apiserver" {
			cmd.Flags().StringVar(&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain, "Use alternative domain for services, e.g. \"myorg.internal\"")
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "Use alternative range of IP address for service VIPs")
			cmd.Flags().StringSliceVar(&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", []string{}, "Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.")
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}

// runCmdFunc creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runCmdFunc(cmdFunc func(cfg *kubeadmapi.MasterConfiguration) error, cfgPath *string, cfg *kubeadmapiext.MasterConfiguration) func(cmd *cobra.Command, args []string) {

	// the following statement build a clousure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {

		// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)

		// Execute the cmdFunc
		err = cmdFunc(internalcfg)
		kubeadmutil.CheckErr(err)
	}
}

// CreatePKIAssets will create and write to disk all PKI assets necessary to establish the control plane.
// Please note that this action is a bulk action calling all the atomic certphase actions
func CreatePKIAssets(cfg *kubeadmapi.MasterConfiguration) error {

	certActions := []func(cfg *kubeadmapi.MasterConfiguration) error{
		createOrUseCACertAndKey,
		createOrUseAPIServerCertAndKey,
		createOrUseAPIServerKubeletClientCertAndKey,
		createOrUseServiceAccountKeyAndPublicKey,
		createOrUseFrontProxyCACertAndKey,
		createOrUseFrontProxyClientCertAndKey,
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

// createOrUseCACertAndKey create a new self signed CA, or use the existing one.
func createOrUseCACertAndKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseCertificateAuthorithy(
		cfg.CertificatesDir,
		kubeadmconstants.CACertAndKeyBaseName,
		"CA",
		certphase.NewCACertAndKey,
	)
}

// createOrUseAPIServerCertAndKey create a new CA certificate for apiserver, or use the existing one.
// It assumes the CA certificates should exists into the CertificatesDir
func createOrUseAPIServerCertAndKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseSignedCertificate(
		cfg.CertificatesDir,
		kubeadmconstants.CACertAndKeyBaseName,
		kubeadmconstants.APIServerCertAndKeyBaseName,
		"API server",
		func(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {
			return certphase.NewAPIServerCertAndKey(cfg, caCert, caKey)
		},
	)
}

// create a new CA certificate for kubelets calling apiserver, or use the existing one
// It assumes the CA certificates should exists into the CertificatesDir
func createOrUseAPIServerKubeletClientCertAndKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseSignedCertificate(
		cfg.CertificatesDir,
		kubeadmconstants.CACertAndKeyBaseName,
		kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
		"API server kubelet client",
		certphase.NewAPIServerKubeletClientCertAndKey,
	)
}

// createOrUseServiceAccountKeyAndPublicKey create a new public/private key pairs for signing service account user, or use the existing one.
func createOrUseServiceAccountKeyAndPublicKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseKeyAndPublicKey(
		cfg.CertificatesDir,
		kubeadmconstants.ServiceAccountKeyBaseName,
		"service account",
		certphase.NewServiceAccountSigningKey,
	)
}

// createOrUseFrontProxyCACertAndKey create a new self signed front proxy CA, or use the existing one.
func createOrUseFrontProxyCACertAndKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseCertificateAuthorithy(
		cfg.CertificatesDir,
		kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		"front-proxy CA",
		certphase.NewFrontProxyCACertAndKey,
	)
}

// createOrUseFrontProxyClientCertAndKey create a new certificate for proxy server client, or use the existing one.
// It assumes the front proxy CA certificates should exists into the CertificatesDir
func createOrUseFrontProxyClientCertAndKey(cfg *kubeadmapi.MasterConfiguration) error {

	return createOrUseSignedCertificate(
		cfg.CertificatesDir,
		kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
		"front-proxy client",
		certphase.NewFrontProxyClientCertAndKey,
	)
}

// createOrUseCertificateAuthorithy is a generic function that will create a new certificate Authorithy using the given newFunc,
// assign file names according to the given baseName, or use the existing one already present in pkiDir.
func createOrUseCertificateAuthorithy(pkiDir string, baseName string, UXName string, newFunc func() (*x509.Certificate, *rsa.PrivateKey, error)) error {

	// If cert or key exists, we should try to load them
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {

		// Try to load .crt and .key from the PKI directory
		caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("failure loading %s certificate: %v", UXName, err)
		}

		// Check if the existing cert is a CA
		if !caCert.IsCA {
			return fmt.Errorf("certificate %s is not a CA", UXName)
		}

		fmt.Printf("[certificates] Using the existing %s certificate and key.\n", UXName)
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		caCert, caKey, err := newFunc()
		if err != nil {
			return fmt.Errorf("failure while generating %s certificate and key: %v", UXName, err)
		}

		// Write .crt and .key files to disk
		if err = pkiutil.WriteCertAndKey(pkiDir, baseName, caCert, caKey); err != nil {
			return fmt.Errorf("failure while saving %s certificate and key: %v", UXName, err)
		}

		fmt.Printf("[certificates] Generated %s certificate and key.\n", UXName)
	}
	return nil
}

// createOrUseSignedCertificate is a generic function that will create a new signed certificate using the given newFunc,
// assign file names according to the given baseName, or use the existing one already present in pkiDir.
func createOrUseSignedCertificate(pkiDir string, CABaseName string, baseName string, UXName string, newFunc func(*x509.Certificate, *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error)) error {

	// Checks if certificate authorithy exists in the PKI directory
	if !pkiutil.CertOrKeyExist(pkiDir, CABaseName) {
		return fmt.Errorf("couldn't load certificate authorithy for %s from certificate dir", UXName)
	}

	// Try to load certificate authorithy .crt and .key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, CABaseName)
	if err != nil {
		return fmt.Errorf("failure loading certificate authorithy for %s: %v", UXName, err)
	}

	// Make sure the loaded CA cert actually is a CA
	if !caCert.IsCA {
		return fmt.Errorf("certificate authorithy for %s is not a CA", UXName)
	}

	// Checks if the signed certificate exists in the PKI directory
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {
		// Try to load signed certificate .crt and .key from the PKI directory
		signedCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("failure loading %s certificate: %v", UXName, err)
		}

		// Check if the existing cert is signed by the given CA
		if err := signedCert.CheckSignatureFrom(caCert); err != nil {
			return fmt.Errorf("certificate %s is not signed by corresponding CA", UXName)
		}

		fmt.Printf("[certificates] Using the existing %s certificate and key.\n", UXName)
	} else {
		// The certificate and the key did NOT exist, let's generate them now
		signedCert, signedKey, err := newFunc(caCert, caKey)
		if err != nil {
			return fmt.Errorf("failure while generating %s key and certificate: %v", UXName, err)
		}

		// Write .crt and .key files to disk
		if err = pkiutil.WriteCertAndKey(pkiDir, baseName, signedCert, signedKey); err != nil {
			return fmt.Errorf("failure while saving %s certificate and key: %v", UXName, err)
		}

		fmt.Printf("[certificates] Generated %s certificate and key.\n", UXName)
		if pkiutil.HasServerAuth(signedCert) {
			fmt.Printf("[certificates] %s serving cert is signed for DNS names %v and IPs %v\n", UXName, signedCert.DNSNames, signedCert.IPAddresses)
		}
	}

	return nil
}

// createOrUseKeyAndPublicKey is a generic function that will create a new public/private key pairs using the given newFunc,
// assign file names according to the given baseName, or use the existing one already present in pkiDir.
func createOrUseKeyAndPublicKey(pkiDir string, baseName string, UXName string, newFunc func() (*rsa.PrivateKey, error)) error {

	// Checks if the key exists in the PKI directory
	if pkiutil.CertOrKeyExist(pkiDir, baseName) {

		// Try to load .key from the PKI directory
		_, err := pkiutil.TryLoadKeyFromDisk(pkiDir, baseName)
		if err != nil {
			return fmt.Errorf("%s key existed but they could not be loaded properly: %v", UXName, err)
		}

		fmt.Printf("[certificates] Using the existing %s key.\n", UXName)
	} else {
		// The key does NOT exist, let's generate it now
		key, err := newFunc()
		if err != nil {
			return fmt.Errorf("failure while generating %s key: %v", UXName, err)
		}

		// Write .key and .pub files to disk
		if err = pkiutil.WriteKey(pkiDir, baseName, key); err != nil {
			return fmt.Errorf("failure while saving %s key: %v", UXName, err)
		}

		if err = pkiutil.WritePublicKey(pkiDir, baseName, &key.PublicKey); err != nil {
			return fmt.Errorf("failure while saving %s public key: %v", UXName, err)
		}
		fmt.Printf("[certificates] Generated %s key and public key.\n", UXName)
	}

	return nil
}
