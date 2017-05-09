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
	"fmt"
	"io/ioutil"
	"net"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	netutil "k8s.io/apimachinery/pkg/util/net"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdCerts returns main command for kubeadm certs phase.
// This command is not meant to be run on its own, but acts as container for related sum commands
func NewCmdCerts() *cobra.Command {

	certCmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Generate certificates for a Kubernetes cluster.",
		RunE:    subCmdRunE("certs"),
	}

	certCmd.AddCommand(newSubCmdCerts()...)

	return certCmd
}

// newSubCmdCerts returns sub commands for certs phase
func newSubCmdCerts() []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use            string
		short          string
		createCertFunc certphase.BulkCreateCertFunc
	}{
		{
			use:            "all",
			short:          "Generate all PKI assets necessary to establish the control plane",
			createCertFunc: certphase.CreatePKIAssets,
		},
		{
			use:            "ca",
			short:          "Generate CA certificate and key for a Kubernetes cluster.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateCACertAndKey),
		},
		{
			use:            "apiserver",
			short:          "Generate api server certificate and key.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateAPIServerCertAndKey),
		},
		{
			use:            "apiserver-client",
			short:          "Generate a client certificate for the apiservers to connect to the kubelets securely.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateAPIServerKubeletClientCertAndKey),
		},
		{
			use:            "sa",
			short:          "Generate a private key for signing service account tokens along with its public key.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateServiceAccountKeyAndPublicKey),
		},
		{
			use:            "frontproxy",
			short:          "Generate front proxy CA certificate and key for a Kubernetes cluster.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateFrontProxyCACertAndKey),
		},
		{
			use:            "frontproxy-client",
			short:          "Generate front proxy CA client certificate and key for a Kubernetes cluster.",
			createCertFunc: wrapCreateCertFunc(certphase.CreateFrontProxyClientCertAndKey),
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCreateCerts(&cfgPath, cfg, properties.createCertFunc),
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

// CreateCertFunc is a utility method that wraps "atomic" CreateCertFunction into a BulkCreateCertFunc with one single step,
// thus allowing to treat both "atomic" and bulk functions in the same way
func wrapCreateCertFunc(simpleFunc certphase.CreateCertFunc) certphase.BulkCreateCertFunc {

	return func(cfg *kubeadmapi.MasterConfiguration) (certphase.BulkCreateCertResult, error) {
		result, err := simpleFunc(cfg)
		if err != nil {
			return nil, err
		}
		return []*certphase.CreateCertResult{result}, nil
	}
}

// runCreateCerts executes the given createCertFunc, including preparation of inpuut parameter and handling of results
func runCreateCerts(cfgPath *string, cfg *kubeadmapiext.MasterConfiguration, createCertFunc certphase.BulkCreateCertFunc) func(cmd *cobra.Command, args []string) {

	// the following statement builds a clousure that wraps a call to a CreateCertFunc, binding
	// the called function with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as values, while other parameters - passed as reference -
	// are shared between sub commnands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {
		internalcfg := &kubeadmapi.MasterConfiguration{}

		// Takes passed flags into account; the defaulting is run once again enforcing assignement of
		// static default values to cfg only for values not provided with flags
		api.Scheme.Default(cfg)
		api.Scheme.Convert(cfg, internalcfg, nil)

		// Loads configuration from config file, if provided
		// TODO: with current implementation --config overrides command line flag, and this is counter inutitive
		// (e.g. defining api.Scheme.Merge instead of api.Scheme.Convert)
		// see https://github.com/kubernetes/kubeadm/issues/267
		err := tryLoadCfg(*cfgPath, internalcfg)
		kubeadmutil.CheckErr(err)

		// Applies dynamic defaults to cfg settings not provided with flags
		err = setInitDynamicDefaults(internalcfg)
		kubeadmutil.CheckErr(err)

		// Validates cfg (flags/configs + defaults + dynamics defaults)
		err = validation.ValidateMasterConfiguration(internalcfg).ToAggregate()
		kubeadmutil.CheckErr(err)

		// Execute the create createCertFunc
		results, err := createCertFunc(internalcfg)
		kubeadmutil.CheckErr(err)

		// Prints results to UX
		fmt.Printf("%v", results)
	}
}

// SetInitDynamicDefaults set defaults dynamically by fetching information from the internet, looking up network interfaces, etc. (the API group defaulting can't co this
// TODO: this function should be centralized across all phases & init. see https://github.com/kubernetes/kubeadm/issues/267
func setInitDynamicDefaults(cfg *kubeadmapi.MasterConfiguration) error {

	// Choose the right address for the API Server to advertise. If the advertise address is localhost or 0.0.0.0, the default interface's IP address is used
	// This is the same logic as the API Server uses
	ip, err := netutil.ChooseBindAddress(net.ParseIP(cfg.API.AdvertiseAddress))
	if err != nil {
		return err
	}
	cfg.API.AdvertiseAddress = ip.String()

	if cfg.Token == "" {
		var err error
		cfg.Token, err = tokenutil.GenerateToken()
		if err != nil {
			return fmt.Errorf("couldn't generate random token: %v", err)
		}
	}

	return nil
}

// TryLoadCfg tries to loads a Master configuration from the given file (if defined)
// TODO: this function should be centralized across all phases & init. see https://github.com/kubernetes/kubeadm/issues/267
func tryLoadCfg(cfgPath string, cfg *kubeadmapi.MasterConfiguration) error {

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	if cfg.Token == "" {
		var err error
		cfg.Token, err = tokenutil.GenerateToken()
		if err != nil {
			return fmt.Errorf("couldn't generate random token: %v", err)
		}
	}

	return nil
}
