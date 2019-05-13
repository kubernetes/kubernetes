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

package alpha

import (
	"fmt"

	"github.com/spf13/cobra"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	genericCertRenewLongDesc = normalizer.LongDesc(`
		Renew the %[1]s, and save them into %[2]s.cert and %[2]s.key files.

    Extra attributes such as SANs will be based on the existing certificates, there is no need to resupply them.
`)
	genericCertRenewEmbeddedLongDesc = normalizer.LongDesc(`
Renew the certificate embedded in the kubeconfig file %s.

Kubeconfig attributes and certificate extra attributes such as SANs will be based on the existing kubeconfig/certificates, there is no need to resupply them.
`)

	allLongDesc = normalizer.LongDesc(`
    Renew all known certificates necessary to run the control plane. Renewals are run unconditionally, regardless
    of expiration date. Renewals can also be run individually for more control.
`)
)

// newCmdCertsUtility returns main command for certs phase
func newCmdCertsUtility() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Commands related to handling kubernetes certificates",
	}

	cmd.AddCommand(newCmdCertsRenewal())
	return cmd
}

// newCmdCertsRenewal creates a new `cert renew` command.
func newCmdCertsRenewal() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "renew",
		Short: "Renew certificates for a Kubernetes cluster",
		Long:  cmdutil.MacroCommandLongDescription,
		RunE:  cmdutil.SubCmdRunE("renew"),
	}

	cmd.AddCommand(getRenewSubCommands(kubeadmconstants.KubernetesDir)...)

	return cmd
}

type renewConfig struct {
	cfgPath        string
	kubeconfigPath string
	cfg            kubeadmapiv1beta2.InitConfiguration
	useAPI         bool
	useCSR         bool
	csrPath        string
}

func getRenewSubCommands(kdir string) []*cobra.Command {
	cfg := &renewConfig{
		cfg: kubeadmapiv1beta2.InitConfiguration{
			ClusterConfiguration: kubeadmapiv1beta2.ClusterConfiguration{
				// Setting kubernetes version to a default value in order to allow a not necessary internet lookup
				KubernetesVersion: constants.CurrentKubernetesVersion.String(),
			},
		},
	}
	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(&cfg.cfg)

	certTree, err := certsphase.GetDefaultCertList().AsMap().CertTree()
	kubeadmutil.CheckErr(err)

	cmdList := []*cobra.Command{}
	funcList := []func(){}

	for caCert, certs := range certTree {
		// Don't offer to renew CAs; would cause serious consequences
		for _, cert := range certs {
			// get the cobra.Command skeleton for this command
			cmd := generateCertRenewalCommand(cert, cfg)
			// get the implementation of renewing this certificate
			renewalFunc := func(cert *certsphase.KubeadmCert, caCert *certsphase.KubeadmCert) func() {
				return func() { renewCert(cert, caCert, cfg) }
			}(cert, caCert)
			// install the implementation into the command
			cmd.Run = func(*cobra.Command, []string) { renewalFunc() }
			cmdList = append(cmdList, cmd)
			// Collect renewal functions for `renew all`
			funcList = append(funcList, renewalFunc)
		}
	}

	kubeconfigs := []string{
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
		//NB. we are escluding KubeletKubeConfig from renewal because management of this certificate is delegated to kubelet
	}

	for _, k := range kubeconfigs {
		// get the cobra.Command skeleton for this command
		cmd := generateEmbeddedCertRenewalCommand(k, cfg)
		// get the implementation of renewing this certificate
		renewalFunc := func(kdir, k string) func() {
			return func() { renewEmbeddedCert(kdir, k, cfg) }
		}(kdir, k)
		// install the implementation into the command
		cmd.Run = func(*cobra.Command, []string) { renewalFunc() }
		cmdList = append(cmdList, cmd)
		// Collect renewal functions for `renew all`
		funcList = append(funcList, renewalFunc)
	}

	allCmd := &cobra.Command{
		Use:   "all",
		Short: "Renew all available certificates",
		Long:  allLongDesc,
		Run: func(*cobra.Command, []string) {
			for _, f := range funcList {
				f()
			}
		},
	}
	addFlags(allCmd, cfg)

	cmdList = append(cmdList, allCmd)
	return cmdList
}

func addFlags(cmd *cobra.Command, cfg *renewConfig) {
	options.AddConfigFlag(cmd.Flags(), &cfg.cfgPath)
	options.AddCertificateDirFlag(cmd.Flags(), &cfg.cfg.CertificatesDir)
	options.AddKubeConfigFlag(cmd.Flags(), &cfg.kubeconfigPath)
	options.AddCSRFlag(cmd.Flags(), &cfg.useCSR)
	options.AddCSRDirFlag(cmd.Flags(), &cfg.csrPath)
	cmd.Flags().BoolVar(&cfg.useAPI, "use-api", cfg.useAPI, "Use the Kubernetes certificate API to renew certificates")
}

func renewCert(cert *certsphase.KubeadmCert, caCert *certsphase.KubeadmCert, cfg *renewConfig) {
	internalcfg, err := configutil.LoadOrDefaultInitConfiguration(cfg.cfgPath, &cfg.cfg)
	kubeadmutil.CheckErr(err)

	// if the renewal operation is set to generate only CSR request
	if cfg.useCSR {
		// trigger CSR generation in the csrPath, or if this one is missing, in the CertificateDir
		path := cfg.csrPath
		if path == "" {
			path = cfg.cfg.CertificatesDir
		}
		err := certsphase.CreateCSR(cert, internalcfg, path)
		kubeadmutil.CheckErr(err)
		return
	}

	// otherwise, the renewal operation has to actually renew a certificate

	var externalCA bool
	switch caCert.BaseName {
	case kubeadmconstants.CACertAndKeyBaseName:
		// Check if an external CA is provided by the user (when the CA Cert is present but the CA Key is not)
		externalCA, _ = certsphase.UsingExternalCA(&internalcfg.ClusterConfiguration)
	case kubeadmconstants.FrontProxyCACertAndKeyBaseName:
		// Check if an external Front-Proxy CA is provided by the user (when the Front-Proxy CA Cert is present but the Front-Proxy CA Key is not)
		externalCA, _ = certsphase.UsingExternalFrontProxyCA(&internalcfg.ClusterConfiguration)
	default:
		externalCA = false
	}

	if !externalCA {
		renewer, err := getRenewer(cfg, caCert.BaseName)
		kubeadmutil.CheckErr(err)

		err = renewal.RenewExistingCert(internalcfg.CertificatesDir, cert.BaseName, renewer)
		kubeadmutil.CheckErr(err)

		fmt.Printf("Certificate %s renewed\n", cert.Name)
		return
	}

	fmt.Printf("Detected external %s, certificate %s can't be renewed\n", cert.CAName, cert.Name)
}

func renewEmbeddedCert(kdir, k string, cfg *renewConfig) {
	internalcfg, err := configutil.LoadOrDefaultInitConfiguration(cfg.cfgPath, &cfg.cfg)
	kubeadmutil.CheckErr(err)

	// if the renewal operation is set to generate only CSR request
	if cfg.useCSR {
		// trigger CSR generation in the csrPath, or if this one is missing, in the CertificateDir
		path := cfg.csrPath
		if path == "" {
			path = cfg.cfg.CertificatesDir
		}
		err := certsphase.CreateCSR(nil, internalcfg, path)
		kubeadmutil.CheckErr(err)
		return
	}

	// otherwise, the renewal operation has to actually renew a certificate

	// Check if an external CA is provided by the user (when the CA Cert is present but the CA Key is not)
	externalCA, _ := certsphase.UsingExternalCA(&internalcfg.ClusterConfiguration)

	if !externalCA {
		renewer, err := getRenewer(cfg, certsphase.KubeadmCertRootCA.BaseName)
		kubeadmutil.CheckErr(err)

		err = renewal.RenewEmbeddedClientCert(kdir, k, renewer)
		kubeadmutil.CheckErr(err)

		fmt.Printf("Certificate embedded in %s renewed\n", k)
		return
	}

	fmt.Printf("Detected external CA, certificate embedded in %s can't be renewed\n", k)
}

func generateCertRenewalCommand(cert *certsphase.KubeadmCert, cfg *renewConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   cert.Name,
		Short: fmt.Sprintf("Renew the %s", cert.LongName),
		Long:  fmt.Sprintf(genericCertRenewLongDesc, cert.LongName, cert.BaseName),
	}
	addFlags(cmd, cfg)
	return cmd
}

func generateEmbeddedCertRenewalCommand(k string, cfg *renewConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   k,
		Short: fmt.Sprintf("Renew the certificate embedded in %s", k),
		Long:  fmt.Sprintf(genericCertRenewEmbeddedLongDesc, k),
	}
	addFlags(cmd, cfg)
	return cmd
}

func getRenewer(cfg *renewConfig, caCertBaseName string) (renewal.Interface, error) {
	if cfg.useAPI {
		kubeConfigPath := cmdutil.GetKubeConfigPath(cfg.kubeconfigPath)
		client, err := kubeconfigutil.ClientSetFromFile(kubeConfigPath)
		if err != nil {
			return nil, err
		}
		return renewal.NewCertsAPIRenawal(client), nil
	}

	caCert, caKey, err := certsphase.LoadCertificateAuthority(cfg.cfg.CertificatesDir, caCertBaseName)
	if err != nil {
		return nil, err
	}

	return renewal.NewFileRenewal(caCert, caKey), nil
}
