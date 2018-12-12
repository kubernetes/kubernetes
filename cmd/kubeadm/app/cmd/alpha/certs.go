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
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	genericLongDesc = normalizer.LongDesc(`
		Renews the %[1]s, and saves them into %[2]s.cert and %[2]s.key files.

    Extra attributes such as SANs will be based on the existing certificates, there is no need to resupply them.
`)
	allLongDesc = normalizer.LongDesc(`
    Renews all known certificates necessary to run the control plan. Renewals are run unconditionally, regardless
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
		Short: "Renews certificates for a Kubernetes cluster",
		Long:  cmdutil.MacroCommandLongDescription,
		RunE:  cmdutil.SubCmdRunE("renew"),
	}

	cmd.AddCommand(getRenewSubCommands()...)

	return cmd
}

type renewConfig struct {
	cfgPath        string
	kubeconfigPath string
	cfg            kubeadmapiv1beta1.InitConfiguration
	useAPI         bool
	useCSR         bool
	csrPath        string
}

func getRenewSubCommands() []*cobra.Command {
	cfg := &renewConfig{
		kubeconfigPath: constants.GetAdminKubeConfigPath(),
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
			cmd := generateRenewalCommand(cert, cfg)
			// get the implementation of renewing this certificate
			renewalFunc := generateRenewalFunction(cert, caCert, cfg)
			// install the implementation into the command
			cmd.Run = func(*cobra.Command, []string) { renewalFunc() }
			cmdList = append(cmdList, cmd)
			// Collect renewal functions for `renew all`
			funcList = append(funcList, renewalFunc)
		}
	}

	allCmd := &cobra.Command{
		Use:   "all",
		Short: "renew all available certificates",
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

func generateRenewalFunction(cert *certsphase.KubeadmCert, caCert *certsphase.KubeadmCert, cfg *renewConfig) func() {
	return func() {
		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfg.cfgPath, &cfg.cfg)
		kubeadmutil.CheckErr(err)

		if cfg.useCSR {
			path := cfg.csrPath
			if path == "" {
				path = cfg.cfg.CertificatesDir
			}
			err := certsphase.CreateCSR(cert, internalcfg, path)
			kubeadmutil.CheckErr(err)
			return
		}

		renewer, err := getRenewer(cfg, caCert.BaseName)
		kubeadmutil.CheckErr(err)

		err = renewal.RenewExistingCert(internalcfg.CertificatesDir, cert.BaseName, renewer)
		kubeadmutil.CheckErr(err)
	}
}

func generateRenewalCommand(cert *certsphase.KubeadmCert, cfg *renewConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   cert.Name,
		Short: fmt.Sprintf("Generates the %s", cert.LongName),
		Long:  fmt.Sprintf(genericLongDesc, cert.LongName, cert.BaseName),
	}
	addFlags(cmd, cfg)
	return cmd
}

func getRenewer(cfg *renewConfig, caCertBaseName string) (renewal.Interface, error) {
	if cfg.useAPI {
		kubeConfigPath := cmdutil.FindExistingKubeConfig(cfg.kubeconfigPath)
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
