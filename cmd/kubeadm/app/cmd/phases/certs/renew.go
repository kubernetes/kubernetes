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

package renew

import (
	"fmt"

	"github.com/spf13/cobra"

	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
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

// NewCmdCertsRenewal creates a new `cert renew` command.
func NewCmdCertsRenewal() *cobra.Command {
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
	cfg            kubeadmapiv1alpha3.InitConfiguration
	useAPI         bool
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
	allCmds := []func() error{}

	for caCert, certs := range certTree {
		// Don't offer to renew CAs; would cause serious consequences
		for _, cert := range certs {
			cmd := makeCommandForRenew(cert, caCert, cfg)
			cmdList = append(cmdList, cmd)
			allCmds = append(allCmds, cmd.Execute)
		}
	}

	allCmd := &cobra.Command{
		Use:   "all",
		Short: "renew all available certificates",
		Long:  allLongDesc,
		Run: func(*cobra.Command, []string) {
			for _, cmd := range allCmds {
				err := cmd()
				kubeadmutil.CheckErr(err)
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
	cmd.Flags().BoolVar(&cfg.useAPI, "use-api", cfg.useAPI, "Use the Kubernetes certificate API to renew certificates")
}

// generateCertCommand takes mostly strings instead of structs to avoid using structs in a for loop
func generateCertCommand(name, longName, baseName, caCertBaseName string, cfg *renewConfig) *cobra.Command {
	return &cobra.Command{
		Use:   name,
		Short: fmt.Sprintf("Generates the %s", longName),
		Long:  fmt.Sprintf(genericLongDesc, longName, baseName),
		Run: func(cmd *cobra.Command, args []string) {
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfg.cfgPath, &cfg.cfg)
			kubeadmutil.CheckErr(err)
			renewer, err := getRenewer(cfg, caCertBaseName)
			kubeadmutil.CheckErr(err)

			err = renewal.RenewExistingCert(internalcfg.CertificatesDir, baseName, renewer)
			kubeadmutil.CheckErr(err)
		},
	}
}

func makeCommandForRenew(cert *certsphase.KubeadmCert, caCert *certsphase.KubeadmCert, cfg *renewConfig) *cobra.Command {
	certCmd := generateCertCommand(cert.Name, cert.LongName, cert.BaseName, caCert.BaseName, cfg)
	addFlags(certCmd, cfg)
	return certCmd
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
