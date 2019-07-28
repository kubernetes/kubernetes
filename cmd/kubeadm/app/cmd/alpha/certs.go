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
	"io"
	"text/tabwriter"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/duration"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/copycerts"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

var (
	genericCertRenewLongDesc = cmdutil.LongDesc(`
	Renew the %s.

	Renewals run unconditionally, regardless of certificate expiration date; extra attributes such as SANs will 
	be based on the existing file/certificates, there is no need to resupply them.

	Renewal by default tries to use the certificate authority in the local PKI managed by kubeadm; as alternative
	it is possible to use K8s certificate API for certificate renewal, or as a last option, to generate a CSR request.

	After renewal, in order to make changes effective, is is required to restart control-plane components and
	eventually re-distribute the renewed certificate in case the file is used elsewhere.
`)

	allLongDesc = cmdutil.LongDesc(`
    Renew all known certificates necessary to run the control plane. Renewals are run unconditionally, regardless
    of expiration date. Renewals can also be run individually for more control.
`)

	expirationLongDesc = cmdutil.LongDesc(`
	Checks expiration for the certificates in the local PKI managed by kubeadm.
`)

	certificateKeyLongDesc = dedent.Dedent(`
	This command will print out a secure randomly-generated certificate key that can be used with
	the "init" command.

	You can also use "kubeadm init --upload-certs" without specifying a certificate key and it will
	generate and print one for you.
`)
)

// newCmdCertsUtility returns main command for certs phase
func newCmdCertsUtility(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Commands related to handling kubernetes certificates",
	}

	cmd.AddCommand(newCmdCertsRenewal())
	cmd.AddCommand(newCmdCertsExpiration(out, kubeadmconstants.KubernetesDir))
	cmd.AddCommand(NewCmdCertificateKey())
	return cmd
}

// NewCmdCertificateKey returns cobra.Command for certificate key generate
func NewCmdCertificateKey() *cobra.Command {
	return &cobra.Command{
		Use:   "certificate-key",
		Short: "Generate certificate keys",
		Long:  certificateKeyLongDesc,

		Run: func(cmd *cobra.Command, args []string) {
			key, err := copycerts.CreateCertificateKey()
			kubeadmutil.CheckErr(err)
			fmt.Println(key)
		},
	}
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

type renewFlags struct {
	cfgPath        string
	kubeconfigPath string
	cfg            kubeadmapiv1beta2.ClusterConfiguration
	useAPI         bool
	csrOnly        bool
	csrPath        string
}

func getRenewSubCommands(kdir string) []*cobra.Command {
	flags := &renewFlags{
		cfg: kubeadmapiv1beta2.ClusterConfiguration{
			// Setting kubernetes version to a default value in order to allow a not necessary internet lookup
			KubernetesVersion: constants.CurrentKubernetesVersion.String(),
		},
	}
	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(&flags.cfg)

	// Get a renewal manager for a generic Cluster configuration, that is used only for getting
	// the list of certificates for building subcommands
	rm, err := renewal.NewManager(&kubeadmapi.ClusterConfiguration{}, "")
	kubeadmutil.CheckErr(err)

	cmdList := []*cobra.Command{}
	funcList := []func(){}

	for _, handler := range rm.Certificates() {
		// get the cobra.Command skeleton for this command
		cmd := &cobra.Command{
			Use:   handler.Name,
			Short: fmt.Sprintf("Renew the %s", handler.LongName),
			Long:  fmt.Sprintf(genericCertRenewLongDesc, handler.LongName),
		}
		addRenewFlags(cmd, flags)
		// get the implementation of renewing this certificate
		renewalFunc := func(handler *renewal.CertificateRenewHandler) func() {
			return func() { renewCert(flags, kdir, handler) }
		}(handler)
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
	addRenewFlags(allCmd, flags)

	cmdList = append(cmdList, allCmd)
	return cmdList
}

func addRenewFlags(cmd *cobra.Command, flags *renewFlags) {
	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	options.AddCertificateDirFlag(cmd.Flags(), &flags.cfg.CertificatesDir)
	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeconfigPath)
	options.AddCSRFlag(cmd.Flags(), &flags.csrOnly)
	options.AddCSRDirFlag(cmd.Flags(), &flags.csrPath)
	cmd.Flags().BoolVar(&flags.useAPI, "use-api", flags.useAPI, "Use the Kubernetes certificate API to renew certificates")
}

func renewCert(flags *renewFlags, kdir string, handler *renewal.CertificateRenewHandler) {
	internalcfg, err := configutil.LoadOrDefaultInitConfiguration(flags.cfgPath, &kubeadmapiv1beta2.InitConfiguration{}, &flags.cfg)
	kubeadmutil.CheckErr(err)

	// Get a renewal manager for the given cluster configuration
	rm, err := renewal.NewManager(&internalcfg.ClusterConfiguration, kdir)
	kubeadmutil.CheckErr(err)

	// if the renewal operation is set to generate CSR request only
	if flags.csrOnly {
		// checks a path for storing CSR request is given
		if flags.csrPath == "" {
			kubeadmutil.CheckErr(errors.New("please provide a path where CSR request should be stored"))
		}
		err := rm.CreateRenewCSR(handler.Name, flags.csrPath)
		kubeadmutil.CheckErr(err)
		return
	}

	// otherwise, the renewal operation has to actually renew a certificate

	// renew the certificate using the requested renew method
	if flags.useAPI {
		// renew using K8s certificate API
		kubeConfigPath := cmdutil.GetKubeConfigPath(flags.kubeconfigPath)
		client, err := kubeconfigutil.ClientSetFromFile(kubeConfigPath)
		kubeadmutil.CheckErr(err)

		err = rm.RenewUsingCSRAPI(handler.Name, client)
		kubeadmutil.CheckErr(err)
	} else {
		// renew using local certificate authorities.
		// this operation can't complete in case the certificate key is not provided (external CA)
		renewed, err := rm.RenewUsingLocalCA(handler.Name)
		kubeadmutil.CheckErr(err)
		if !renewed {
			fmt.Printf("Detected external %s, %s can't be renewed\n", handler.CABaseName, handler.LongName)
			return
		}
	}
	fmt.Printf("%s renewed\n", handler.LongName)
}

// newCmdCertsExpiration creates a new `cert check-expiration` command.
func newCmdCertsExpiration(out io.Writer, kdir string) *cobra.Command {
	flags := &expirationFlags{
		cfg: kubeadmapiv1beta2.ClusterConfiguration{
			// Setting kubernetes version to a default value in order to allow a not necessary internet lookup
			KubernetesVersion: constants.CurrentKubernetesVersion.String(),
		},
	}
	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(&flags.cfg)

	cmd := &cobra.Command{
		Use:   "check-expiration",
		Short: "Check certificates expiration for a Kubernetes cluster",
		Long:  expirationLongDesc,
		Run: func(cmd *cobra.Command, args []string) {
			internalcfg, err := configutil.LoadOrDefaultInitConfiguration(flags.cfgPath, &kubeadmapiv1beta2.InitConfiguration{}, &flags.cfg)
			kubeadmutil.CheckErr(err)

			// Get a renewal manager for the given cluster configuration
			rm, err := renewal.NewManager(&internalcfg.ClusterConfiguration, kdir)
			kubeadmutil.CheckErr(err)

			// Get all the certificate expiration info
			yesNo := func(b bool) string {
				if b {
					return "yes"
				}
				return "no"
			}
			w := tabwriter.NewWriter(out, 10, 4, 3, ' ', 0)
			fmt.Fprintln(w, "CERTIFICATE\tEXPIRES\tRESIDUAL TIME\tEXTERNALLY MANAGED")
			for _, handler := range rm.Certificates() {
				e, err := rm.GetExpirationInfo(handler.Name)
				if err != nil {
					kubeadmutil.CheckErr(err)
				}

				s := fmt.Sprintf("%s\t%s\t%s\t%-8v",
					e.Name,
					e.ExpirationDate.Format("Jan 02, 2006 15:04 MST"),
					duration.ShortHumanDuration(e.ResidualTime()),
					yesNo(e.ExternallyManaged),
				)

				fmt.Fprintln(w, s)
			}
			w.Flush()
		},
	}
	addExpirationFlags(cmd, flags)

	return cmd
}

type expirationFlags struct {
	cfgPath string
	cfg     kubeadmapiv1beta2.ClusterConfiguration
}

func addExpirationFlags(cmd *cobra.Command, flags *expirationFlags) {
	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	options.AddCertificateDirFlag(cmd.Flags(), &flags.cfg.CertificatesDir)
}
