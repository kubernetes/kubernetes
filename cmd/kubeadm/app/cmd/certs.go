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

package cmd

import (
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	outputapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/copycerts"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

var (
	genericCertRenewLongDesc = cmdutil.LongDesc(`
	Renew the %s.

	Renewals run unconditionally, regardless of certificate expiration date; extra attributes such as SANs will
	be based on the existing file/certificates, there is no need to resupply them.

	Renewal by default tries to use the certificate authority in the local PKI managed by kubeadm; as alternative
	it is possible to use K8s certificate API for certificate renewal, or as a last option, to generate a CSR request.

	After renewal, in order to make changes effective, is required to restart control-plane components and
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
	generateCSRLongDesc = cmdutil.LongDesc(`
	Generates keys and certificate signing requests (CSRs) for all the certificates required to run the control plane.
	This command also generates partial kubeconfig files with private key data in the  "users > user > client-key-data" field,
	and for each kubeconfig file an accompanying ".csr" file is created.

	This command is designed for use in [Kubeadm External CA Mode](https://kubernetes.io/docs/tasks/administer-cluster/kubeadm/kubeadm-certs/#external-ca-mode).
	It generates CSRs which you can then submit to your external certificate authority for signing.

	The PEM encoded signed certificates should then be saved alongside the key files, using ".crt" as the file extension,
	or in the case of kubeconfig files, the PEM encoded signed certificate should be base64 encoded
	and added to the kubeconfig file in the "users > user > client-certificate-data" field.
`)
	generateCSRExample = cmdutil.Examples(`
	# The following command will generate keys and CSRs for all control-plane certificates and kubeconfig files:
	kubeadm certs generate-csr --kubeconfig-dir /tmp/etc-k8s --cert-dir /tmp/etc-k8s/pki
`)
)

// newCmdCertsUtility returns main command for certs phase
func newCmdCertsUtility(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Commands related to handling kubernetes certificates",
		Run:     cmdutil.SubCmdRun(),
	}

	cmd.AddCommand(newCmdCertsRenewal(out))
	cmd.AddCommand(newCmdCertsExpiration(out, kubeadmconstants.KubernetesDir))
	cmd.AddCommand(newCmdCertificateKey())
	cmd.AddCommand(newCmdGenCSR(out))
	return cmd
}

// genCSRConfig is the configuration required by the gencsr command
type genCSRConfig struct {
	kubeadmConfigPath string
	certDir           string
	kubeConfigDir     string
	kubeadmConfig     *kubeadmapi.InitConfiguration
}

func newGenCSRConfig() *genCSRConfig {
	return &genCSRConfig{
		kubeConfigDir: kubeadmconstants.KubernetesDir,
	}
}

func (o *genCSRConfig) addFlagSet(flagSet *pflag.FlagSet) {
	options.AddConfigFlag(flagSet, &o.kubeadmConfigPath)
	options.AddCertificateDirFlag(flagSet, &o.certDir)
	options.AddKubeConfigDirFlag(flagSet, &o.kubeConfigDir)
}

// load merges command line flag values into kubeadm's config.
// Reads Kubeadm config from a file (if present)
// else use dynamically generated default config.
// This configuration contains the DNS names and IP addresses which
// are encoded in the control-plane CSRs.
func (o *genCSRConfig) load() (err error) {
	o.kubeadmConfig, err = configutil.LoadOrDefaultInitConfiguration(
		o.kubeadmConfigPath,
		&kubeadmapiv1.InitConfiguration{},
		&kubeadmapiv1.ClusterConfiguration{},
		configutil.LoadOrDefaultConfigurationOptions{
			SkipCRIDetect: true,
		},
	)
	if err != nil {
		return err
	}
	// --cert-dir takes priority over kubeadm config if set.
	if o.certDir != "" {
		o.kubeadmConfig.CertificatesDir = o.certDir
	}
	return nil
}

// newCmdGenCSR returns cobra.Command for generating keys and CSRs
func newCmdGenCSR(out io.Writer) *cobra.Command {
	config := newGenCSRConfig()

	cmd := &cobra.Command{
		Use:     "generate-csr",
		Short:   "Generate keys and certificate signing requests",
		Long:    generateCSRLongDesc,
		Example: generateCSRExample,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := config.load(); err != nil {
				return err
			}
			return runGenCSR(out, config)
		},
	}
	config.addFlagSet(cmd.Flags())
	return cmd
}

// runGenCSR contains the logic of the generate-csr sub-command.
func runGenCSR(out io.Writer, config *genCSRConfig) error {
	if err := certsphase.CreateDefaultKeysAndCSRFiles(out, config.kubeadmConfig); err != nil {
		return err
	}
	if err := kubeconfigphase.CreateDefaultKubeConfigsAndCSRFiles(out, config.kubeConfigDir, config.kubeadmConfig); err != nil {
		return err
	}
	return nil
}

// newCmdCertificateKey returns cobra.Command for certificate key generate
func newCmdCertificateKey() *cobra.Command {
	return &cobra.Command{
		Use:   "certificate-key",
		Short: "Generate certificate keys",
		Long:  certificateKeyLongDesc,

		RunE: func(cmd *cobra.Command, args []string) error {
			key, err := copycerts.CreateCertificateKey()
			if err != nil {
				return err
			}
			fmt.Println(key)
			return nil
		},
		Args: cobra.NoArgs,
	}
}

// newCmdCertsRenewal creates a new `cert renew` command.
func newCmdCertsRenewal(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "renew",
		Short: "Renew certificates for a Kubernetes cluster",
		Long:  cmdutil.MacroCommandLongDescription,
		Run:   cmdutil.SubCmdRun(),
	}

	cmd.AddCommand(getRenewSubCommands(out, kubeadmconstants.KubernetesDir)...)

	return cmd
}

type renewFlags struct {
	cfgPath        string
	kubeconfigPath string
	cfg            kubeadmapiv1.ClusterConfiguration
}

func getRenewSubCommands(out io.Writer, kdir string) []*cobra.Command {
	flags := &renewFlags{
		cfg: kubeadmapiv1.ClusterConfiguration{
			// Setting kubernetes version to a default value in order to allow a not necessary internet lookup
			KubernetesVersion: kubeadmconstants.CurrentKubernetesVersion.String(),
		},
		kubeconfigPath: kubeadmconstants.GetAdminKubeConfigPath(),
	}
	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(&flags.cfg)

	// Get a renewal manager for a generic Cluster configuration, that is used only for getting
	// the list of certificates for building subcommands
	rm, err := renewal.NewManager(&kubeadmapi.ClusterConfiguration{}, "")
	if err != nil {
		return nil
	}

	cmdList := []*cobra.Command{}
	for _, handler := range rm.Certificates() {
		// get the cobra.Command skeleton for this command
		cmd := &cobra.Command{
			Use:   handler.Name,
			Short: fmt.Sprintf("Renew the %s", handler.LongName),
			Long:  fmt.Sprintf(genericCertRenewLongDesc, handler.LongName),
		}
		addRenewFlags(cmd, flags)
		// get the implementation of renewing this certificate
		renewalFunc := func(handler *renewal.CertificateRenewHandler) func() error {
			return func() error {
				// Get cluster configuration (from --config, kubeadm-config ConfigMap, or default as a fallback)
				client, _ := kubeconfigutil.ClientSetFromFile(flags.kubeconfigPath)
				internalcfg, err := getInternalCfg(flags.cfgPath, client, flags.cfg, &output.TextPrinter{}, "renew")
				if err != nil {
					return err
				}

				return renewCert(kdir, internalcfg, handler)
			}
		}(handler)
		// install the implementation into the command
		cmd.RunE = func(*cobra.Command, []string) error { return renewalFunc() }
		cmd.Args = cobra.NoArgs
		cmdList = append(cmdList, cmd)
	}

	allCmd := &cobra.Command{
		Use:   "all",
		Short: "Renew all available certificates",
		Long:  allLongDesc,
		RunE: func(*cobra.Command, []string) error {
			// Get cluster configuration (from --config, kubeadm-config ConfigMap, or default as a fallback)
			client, _ := kubeconfigutil.ClientSetFromFile(flags.kubeconfigPath)
			internalcfg, err := getInternalCfg(flags.cfgPath, client, flags.cfg, &output.TextPrinter{}, "renew")
			if err != nil {
				return err
			}

			// Get a renewal manager for an actual Cluster configuration
			rm, err := renewal.NewManager(&internalcfg.ClusterConfiguration, kdir)
			if err != nil {
				return err
			}

			// Renew certificates
			for _, handler := range rm.Certificates() {
				if err := renewCert(kdir, internalcfg, handler); err != nil {
					return err
				}
			}
			fmt.Printf("\nDone renewing certificates. You must restart the kube-apiserver, kube-controller-manager, kube-scheduler and etcd, so that they can use the new certificates.\n")
			return nil
		},
		Args: cobra.NoArgs,
	}
	addRenewFlags(allCmd, flags)

	cmdList = append(cmdList, allCmd)
	return cmdList
}

func addRenewFlags(cmd *cobra.Command, flags *renewFlags) {
	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	options.AddCertificateDirFlag(cmd.Flags(), &flags.cfg.CertificatesDir)
	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeconfigPath)
}

func renewCert(kdir string, internalcfg *kubeadmapi.InitConfiguration, handler *renewal.CertificateRenewHandler) error {
	// Get a renewal manager for the given cluster configuration
	rm, err := renewal.NewManager(&internalcfg.ClusterConfiguration, kdir)
	if err != nil {
		return err
	}

	if ok, _ := rm.CertificateExists(handler.Name); !ok {
		fmt.Printf("MISSING! %s\n", handler.LongName)
		return nil
	}

	// renew using local certificate authorities.
	// this operation can't complete in case the certificate key is not provided (external CA)
	renewed, err := rm.RenewUsingLocalCA(handler.Name)
	if err != nil {
		return err
	}
	if !renewed {
		fmt.Printf("Detected external %s, %s can't be renewed\n", handler.CABaseName, handler.LongName)
		return nil
	}
	fmt.Printf("%s renewed\n", handler.LongName)
	return nil
}

func getInternalCfg(cfgPath string, client kubernetes.Interface, cfg kubeadmapiv1.ClusterConfiguration, printer output.Printer, logPrefix string) (*kubeadmapi.InitConfiguration, error) {
	// In case the user is not providing a custom config, try to get current config from the cluster.
	// NB. this operation should not block, because we want to allow certificate renewal also in case of not-working clusters
	if cfgPath == "" && client != nil {
		internalcfg, err := configutil.FetchInitConfigurationFromCluster(client, printer, logPrefix, false, false)
		if err == nil {
			printer.Println() // add empty line to separate the FetchInitConfigurationFromCluster output from the command output
			// certificate renewal or expiration checking doesn't depend on a running cluster, which means the CertificatesDir
			// could be set to a value other than the default value or the value fetched from the cluster.
			// cfg.CertificatesDir could be empty if the default value is set to empty (not true today).
			if len(cfg.CertificatesDir) != 0 {
				klog.V(1).Infof("Overriding the cluster certificate directory with the value from command line flag --%s: %s", options.CertificatesDir, cfg.CertificatesDir)
				internalcfg.ClusterConfiguration.CertificatesDir = cfg.CertificatesDir
			}

			return internalcfg, nil
		}
		printer.Printf("[%s] Error reading configuration from the Cluster. Falling back to default configuration\n\n", logPrefix)
	}

	// Read config from --config if provided. Otherwise, use the default configuration
	return configutil.LoadOrDefaultInitConfiguration(cfgPath, &kubeadmapiv1.InitConfiguration{}, &cfg, configutil.LoadOrDefaultConfigurationOptions{
		SkipCRIDetect: true,
	})
}

// fetchCertificateExpirationInfo returns the certificate expiration info for the given renewal manager
func fetchCertificateExpirationInfo(rm *renewal.Manager) (*outputapiv1alpha3.CertificateExpirationInfo, error) {
	info := &outputapiv1alpha3.CertificateExpirationInfo{}

	for _, handler := range rm.Certificates() {
		if ok, _ := rm.CertificateExists(handler.Name); ok {
			e, err := rm.GetCertificateExpirationInfo(handler.Name)
			if err != nil {
				return nil, err
			}
			info.Certificates = append(info.Certificates, outputapiv1alpha3.Certificate{
				Name:                e.Name,
				ExpirationDate:      metav1.Time{Time: e.ExpirationDate},
				ResidualTimeSeconds: int64(e.ResidualTime() / time.Second),
				CAName:              handler.CAName,
				ExternallyManaged:   e.ExternallyManaged,
			})
		} else {
			// the certificate does not exist (for any reason)
			info.Certificates = append(info.Certificates, outputapiv1alpha3.Certificate{
				Name:    handler.Name,
				Missing: true,
			})
		}
	}

	for _, handler := range rm.CAs() {
		if ok, _ := rm.CAExists(handler.Name); ok {
			e, err := rm.GetCAExpirationInfo(handler.Name)
			if err != nil {
				return nil, err
			}
			info.CertificateAuthorities = append(info.CertificateAuthorities, outputapiv1alpha3.Certificate{
				Name:                e.Name,
				ExpirationDate:      metav1.Time{Time: e.ExpirationDate},
				ResidualTimeSeconds: int64(e.ResidualTime() / time.Second),
				ExternallyManaged:   e.ExternallyManaged,
			})
		} else {
			// the CA does not exist (for any reason)
			info.CertificateAuthorities = append(info.CertificateAuthorities, outputapiv1alpha3.Certificate{
				Name:    handler.Name,
				Missing: true,
			})
		}
	}

	return info, nil
}

// clientSetFromFile is a variable that holds the function to create a clientset from a kubeconfig file.
// It is used for testing purposes.
var clientSetFromFile = kubeconfigutil.ClientSetFromFile

// newCmdCertsExpiration creates a new `cert check-expiration` command.
func newCmdCertsExpiration(out io.Writer, kdir string) *cobra.Command {
	flags := &expirationFlags{
		cfg: kubeadmapiv1.ClusterConfiguration{
			// Setting kubernetes version to a default value in order to allow a not necessary internet lookup
			KubernetesVersion: kubeadmconstants.CurrentKubernetesVersion.String(),
		},
		kubeconfigPath: kubeadmconstants.GetAdminKubeConfigPath(),
	}
	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(&flags.cfg)

	outputFlags := output.NewOutputFlags(&certTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(output.TextOutput)

	cmd := &cobra.Command{
		Use:   "check-expiration",
		Short: "Check certificates expiration for a Kubernetes cluster",
		Long:  expirationLongDesc,
		RunE: func(cmd *cobra.Command, args []string) error {
			printer, err := outputFlags.ToPrinter()
			if err != nil {
				return errors.Wrap(err, "could not construct output printer")
			}

			// Get cluster configuration (from --config, kubeadm-config ConfigMap, or default as a fallback)
			client, _ := clientSetFromFile(flags.kubeconfigPath)
			internalcfg, err := getInternalCfg(flags.cfgPath, client, flags.cfg, printer, "check-expiration")
			if err != nil {
				return err
			}

			// Get a renewal manager for the given cluster configuration
			rm, err := renewal.NewManager(&internalcfg.ClusterConfiguration, kdir)
			if err != nil {
				return err
			}

			info, err := fetchCertificateExpirationInfo(rm)
			if err != nil {
				return err
			}
			return printer.PrintObj(info, out)
		},
		Args: cobra.NoArgs,
	}
	addExpirationFlags(cmd, flags)
	outputFlags.AddFlags(cmd)
	return cmd
}

type expirationFlags struct {
	cfgPath        string
	kubeconfigPath string
	cfg            kubeadmapiv1.ClusterConfiguration
}

func addExpirationFlags(cmd *cobra.Command, flags *expirationFlags) {
	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	options.AddCertificateDirFlag(cmd.Flags(), &flags.cfg.CertificatesDir)
	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeconfigPath)
}

// certsTextPrinter prints all certificates in a text form
type certTextPrinter struct {
	output.TextPrinter
}

// PrintObj is an implementation of ResourcePrinter.PrintObj for plain text output
func (p *certTextPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	info, ok := obj.(*outputapiv1alpha3.CertificateExpirationInfo)
	if !ok {
		return errors.New("unexpected type")
	}

	yesNo := func(b bool) string {
		if b {
			return "yes"
		}
		return "no"
	}

	tabw := tabwriter.NewWriter(writer, 10, 4, 3, ' ', 0)
	fmt.Fprintln(tabw, "CERTIFICATE\tEXPIRES\tRESIDUAL TIME\tCERTIFICATE AUTHORITY\tEXTERNALLY MANAGED")
	for _, cert := range info.Certificates {
		if cert.Missing {
			s := fmt.Sprintf("!MISSING! %s\t\t\t\t", cert.Name)
			fmt.Fprintln(tabw, s)
			continue
		}

		s := fmt.Sprintf("%s\t%s\t%s\t%s\t%-8v",
			cert.Name,
			cert.ExpirationDate.Format("Jan 02, 2006 15:04 MST"),
			duration.ShortHumanDuration(time.Duration(cert.ResidualTimeSeconds)*time.Second),
			cert.CAName,
			yesNo(cert.ExternallyManaged),
		)
		fmt.Fprintln(tabw, s)
	}

	fmt.Fprintln(tabw)
	fmt.Fprintln(tabw, "CERTIFICATE AUTHORITY\tEXPIRES\tRESIDUAL TIME\tEXTERNALLY MANAGED")
	for _, ca := range info.CertificateAuthorities {
		if ca.Missing {
			s := fmt.Sprintf("!MISSING! %s\t\t\t", ca.Name)
			fmt.Fprintln(tabw, s)
			continue
		}

		s := fmt.Sprintf("%s\t%s\t%s\t%-8v",
			ca.Name,
			ca.ExpirationDate.Format("Jan 02, 2006 15:04 MST"),
			duration.ShortHumanDuration(time.Duration(ca.ResidualTimeSeconds)*time.Second),
			yesNo(ca.ExternallyManaged),
		)
		fmt.Fprintln(tabw, s)
	}
	return tabw.Flush()
}

// certTextPrintFlags provides flags necessary for printing
type certTextPrintFlags struct{}

// ToPrinter returns a kubeadm printer for the text output format
func (tpf *certTextPrintFlags) ToPrinter(outputFormat string) (output.Printer, error) {
	if outputFormat == output.TextOutput {
		return &certTextPrinter{}, nil
	}
	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: []string{output.TextOutput}}
}
