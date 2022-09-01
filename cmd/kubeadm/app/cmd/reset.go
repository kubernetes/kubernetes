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

package cmd

import (
	"fmt"
	"io"
	"path"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/reset"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
)

var (
	iptablesCleanupInstructions = dedent.Dedent(`
		The reset process does not reset or clean up iptables rules or IPVS tables.
		If you wish to reset iptables, you must do so manually by using the "iptables" command.

		If your cluster was setup to utilize IPVS, run ipvsadm --clear (or similar)
		to reset your system's IPVS tables.

		The reset process does not clean your kubeconfig files and you must remove them manually.
		Please, check the contents of the $HOME/.kube/config file.
	`)

	cniCleanupInstructions = dedent.Dedent(`
		The reset process does not clean CNI configuration. To do so, you must remove /etc/cni/net.d
	`)
)

// resetOptions defines all the options exposed via flags by kubeadm reset.
type resetOptions struct {
	certificatesDir       string
	criSocketPath         string
	forceReset            bool
	ignorePreflightErrors []string
	kubeconfigPath        string
	dryRun                bool
	cleanupTmpDir         bool
}

// resetData defines all the runtime information used when running the kubeadm reset workflow;
// this data is shared across all the phases that are included in the workflow.
type resetData struct {
	certificatesDir       string
	client                clientset.Interface
	criSocketPath         string
	forceReset            bool
	ignorePreflightErrors sets.String
	inputReader           io.Reader
	outputWriter          io.Writer
	cfg                   *kubeadmapi.InitConfiguration
	dryRun                bool
	cleanupTmpDir         bool
}

// newResetOptions returns a struct ready for being used for creating cmd join flags.
func newResetOptions() *resetOptions {
	return &resetOptions{
		certificatesDir: kubeadmapiv1.DefaultCertificatesDir,
		forceReset:      false,
		kubeconfigPath:  kubeadmconstants.GetAdminKubeConfigPath(),
		cleanupTmpDir:   false,
	}
}

// newResetData returns a new resetData struct to be used for the execution of the kubeadm reset workflow.
func newResetData(cmd *cobra.Command, options *resetOptions, in io.Reader, out io.Writer) (*resetData, error) {
	var cfg *kubeadmapi.InitConfiguration

	client, err := cmdutil.GetClientset(options.kubeconfigPath, false)
	if err == nil {
		klog.V(1).Infof("[reset] Loaded client set from kubeconfig file: %s", options.kubeconfigPath)
		cfg, err = configutil.FetchInitConfigurationFromCluster(client, nil, "reset", false, false)
		if err != nil {
			klog.Warningf("[reset] Unable to fetch the kubeadm-config ConfigMap from cluster: %v", err)
		}
	} else {
		klog.V(1).Infof("[reset] Could not obtain a client set from the kubeconfig file: %s", options.kubeconfigPath)
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(options.ignorePreflightErrors, ignorePreflightErrors(cfg))
	if err != nil {
		return nil, err
	}
	if cfg != nil {
		// Also set the union of pre-flight errors to InitConfiguration, to provide a consistent view of the runtime configuration:
		cfg.NodeRegistration.IgnorePreflightErrors = ignorePreflightErrorsSet.List()
	}

	var criSocketPath string
	if options.criSocketPath == "" {
		criSocketPath, err = resetDetectCRISocket(cfg)
		if err != nil {
			return nil, err
		}
		klog.V(1).Infof("[reset] Detected and using CRI socket: %s", criSocketPath)
	} else {
		criSocketPath = options.criSocketPath
		klog.V(1).Infof("[reset] Using specified CRI socket: %s", criSocketPath)
	}

	return &resetData{
		certificatesDir:       options.certificatesDir,
		client:                client,
		criSocketPath:         criSocketPath,
		forceReset:            options.forceReset,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		inputReader:           in,
		outputWriter:          out,
		cfg:                   cfg,
		dryRun:                options.dryRun,
		cleanupTmpDir:         options.cleanupTmpDir,
	}, nil
}

func ignorePreflightErrors(cfg *kubeadmapi.InitConfiguration) []string {
	if cfg == nil {
		return []string{}
	}
	return cfg.NodeRegistration.IgnorePreflightErrors
}

// AddResetFlags adds reset flags
func AddResetFlags(flagSet *flag.FlagSet, resetOptions *resetOptions) {
	flagSet.StringVar(
		&resetOptions.certificatesDir, options.CertificatesDir, resetOptions.certificatesDir,
		`The path to the directory where the certificates are stored. If specified, clean this directory.`,
	)
	flagSet.BoolVarP(
		&resetOptions.forceReset, options.ForceReset, "f", false,
		"Reset the node without prompting for confirmation.",
	)
	flagSet.BoolVar(
		&resetOptions.dryRun, options.DryRun, resetOptions.dryRun,
		"Don't apply any changes; just output what would be done.",
	)
	flagSet.BoolVar(
		&resetOptions.cleanupTmpDir, options.CleanupTmpDir, resetOptions.cleanupTmpDir,
		fmt.Sprintf("Cleanup the %q directory", path.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.TempDirForKubeadm)),
	)

	options.AddKubeConfigFlag(flagSet, &resetOptions.kubeconfigPath)
	options.AddIgnorePreflightErrorsFlag(flagSet, &resetOptions.ignorePreflightErrors)
	cmdutil.AddCRISocketFlag(flagSet, &resetOptions.criSocketPath)
}

// newCmdReset returns the "kubeadm reset" command
func newCmdReset(in io.Reader, out io.Writer, resetOptions *resetOptions) *cobra.Command {
	if resetOptions == nil {
		resetOptions = newResetOptions()
	}
	resetRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Performs a best effort revert of changes made to this host by 'kubeadm init' or 'kubeadm join'",
		RunE: func(cmd *cobra.Command, args []string) error {
			err := resetRunner.Run(args)
			if err != nil {
				return err
			}

			// output help text instructing user how to remove cni folders
			fmt.Print(cniCleanupInstructions)
			// Output help text instructing user how to remove iptables rules
			fmt.Print(iptablesCleanupInstructions)
			return nil
		},
	}

	AddResetFlags(cmd.Flags(), resetOptions)

	// initialize the workflow runner with the list of phases
	resetRunner.AppendPhase(phases.NewPreflightPhase())
	resetRunner.AppendPhase(phases.NewRemoveETCDMemberPhase())
	resetRunner.AppendPhase(phases.NewCleanupNodePhase())

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	resetRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newResetData(cmd, resetOptions, in, out)
	})

	// binds the Runner to kubeadm init command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	resetRunner.BindToCommand(cmd)

	return cmd
}

// Cfg returns the InitConfiguration.
func (r *resetData) Cfg() *kubeadmapi.InitConfiguration {
	return r.cfg
}

// DryRun returns the dryRun flag.
func (r *resetData) DryRun() bool {
	return r.dryRun
}

// CleanupTmpDir returns the cleanupTmpDir flag.
func (r *resetData) CleanupTmpDir() bool {
	return r.cleanupTmpDir
}

// CertificatesDir returns the CertificatesDir.
func (r *resetData) CertificatesDir() string {
	return r.certificatesDir
}

// Client returns the Client for accessing the cluster.
func (r *resetData) Client() clientset.Interface {
	return r.client
}

// ForceReset returns the forceReset flag.
func (r *resetData) ForceReset() bool {
	return r.forceReset
}

// InputReader returns the io.reader used to read messages.
func (r *resetData) InputReader() io.Reader {
	return r.inputReader
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (r *resetData) IgnorePreflightErrors() sets.String {
	return r.ignorePreflightErrors
}

// CRISocketPath returns the criSocketPath.
func (r *resetData) CRISocketPath() string {
	return r.criSocketPath
}

func resetDetectCRISocket(cfg *kubeadmapi.InitConfiguration) (string, error) {
	if cfg != nil {
		// first try to get the CRI socket from the cluster configuration
		return cfg.NodeRegistration.CRISocket, nil
	}

	// if this fails, try to detect it
	return utilruntime.DetectCRISocket()
}
