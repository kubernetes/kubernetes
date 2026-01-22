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
	"os"
	"path"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/reset"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

var (
	manualCleanupInstructions = dedent.Dedent(`
		The reset process does not perform cleanup of CNI plugin configuration,
		network filtering rules and kubeconfig files.

		For information on how to perform this cleanup manually, please see:
		    https://k8s.io/docs/reference/setup-tools/kubeadm/kubeadm-reset/

	`)
)

// resetOptions defines all the options exposed via flags by kubeadm reset.
type resetOptions struct {
	kubeconfigPath        string
	cfgPath               string
	ignorePreflightErrors []string
	externalcfg           *kubeadmapiv1.ResetConfiguration
	skipCRIDetect         bool
}

// resetData defines all the runtime information used when running the kubeadm reset workflow;
// this data is shared across all the phases that are included in the workflow.
type resetData struct {
	certificatesDir       string
	client                clientset.Interface
	criSocketPath         string
	forceReset            bool
	ignorePreflightErrors sets.Set[string]
	inputReader           io.Reader
	outputWriter          io.Writer
	cfg                   *kubeadmapi.InitConfiguration
	resetCfg              *kubeadmapi.ResetConfiguration
	dryRun                bool
	cleanupTmpDir         bool
}

// newResetOptions returns a struct ready for being used for creating cmd join flags.
func newResetOptions() *resetOptions {
	// initialize the public kubeadm config API by applying defaults
	externalcfg := &kubeadmapiv1.ResetConfiguration{}
	// Apply defaults
	kubeadmscheme.Scheme.Default(externalcfg)
	return &resetOptions{
		kubeconfigPath: kubeadmconstants.GetAdminKubeConfigPath(),
		externalcfg:    externalcfg,
	}
}

// newResetData returns a new resetData struct to be used for the execution of the kubeadm reset workflow.
func newResetData(cmd *cobra.Command, opts *resetOptions, in io.Reader, out io.Writer, allowExperimental bool) (*resetData, error) {
	// Validate the mixed arguments with --config and return early on errors
	if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
		return nil, err
	}

	var (
		initCfg *kubeadmapi.InitConfiguration
		client  clientset.Interface
	)

	// Either use the config file if specified, or convert public kubeadm API to the internal ResetConfiguration and validates cfg.
	resetCfg, err := configutil.LoadOrDefaultResetConfiguration(opts.cfgPath, opts.externalcfg, configutil.LoadOrDefaultConfigurationOptions{
		AllowExperimental: allowExperimental,
		SkipCRIDetect:     opts.skipCRIDetect,
	})
	if err != nil {
		return nil, err
	}

	dryRunFlag := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.DryRun, resetCfg.DryRun, opts.externalcfg.DryRun).(bool)
	if dryRunFlag {
		dryRun := apiclient.NewDryRun().WithDefaultMarshalFunction().WithWriter(os.Stdout)
		dryRun.AppendReactor(dryRun.GetKubeadmConfigReactor()).
			AppendReactor(dryRun.GetKubeletConfigReactor()).
			AppendReactor(dryRun.GetKubeProxyConfigReactor())
		client = dryRun.FakeClient()
		_, err = os.Stat(opts.kubeconfigPath)
		if err == nil {
			err = dryRun.WithKubeConfigFile(opts.kubeconfigPath)
		}
	} else {
		client, err = kubeconfigutil.ClientSetFromFile(opts.kubeconfigPath)
	}

	if err == nil {
		klog.V(1).Infof("[reset] Loaded client set from kubeconfig file: %s", opts.kubeconfigPath)
		getNodeRegistration := true
		getAPIEndpoint := staticpodutil.IsControlPlaneNode()
		getComponentConfigs := true
		initCfg, err = configutil.FetchInitConfigurationFromCluster(client, nil, "reset", getNodeRegistration, getAPIEndpoint, getComponentConfigs)
		if err != nil {
			klog.Warningf("[reset] Unable to fetch the kubeadm-config ConfigMap from cluster: %v", err)
		}
	} else {
		klog.V(1).Infof("[reset] Could not obtain a client set from the kubeconfig file: %s", opts.kubeconfigPath)
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(opts.ignorePreflightErrors, resetCfg.IgnorePreflightErrors)
	if err != nil {
		return nil, err
	}
	if initCfg != nil {
		// Also set the union of pre-flight errors to InitConfiguration, to provide a consistent view of the runtime configuration:
		initCfg.NodeRegistration.IgnorePreflightErrors = sets.List(ignorePreflightErrorsSet)
	}

	criSocketPath := opts.externalcfg.CRISocket
	if criSocketPath == "" {
		criSocketPath, err = resetDetectCRISocket(resetCfg, initCfg)
		if err != nil {
			return nil, err
		}
		klog.V(1).Infof("[reset] Using specified CRI socket: %s", criSocketPath)
	}

	certificatesDir := kubeadmapiv1.DefaultCertificatesDir
	if cmd.Flags().Changed(options.CertificatesDir) { // flag is specified
		certificatesDir = opts.externalcfg.CertificatesDir
	} else if len(resetCfg.CertificatesDir) > 0 { // configured in the ResetConfiguration
		certificatesDir = resetCfg.CertificatesDir
	} else if initCfg != nil && len(initCfg.ClusterConfiguration.CertificatesDir) > 0 { // fetch from cluster
		certificatesDir = initCfg.ClusterConfiguration.CertificatesDir
	}

	return &resetData{
		certificatesDir:       certificatesDir,
		client:                client,
		criSocketPath:         criSocketPath,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		inputReader:           in,
		outputWriter:          out,
		cfg:                   initCfg,
		resetCfg:              resetCfg,
		dryRun:                dryRunFlag,
		forceReset:            cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.Force, resetCfg.Force, opts.externalcfg.Force).(bool),
		cleanupTmpDir:         cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.CleanupTmpDir, resetCfg.CleanupTmpDir, opts.externalcfg.CleanupTmpDir).(bool),
	}, nil
}

// AddResetFlags adds reset flags
func AddResetFlags(flagSet *flag.FlagSet, resetOptions *resetOptions) {
	flagSet.StringVar(
		&resetOptions.externalcfg.CertificatesDir, options.CertificatesDir, kubeadmapiv1.DefaultCertificatesDir,
		`The path to the directory where the certificates are stored. If specified, clean this directory.`,
	)
	flagSet.BoolVarP(
		&resetOptions.externalcfg.Force, options.Force, "f", resetOptions.externalcfg.Force,
		"Reset the node without prompting for confirmation.",
	)
	flagSet.BoolVar(
		&resetOptions.externalcfg.DryRun, options.DryRun, resetOptions.externalcfg.DryRun,
		"Don't apply any changes; just output what would be done.",
	)
	flagSet.BoolVar(
		&resetOptions.externalcfg.CleanupTmpDir, options.CleanupTmpDir, resetOptions.externalcfg.CleanupTmpDir,
		fmt.Sprintf("Cleanup the %q directory", path.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.TempDir)),
	)
	options.AddKubeConfigFlag(flagSet, &resetOptions.kubeconfigPath)
	options.AddConfigFlag(flagSet, &resetOptions.cfgPath)
	options.AddIgnorePreflightErrorsFlag(flagSet, &resetOptions.ignorePreflightErrors)
	cmdutil.AddCRISocketFlag(flagSet, &resetOptions.externalcfg.CRISocket)
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
			data, err := resetRunner.InitData(args)
			if err != nil {
				return err
			}
			if _, ok := data.(*resetData); !ok {
				return errors.New("invalid data struct")
			}
			if err := resetRunner.Run(args); err != nil {
				return err
			}

			fmt.Print(manualCleanupInstructions)
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
		if cmd.Flags().Lookup(options.NodeCRISocket) == nil {
			// skip CRI detection
			// assume that the command execution does not depend on CRISocket when --cri-socket flag is not set
			resetOptions.skipCRIDetect = true
		}
		data, err := newResetData(cmd, resetOptions, in, out, true)
		if err != nil {
			return nil, err
		}
		// If the flag for skipping phases was empty, use the values from config
		if len(resetRunner.Options.SkipPhases) == 0 {
			resetRunner.Options.SkipPhases = data.resetCfg.SkipPhases
		}
		return data, nil
	})

	// binds the Runner to kubeadm reset command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	resetRunner.BindToCommand(cmd)

	return cmd
}

// ResetCfg returns the ResetConfiguration.
func (r *resetData) ResetCfg() *kubeadmapi.ResetConfiguration {
	return r.resetCfg
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
func (r *resetData) IgnorePreflightErrors() sets.Set[string] {
	return r.ignorePreflightErrors
}

// CRISocketPath returns the criSocketPath.
func (r *resetData) CRISocketPath() string {
	return r.criSocketPath
}

func resetDetectCRISocket(resetCfg *kubeadmapi.ResetConfiguration, initCfg *kubeadmapi.InitConfiguration) (string, error) {
	if resetCfg != nil && len(resetCfg.CRISocket) > 0 {
		return resetCfg.CRISocket, nil
	}
	if initCfg != nil && len(initCfg.NodeRegistration.CRISocket) > 0 {
		return initCfg.NodeRegistration.CRISocket, nil
	}

	// try to detect it on host
	return utilruntime.DetectCRISocket()
}
