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

package upgrade

import (
	"fmt"
	"io"
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	commonphases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade/apply"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// applyFlags holds the information about the flags that can be passed to apply
type applyFlags struct {
	*applyPlanFlags

	nonInteractiveMode bool
	force              bool
	dryRun             bool
	renewCerts         bool
	patchesDir         string
}

// compile-time assert that the local data object satisfies the phases data interface.
var _ phases.Data = &applyData{}

// applyData defines all the runtime information used when running the kubeadm upgrade apply workflow;
// this data is shared across all the phases that are included in the workflow.
type applyData struct {
	nonInteractiveMode        bool
	force                     bool
	dryRun                    bool
	etcdUpgrade               bool
	renewCerts                bool
	allowExperimentalUpgrades bool
	allowRCUpgrades           bool
	printConfig               bool
	cfg                       *kubeadmapi.UpgradeConfiguration
	initCfg                   *kubeadmapi.InitConfiguration
	client                    clientset.Interface
	patchesDir                string
	ignorePreflightErrors     sets.Set[string]
	outputWriter              io.Writer
}

// newCmdApply returns the cobra command for `kubeadm upgrade apply`
func newCmdApply(apf *applyPlanFlags) *cobra.Command {
	flags := &applyFlags{
		applyPlanFlags: apf,
		renewCerts:     true,
	}

	applyRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:                   "apply [version]",
		DisableFlagsInUseLine: true,
		Short:                 "Upgrade your Kubernetes cluster to the specified version",
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
				return err
			}

			data, err := applyRunner.InitData(args)
			if err != nil {
				return err
			}
			applyData, ok := data.(*applyData)
			if !ok {
				return errors.New("invalid data struct")
			}
			if err := applyRunner.Run(args); err != nil {
				return err
			}
			if flags.dryRun {
				fmt.Println("[upgrade/successful] Finished dryrunning successfully!")
				return nil
			}

			fmt.Println("")
			fmt.Printf("[upgrade] SUCCESS! A control plane node of your cluster was upgraded to %q.\n\n", applyData.InitCfg().KubernetesVersion)
			fmt.Println("[upgrade] Now please proceed with upgrading the rest of the nodes by following the right order.")

			return nil
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, options.Force, "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, options.DryRun, flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.renewCerts, options.CertificateRenewal, flags.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	options.AddPatchesFlag(cmd.Flags(), &flags.patchesDir)

	// Initialize the workflow runner with the list of phases
	applyRunner.AppendPhase(phases.NewPreflightPhase())
	applyRunner.AppendPhase(phases.NewControlPlanePhase())
	applyRunner.AppendPhase(phases.NewUploadConfigPhase())
	applyRunner.AppendPhase(phases.NewKubeconfigPhase())
	applyRunner.AppendPhase(commonphases.NewKubeletConfigPhase())
	applyRunner.AppendPhase(phases.NewBootstrapTokenPhase())
	applyRunner.AppendPhase(commonphases.NewAddonPhase())
	applyRunner.AppendPhase(commonphases.NewPostUpgradePhase())

	// Sets the data builder function, that will be used by the runner,
	// both when running the entire workflow or single phases.
	applyRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		data, err := newApplyData(cmd, args, flags)
		if err != nil {
			return nil, err
		}
		// If the flag for skipping phases was empty, use the values from config
		if len(applyRunner.Options.SkipPhases) == 0 {
			applyRunner.Options.SkipPhases = data.cfg.Apply.SkipPhases
		}
		return data, nil
	})

	// Binds the Runner to kubeadm upgrade apply command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands.
	applyRunner.BindToCommand(cmd)

	return cmd
}

// newApplyData returns a new applyData struct to be used for the execution of the kubeadm upgrade apply workflow.
func newApplyData(cmd *cobra.Command, args []string, applyFlags *applyFlags) (*applyData, error) {
	externalCfg := &v1beta4.UpgradeConfiguration{}
	opt := configutil.LoadOrDefaultConfigurationOptions{}
	upgradeCfg, err := configutil.LoadOrDefaultUpgradeConfiguration(applyFlags.cfgPath, externalCfg, opt)
	if err != nil {
		return nil, err
	}

	upgradeVersion := upgradeCfg.Apply.KubernetesVersion
	// The version arg is mandatory, unless it's specified in the config file.
	if upgradeVersion == "" {
		if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
			return nil, err
		}
	}

	// If the version was specified in both the arg and config file, the arg will overwrite the config file.
	if len(args) == 1 {
		upgradeVersion = args[0]
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(applyFlags.ignorePreflightErrors, upgradeCfg.Apply.IgnorePreflightErrors)
	if err != nil {
		return nil, err
	}

	force, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.Force, upgradeCfg.Apply.ForceUpgrade, &applyFlags.force).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("forceUpgrade", "bool")
	}

	dryRun, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.DryRun, upgradeCfg.Apply.DryRun, &applyFlags.dryRun).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("dryRun", "bool")
	}

	etcdUpgrade, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.EtcdUpgrade, upgradeCfg.Apply.EtcdUpgrade, &applyFlags.etcdUpgrade).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("etcdUpgrade", "bool")
	}

	renewCerts, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.CertificateRenewal, upgradeCfg.Apply.CertificateRenewal, &applyFlags.renewCerts).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("certificateRenewal", "bool")
	}

	allowExperimentalUpgrades, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), "allow-experimental-upgrades", upgradeCfg.Apply.AllowExperimentalUpgrades, &applyFlags.allowExperimentalUpgrades).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("allowExperimentalUpgrades", "bool")
	}

	allowRCUpgrades, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), "allow-release-candidate-upgrades", upgradeCfg.Apply.AllowRCUpgrades, &applyFlags.allowRCUpgrades).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("allowRCUpgrades", "bool")
	}

	printConfig, ok := cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), "print-config", upgradeCfg.Apply.PrintConfig, &applyFlags.printConfig).(*bool)
	if !ok {
		return nil, cmdutil.TypeMismatchErr("printConfig", "bool")
	}

	printer := &output.TextPrinter{}

	client, err := getClient(applyFlags.kubeConfigPath, *dryRun, printer)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", applyFlags.kubeConfigPath)
	}

	// Fetches the cluster configuration.
	klog.V(1).Infoln("[upgrade] retrieving configuration from cluster")
	initCfg, err := configutil.FetchInitConfigurationFromCluster(client, nil, "upgrade", false, false)
	if err != nil {
		if apierrors.IsNotFound(err) {
			_, _ = printer.Printf("[upgrade] In order to upgrade, a ConfigMap called %q in the %q namespace must exist.\n", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			_, _ = printer.Printf("[upgrade] Use 'kubeadm init phase upload-config --config your-config.yaml' to re-upload it.\n")
			err = errors.Errorf("the ConfigMap %q in the %q namespace was not found", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
		}
		return nil, errors.Wrap(err, "[upgrade] FATAL")
	}

	// Also set the union of pre-flight errors to InitConfiguration, to provide a consistent view of the runtime configuration:
	initCfg.NodeRegistration.IgnorePreflightErrors = sets.List(ignorePreflightErrorsSet)

	// Set the ImagePullPolicy and ImagePullSerial from the UpgradeApplyConfiguration to the InitConfiguration.
	// These are used by preflight.RunPullImagesCheck() when running 'apply'.
	initCfg.NodeRegistration.ImagePullPolicy = upgradeCfg.Apply.ImagePullPolicy
	initCfg.NodeRegistration.ImagePullSerial = upgradeCfg.Apply.ImagePullSerial

	// The `upgrade apply` version always overwrites the KubernetesVersion in the returned cfg with the target
	// version. While this is not the same for `upgrade plan` where the KubernetesVersion should be the old
	// one (because the call to getComponentConfigVersionStates requires the currently installed version).
	// This also makes the KubernetesVersion value returned for `upgrade plan` consistent as that command
	// allows to not specify a target version in which case KubernetesVersion will always hold the currently
	// installed one.
	initCfg.KubernetesVersion = upgradeVersion

	var patchesDir string
	if upgradeCfg.Apply.Patches != nil {
		patchesDir = cmdutil.ValueFromFlagsOrConfig(cmd.Flags(), options.Patches, upgradeCfg.Apply.Patches.Directory, applyFlags.patchesDir).(string)
	} else {
		patchesDir = applyFlags.patchesDir
	}

	if *printConfig {
		printConfiguration(&initCfg.ClusterConfiguration, os.Stdout, printer)
	}

	return &applyData{
		nonInteractiveMode:        applyFlags.nonInteractiveMode,
		force:                     *force,
		dryRun:                    *dryRun,
		etcdUpgrade:               *etcdUpgrade,
		renewCerts:                *renewCerts,
		allowExperimentalUpgrades: *allowExperimentalUpgrades,
		allowRCUpgrades:           *allowRCUpgrades,
		printConfig:               *printConfig,
		cfg:                       upgradeCfg,
		initCfg:                   initCfg,
		client:                    client,
		patchesDir:                patchesDir,
		ignorePreflightErrors:     ignorePreflightErrorsSet,
		outputWriter:              applyFlags.out,
	}, nil
}

// DryRun returns the dryRun flag.
func (d *applyData) DryRun() bool {
	return d.dryRun
}

// EtcdUpgrade returns the etcdUpgrade flag.
func (d *applyData) EtcdUpgrade() bool {
	return d.etcdUpgrade
}

// RenewCerts returns the renewCerts flag.
func (d *applyData) RenewCerts() bool {
	return d.renewCerts
}

// Cfg returns the UpgradeConfiguration.
func (d *applyData) Cfg() *kubeadmapi.UpgradeConfiguration {
	return d.cfg
}

// InitCfg returns the InitConfiguration.
func (d *applyData) InitCfg() *kubeadmapi.InitConfiguration {
	return d.initCfg
}

// Client returns a Kubernetes client to be used by kubeadm.
func (d *applyData) Client() clientset.Interface {
	return d.client
}

// PatchesDir returns the folder where patches for components are stored.
func (d *applyData) PatchesDir() string {
	return d.patchesDir
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (d *applyData) IgnorePreflightErrors() sets.Set[string] {
	return d.ignorePreflightErrors
}

// OutputWriter returns the output writer to be used by kubeadm.
func (d *applyData) OutputWriter() io.Writer {
	return d.outputWriter
}

// SessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run).
func (d *applyData) SessionIsInteractive() bool {
	return !(d.nonInteractiveMode || d.dryRun || d.force)
}

// AllowExperimentalUpgrades returns true if upgrading to an alpha/beta/release candidate version of Kubernetes is allowed.
func (d *applyData) AllowExperimentalUpgrades() bool {
	return d.allowExperimentalUpgrades
}

// AllowRCUpgrades returns true if upgrading to a release candidate version of Kubernetes is allowed.
func (d *applyData) AllowRCUpgrades() bool {
	return d.allowRCUpgrades
}

// ForceUpgrade returns true if force-upgrading is enabled.
func (d *applyData) ForceUpgrade() bool {
	return d.force
}

// IsControlPlaneNode returns if the node is a control-plane node.
func (d *applyData) IsControlPlaneNode() bool {
	// `kubeadm upgrade apply` should always be executed on a control-plane node
	return true
}
