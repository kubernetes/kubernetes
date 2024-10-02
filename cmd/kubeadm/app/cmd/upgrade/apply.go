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
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	utilsexec "k8s.io/utils/exec"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// applyFlags holds the information about the flags that can be passed to apply
type applyFlags struct {
	*applyPlanFlags

	nonInteractiveMode bool
	force              bool
	dryRun             bool
	etcdUpgrade        bool
	renewCerts         bool
	patchesDir         string
}

// sessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run)
func (f *applyFlags) sessionIsInteractive() bool {
	return !(f.nonInteractiveMode || f.dryRun || f.force)
}

// newCmdApply returns the cobra command for `kubeadm upgrade apply`
func newCmdApply(apf *applyPlanFlags) *cobra.Command {
	flags := &applyFlags{
		applyPlanFlags: apf,
		etcdUpgrade:    true,
		renewCerts:     true,
	}

	cmd := &cobra.Command{
		Use:                   "apply [version]",
		DisableFlagsInUseLine: true,
		Short:                 "Upgrade your Kubernetes cluster to the specified version",
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
				return err
			}
			return runApply(cmd.Flags(), flags, args)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, options.DryRun, flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, options.EtcdUpgrade, flags.etcdUpgrade, "Perform the upgrade of etcd.")
	cmd.Flags().BoolVar(&flags.renewCerts, options.CertificateRenewal, flags.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	options.AddPatchesFlag(cmd.Flags(), &flags.patchesDir)

	return cmd
}

// runApply takes care of the actual upgrade functionality
// It does the following things:
// - Checks if the cluster is healthy
// - Gets the configuration from the kubeadm-config ConfigMap in the cluster
// - Enforces all version skew policies
// - Asks the user if they really want to upgrade
// - Makes sure the control plane images are available locally on the control-plane(s)
// - Upgrades the control plane components
// - Applies the other resources that'd be created with kubeadm init as well, like
//   - Uploads the newly used configuration to the cluster ConfigMap
//   - Creating the RBAC rules for the bootstrap tokens and the cluster-info ConfigMap
//   - Applying new CoreDNS and kube-proxy manifests
func runApply(flagSet *pflag.FlagSet, flags *applyFlags, args []string) error {

	// Start with the basics, verify that the cluster is healthy and get the configuration from the cluster (using the ConfigMap)
	klog.V(1).Infoln("[upgrade/apply] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/apply] retrieving configuration from cluster")
	client, versionGetter, initCfg, upgradeCfg, err := enforceRequirements(flagSet, flags.applyPlanFlags, args, flags.dryRun, true, &output.TextPrinter{})
	if err != nil {
		return err
	}

	// Validate requested and validate actual version
	klog.V(1).Infoln("[upgrade/apply] validating requested and actual version")
	if err := configutil.NormalizeKubernetesVersion(&initCfg.ClusterConfiguration); err != nil {
		return err
	}

	// Use normalized version string in all following code.
	newK8sVersion, err := version.ParseSemantic(initCfg.KubernetesVersion)
	if err != nil {
		return errors.Errorf("unable to parse normalized version %q as a semantic version", initCfg.KubernetesVersion)
	}

	if err := features.ValidateVersion(features.InitFeatureGates, initCfg.FeatureGates, initCfg.KubernetesVersion); err != nil {
		return err
	}

	// Enforce the version skew policies
	klog.V(1).Infoln("[upgrade/version] enforcing version skew policies")
	allowRCUpgrades, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.AllowRCUpgrades, upgradeCfg.Apply.AllowRCUpgrades, &flags.allowRCUpgrades).(*bool)
	if ok {
		flags.allowRCUpgrades = *allowRCUpgrades
	} else {
		return cmdutil.TypeMismatchErr("allowRCUpgrades", "bool")
	}

	force, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, "force", upgradeCfg.Apply.ForceUpgrade, &flags.force).(*bool)
	if ok {
		flags.force = *force
	} else {
		return cmdutil.TypeMismatchErr("force", "bool")
	}

	allowExperimentalUpgrades, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.AllowExperimentalUpgrades, upgradeCfg.Apply.AllowExperimentalUpgrades, &flags.allowExperimentalUpgrades).(*bool)
	if ok {
		flags.allowExperimentalUpgrades = *allowExperimentalUpgrades
	} else {
		return cmdutil.TypeMismatchErr("allowExperimentalUpgrades", "bool")
	}

	if err := EnforceVersionPolicies(initCfg.KubernetesVersion, newK8sVersion, flags, versionGetter); err != nil {
		return errors.Wrap(err, "[upgrade/version] FATAL")
	}

	// If the current session is interactive, ask the user whether they really want to upgrade.
	dryRun, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.DryRun, upgradeCfg.Apply.DryRun, &flags.dryRun).(*bool)
	if ok {
		flags.dryRun = *dryRun
	} else {
		return cmdutil.TypeMismatchErr("dryRun", "bool")
	}

	if flags.sessionIsInteractive() {
		if err := cmdutil.InteractivelyConfirmAction("upgrade", "Are you sure you want to proceed?", os.Stdin); err != nil {
			return err
		}
	}

	if !flags.dryRun {
		fmt.Println("[upgrade/prepull] Pulling images required for setting up a Kubernetes cluster")
		fmt.Println("[upgrade/prepull] This might take a minute or two, depending on the speed of your internet connection")
		fmt.Println("[upgrade/prepull] You can also perform this action beforehand using 'kubeadm config images pull'")
		if err := preflight.RunPullImagesCheck(utilsexec.New(), initCfg, sets.New(upgradeCfg.Apply.IgnorePreflightErrors...)); err != nil {
			return err
		}
	} else {
		fmt.Println("[upgrade/prepull] Would pull the required images (like 'kubeadm config images pull')")
	}

	waiter := getWaiter(flags.dryRun, client, upgradeCfg.Timeouts.UpgradeManifests.Duration)

	// If the config is set by flag, just overwrite it!
	etcdUpgrade, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.EtcdUpgrade, upgradeCfg.Apply.EtcdUpgrade, &flags.etcdUpgrade).(*bool)
	if ok {
		upgradeCfg.Apply.EtcdUpgrade = etcdUpgrade
	} else {
		return cmdutil.TypeMismatchErr("etcdUpgrade", "bool")
	}

	renewCerts, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.CertificateRenewal, upgradeCfg.Apply.CertificateRenewal, &flags.renewCerts).(*bool)
	if ok {
		upgradeCfg.Apply.CertificateRenewal = renewCerts
	} else {
		return cmdutil.TypeMismatchErr("renewCerts", "bool")
	}

	if len(flags.patchesDir) > 0 {
		upgradeCfg.Apply.Patches = &kubeadmapi.Patches{Directory: flags.patchesDir}
	} else if upgradeCfg.Apply.Patches == nil {
		upgradeCfg.Apply.Patches = &kubeadmapi.Patches{}
	}

	// Now; perform the upgrade procedure
	if err := PerformControlPlaneUpgrade(flags, client, waiter, initCfg, upgradeCfg); err != nil {
		return errors.Wrap(err, "[upgrade/apply] FATAL")
	}

	// Upgrade RBAC rules and addons.
	klog.V(1).Infoln("[upgrade/postupgrade] upgrading RBAC rules and addons")
	if err := upgrade.PerformPostUpgradeTasks(client, initCfg, upgradeCfg.Apply.Patches.Directory, flags.dryRun, flags.applyPlanFlags.out); err != nil {
		return errors.Wrap(err, "[upgrade/postupgrade] FATAL post-upgrade error")
	}

	if flags.dryRun {
		fmt.Println("[upgrade/successful] Finished dryrunning successfully!")
		return nil
	}

	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!\n", initCfg.KubernetesVersion)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets if you haven't already done so.")

	return nil
}

// EnforceVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func EnforceVersionPolicies(newK8sVersionStr string, newK8sVersion *version.Version, flags *applyFlags, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to change the cluster version to %q\n", newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, newK8sVersionStr, newK8sVersion, flags.allowExperimentalUpgrades, flags.allowRCUpgrades)
	if versionSkewErrs != nil {

		if len(versionSkewErrs.Mandatory) > 0 {
			return errors.Errorf("the --version argument is invalid due to these fatal errors:\n\n%v\nPlease fix the misalignments highlighted above and try upgrading again",
				kubeadmutil.FormatErrMsg(versionSkewErrs.Mandatory))
		}

		if len(versionSkewErrs.Skippable) > 0 {
			// Return the error if the user hasn't specified the --force flag
			if !flags.force {
				return errors.Errorf("the --version argument is invalid due to these errors:\n\n%v\nCan be bypassed if you pass the --force flag",
					kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
			}
			// Soft errors found, but --force was specified
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: \n\n%v", len(versionSkewErrs.Skippable), kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
		}
	}
	return nil
}

// PerformControlPlaneUpgrade actually performs the upgrade procedure for the cluster of your type (self-hosted or static-pod-hosted)
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, initCfg *kubeadmapi.InitConfiguration, upgradeCfg *kubeadmapi.UpgradeConfiguration) error {
	// OK, the cluster is hosted using static pods. Upgrade a static-pod hosted cluster
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q (timeout: %v)...\n",
		initCfg.KubernetesVersion, upgradeCfg.Timeouts.UpgradeManifests.Duration)

	if flags.dryRun {
		return upgrade.DryRunStaticPodUpgrade(upgradeCfg.Apply.Patches.Directory, initCfg)
	}

	return upgrade.PerformStaticPodUpgrade(client, waiter, initCfg, *upgradeCfg.Apply.EtcdUpgrade, *upgradeCfg.Apply.CertificateRenewal, upgradeCfg.Apply.Patches.Directory)
}
