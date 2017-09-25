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
	"time"

	"github.com/spf13/cobra"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	upgradeManifestTimeout = 1 * time.Minute
)

// applyFlags holds the information about the flags that can be passed to apply
type applyFlags struct {
	nonInteractiveMode bool
	force              bool
	dryRun             bool
	newK8sVersionStr   string
	newK8sVersion      *version.Version
	imagePullTimeout   time.Duration
	parent             *cmdUpgradeFlags
}

// SessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run)
func (f *applyFlags) SessionIsInteractive() bool {
	return !f.nonInteractiveMode
}

// NewCmdApply returns the cobra command for `kubeadm upgrade apply`
func NewCmdApply(parentFlags *cmdUpgradeFlags) *cobra.Command {
	flags := &applyFlags{
		parent:           parentFlags,
		imagePullTimeout: 15 * time.Minute,
	}

	cmd := &cobra.Command{
		Use:   "apply [version]",
		Short: "Upgrade your Kubernetes cluster to the specified version",
		Run: func(cmd *cobra.Command, args []string) {
			// Ensure the user is root
			err := runPreflightChecks(flags.parent.skipPreFlight)
			kubeadmutil.CheckErr(err)

			err = cmdutil.ValidateExactArgNumber(args, []string{"version"})
			kubeadmutil.CheckErr(err)

			// It's safe to use args[0] here as the slice has been validated above
			flags.newK8sVersionStr = args[0]

			// Default the flags dynamically, based on each others' value
			err = SetImplicitFlags(flags)
			kubeadmutil.CheckErr(err)

			err = RunApply(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, "dry-run", flags.dryRun, "Do not change any state, just output what actions would be applied.")
	cmd.Flags().DurationVar(&flags.imagePullTimeout, "image-pull-timeout", flags.imagePullTimeout, "The maximum amount of time to wait for the control plane pods to be downloaded.")

	return cmd
}

// RunApply takes care of the actual upgrade functionality
// It does the following things:
// - Checks if the cluster is healthy
// - Gets the configuration from the kubeadm-config ConfigMap in the cluster
// - Enforces all version skew policies
// - Asks the user if they really want to upgrade
// - Makes sure the control plane images are available locally on the master(s)
// - Upgrades the control plane components
// - Applies the other resources that'd be created with kubeadm init as well, like
//   - Creating the RBAC rules for the Bootstrap Tokens and the cluster-info ConfigMap
//   - Applying new kube-dns and kube-proxy manifests
//   - Uploads the newly used configuration to the cluster ConfigMap
func RunApply(flags *applyFlags) error {

	// Start with the basics, verify that the cluster is healthy and get the configuration from the cluster (using the ConfigMap)
	upgradeVars, err := enforceRequirements(flags.parent.kubeConfigPath, flags.parent.cfgPath, flags.parent.printConfig, flags.dryRun)
	if err != nil {
		return err
	}

	// Set the upgraded version on the external config object now
	upgradeVars.cfg.KubernetesVersion = flags.newK8sVersionStr

	// Grab the external, versioned configuration and convert it to the internal type for usage here later
	internalcfg := &kubeadmapi.MasterConfiguration{}
	api.Scheme.Convert(upgradeVars.cfg, internalcfg, nil)

	// Validate requested and validate actual version
	if err := configutil.NormalizeKubernetesVersion(internalcfg); err != nil {
		return err
	}

	// Use normalized version string in all following code.
	flags.newK8sVersionStr = internalcfg.KubernetesVersion
	k8sVer, err := version.ParseSemantic(flags.newK8sVersionStr)
	if err != nil {
		return fmt.Errorf("unable to parse normalized version %q as a semantic version", flags.newK8sVersionStr)
	}
	flags.newK8sVersion = k8sVer

	// Enforce the version skew policies
	if err := EnforceVersionPolicies(flags, upgradeVars.versionGetter); err != nil {
		return fmt.Errorf("[upgrade/version] FATAL: %v", err)
	}

	// If the current session is interactive, ask the user whether they really want to upgrade
	if flags.SessionIsInteractive() {
		if err := InteractivelyConfirmUpgrade("Are you sure you want to proceed with the upgrade?"); err != nil {
			return err
		}
	}

	// Use a prepuller implementation based on creating DaemonSets
	// and block until all DaemonSets are ready; then we know for sure that all control plane images are cached locally
	prepuller := upgrade.NewDaemonSetPrepuller(upgradeVars.client, upgradeVars.waiter, internalcfg)
	upgrade.PrepullImagesInParallel(prepuller, flags.imagePullTimeout)

	// Now; perform the upgrade procedure
	if err := PerformControlPlaneUpgrade(flags, upgradeVars.client, upgradeVars.waiter, internalcfg); err != nil {
		return fmt.Errorf("[upgrade/apply] FATAL: %v", err)
	}

	// Upgrade RBAC rules and addons. Optionally, if needed, perform some extra task for a specific version
	if err := upgrade.PerformPostUpgradeTasks(upgradeVars.client, internalcfg, flags.newK8sVersion); err != nil {
		return fmt.Errorf("[upgrade/postupgrade] FATAL post-upgrade error: %v", err)
	}

	if flags.dryRun {
		fmt.Println("[dryrun] Finished dryrunning successfully!")
		return nil
	}

	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!\n", flags.newK8sVersionStr)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets in turn.")

	return nil
}

// SetImplicitFlags handles dynamically defaulting flags based on each other's value
func SetImplicitFlags(flags *applyFlags) error {
	// If we are in dry-run or force mode; we should automatically execute this command non-interactively
	if flags.dryRun || flags.force {
		flags.nonInteractiveMode = true
	}

	if len(flags.newK8sVersionStr) == 0 {
		return fmt.Errorf("version string can't be empty")
	}

	return nil
}

// EnforceVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func EnforceVersionPolicies(flags *applyFlags, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to upgrade to version %q\n", flags.newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, flags.newK8sVersionStr, flags.newK8sVersion, flags.parent.allowExperimentalUpgrades, flags.parent.allowRCUpgrades)
	if versionSkewErrs != nil {

		if len(versionSkewErrs.Mandatory) > 0 {
			return fmt.Errorf("The --version argument is invalid due to these fatal errors: %v", versionSkewErrs.Mandatory)
		}

		if len(versionSkewErrs.Skippable) > 0 {
			// Return the error if the user hasn't specified the --force flag
			if !flags.force {
				return fmt.Errorf("The --version argument is invalid due to these errors: %v. Can be bypassed if you pass the --force flag", versionSkewErrs.Skippable)
			}
			// Soft errors found, but --force was specified
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: %v\n", len(versionSkewErrs.Skippable), versionSkewErrs.Skippable)
		}
	}
	return nil
}

// PerformControlPlaneUpgrade actually performs the upgrade procedure for the cluster of your type (self-hosted or static-pod-hosted)
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.MasterConfiguration) error {

	// Check if the cluster is self-hosted and act accordingly
	if upgrade.IsControlPlaneSelfHosted(client) {
		fmt.Printf("[upgrade/apply] Upgrading your Self-Hosted control plane to version %q...\n", flags.newK8sVersionStr)

		// Upgrade the self-hosted cluster
		return upgrade.SelfHostedControlPlane(client, waiter, internalcfg, flags.newK8sVersion)
	}

	// OK, the cluster is hosted using static pods. Upgrade a static-pod hosted cluster
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q...\n", flags.newK8sVersionStr)

	if flags.dryRun {
		return DryRunStaticPodUpgrade(internalcfg)
	}
	return PerformStaticPodUpgrade(client, waiter, internalcfg)
}

// PerformStaticPodUpgrade performs the upgrade of the control plane components for a static pod hosted cluster
func PerformStaticPodUpgrade(client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.MasterConfiguration) error {
	pathManager, err := upgrade.NewKubeStaticPodPathManagerUsingTempDirs(constants.GetStaticPodDirectory())
	if err != nil {
		return err
	}

	if err := upgrade.StaticPodControlPlane(waiter, pathManager, internalcfg); err != nil {
		return err
	}
	return nil
}

// DryRunStaticPodUpgrade fakes an upgrade of the control plane
func DryRunStaticPodUpgrade(internalcfg *kubeadmapi.MasterConfiguration) error {

	dryRunManifestDir, err := constants.CreateTempDirForKubeadm("kubeadm-upgrade-dryrun")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dryRunManifestDir)

	if err := controlplane.CreateInitStaticPodManifestFiles(dryRunManifestDir, internalcfg); err != nil {
		return err
	}

	// Print the contents of the upgraded manifests and pretend like they were in /etc/kubernetes/manifests
	files := []dryrunutil.FileToPrint{}
	for _, component := range constants.MasterComponents {
		realPath := constants.GetStaticPodFilepath(component, dryRunManifestDir)
		outputPath := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}

	if err := dryrunutil.PrintDryRunFiles(files, os.Stdout); err != nil {
		return err
	}
	return nil
}
