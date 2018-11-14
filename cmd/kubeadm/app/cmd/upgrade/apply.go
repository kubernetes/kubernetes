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

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
)

const (
	defaultImagePullTimeout = 15 * time.Minute
)

// applyFlags holds the information about the flags that can be passed to apply
type applyFlags struct {
	*applyPlanFlags

	nonInteractiveMode bool
	force              bool
	dryRun             bool
	etcdUpgrade        bool
	criSocket          string
	newK8sVersionStr   string
	newK8sVersion      *version.Version
	imagePullTimeout   time.Duration
}

// SessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run)
func (f *applyFlags) SessionIsInteractive() bool {
	return !f.nonInteractiveMode
}

// NewCmdApply returns the cobra command for `kubeadm upgrade apply`
func NewCmdApply(apf *applyPlanFlags) *cobra.Command {
	flags := &applyFlags{
		applyPlanFlags:   apf,
		imagePullTimeout: defaultImagePullTimeout,
		etcdUpgrade:      true,
		criSocket:        kubeadmapiv1beta1.DefaultCRISocket,
	}

	cmd := &cobra.Command{
		Use:                   "apply [version]",
		DisableFlagsInUseLine: true,
		Short:                 "Upgrade your Kubernetes cluster to the specified version.",
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			flags.ignorePreflightErrorsSet, err = validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors)
			kubeadmutil.CheckErr(err)

			// Ensure the user is root
			klog.V(1).Infof("running preflight checks")
			err = runPreflightChecks(flags.ignorePreflightErrorsSet)
			kubeadmutil.CheckErr(err)

			// If the version is specified in config file, pick up that value.
			if flags.cfgPath != "" {
				klog.V(1).Infof("fetching configuration from file %s", flags.cfgPath)
				// Note that cfg isn't preserved here, it's just an one-off to populate flags.newK8sVersionStr based on --config
				cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(flags.cfgPath, &kubeadmapiv1beta1.InitConfiguration{})
				kubeadmutil.CheckErr(err)

				if cfg.KubernetesVersion != "" {
					flags.newK8sVersionStr = cfg.KubernetesVersion
				}
			}

			// If the new version is already specified in config file, version arg is optional.
			if flags.newK8sVersionStr == "" {
				err = cmdutil.ValidateExactArgNumber(args, []string{"version"})
				kubeadmutil.CheckErr(err)
			}

			// If option was specified in both args and config file, args will overwrite the config file.
			if len(args) == 1 {
				flags.newK8sVersionStr = args[0]
			}

			// Default the flags dynamically, based on each others' value
			err = SetImplicitFlags(flags)
			kubeadmutil.CheckErr(err)

			err = RunApply(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, "dry-run", flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, "etcd-upgrade", flags.etcdUpgrade, "Perform the upgrade of etcd.")
	cmd.Flags().DurationVar(&flags.imagePullTimeout, "image-pull-timeout", flags.imagePullTimeout, "The maximum amount of time to wait for the control plane pods to be downloaded.")
	// TODO: Register this flag in a generic place
	cmd.Flags().StringVar(&flags.criSocket, "cri-socket", flags.criSocket, "Specify the CRI socket to connect to.")
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
//   - Creating the RBAC rules for the bootstrap tokens and the cluster-info ConfigMap
//   - Applying new kube-dns and kube-proxy manifests
//   - Uploads the newly used configuration to the cluster ConfigMap
func RunApply(flags *applyFlags) error {

	// Start with the basics, verify that the cluster is healthy and get the configuration from the cluster (using the ConfigMap)
	klog.V(1).Infof("[upgrade/apply] verifying health of cluster")
	klog.V(1).Infof("[upgrade/apply] retrieving configuration from cluster")
	upgradeVars, err := enforceRequirements(flags.applyPlanFlags, flags.dryRun, flags.newK8sVersionStr)
	if err != nil {
		return err
	}

	if len(flags.criSocket) != 0 {
		fmt.Println("[upgrade/apply] Respecting the --cri-socket flag that is set with higher priority than the config file.")
		upgradeVars.cfg.NodeRegistration.CRISocket = flags.criSocket
	}

	// Validate requested and validate actual version
	klog.V(1).Infof("[upgrade/apply] validating requested and actual version")
	if err := configutil.NormalizeKubernetesVersion(&upgradeVars.cfg.ClusterConfiguration); err != nil {
		return err
	}

	// Use normalized version string in all following code.
	flags.newK8sVersionStr = upgradeVars.cfg.KubernetesVersion
	k8sVer, err := version.ParseSemantic(flags.newK8sVersionStr)
	if err != nil {
		return errors.Errorf("unable to parse normalized version %q as a semantic version", flags.newK8sVersionStr)
	}
	flags.newK8sVersion = k8sVer

	if err := features.ValidateVersion(features.InitFeatureGates, upgradeVars.cfg.FeatureGates, upgradeVars.cfg.KubernetesVersion); err != nil {
		return err
	}

	// Enforce the version skew policies
	klog.V(1).Infof("[upgrade/version] enforcing version skew policies")
	if err := EnforceVersionPolicies(flags, upgradeVars.versionGetter); err != nil {
		return errors.Wrap(err, "[upgrade/version] FATAL")
	}

	// If the current session is interactive, ask the user whether they really want to upgrade
	if flags.SessionIsInteractive() {
		if err := InteractivelyConfirmUpgrade("Are you sure you want to proceed with the upgrade?"); err != nil {
			return err
		}
	}

	// Use a prepuller implementation based on creating DaemonSets
	// and block until all DaemonSets are ready; then we know for sure that all control plane images are cached locally
	klog.V(1).Infof("[upgrade/apply] creating prepuller")
	prepuller := upgrade.NewDaemonSetPrepuller(upgradeVars.client, upgradeVars.waiter, &upgradeVars.cfg.ClusterConfiguration)
	componentsToPrepull := constants.MasterComponents
	if upgradeVars.cfg.Etcd.External == nil && flags.etcdUpgrade {
		componentsToPrepull = append(componentsToPrepull, constants.Etcd)
	}
	if err := upgrade.PrepullImagesInParallel(prepuller, flags.imagePullTimeout, componentsToPrepull); err != nil {
		return errors.Wrap(err, "[upgrade/prepull] Failed prepulled the images for the control plane components error")
	}

	// Now; perform the upgrade procedure
	klog.V(1).Infof("[upgrade/apply] performing upgrade")
	if err := PerformControlPlaneUpgrade(flags, upgradeVars.client, upgradeVars.waiter, upgradeVars.cfg); err != nil {
		return errors.Wrap(err, "[upgrade/apply] FATAL")
	}

	// Upgrade RBAC rules and addons.
	klog.V(1).Infof("[upgrade/postupgrade] upgrading RBAC rules and addons")
	if err := upgrade.PerformPostUpgradeTasks(upgradeVars.client, upgradeVars.cfg, flags.newK8sVersion, flags.dryRun); err != nil {
		return errors.Wrap(err, "[upgrade/postupgrade] FATAL post-upgrade error")
	}

	if flags.dryRun {
		fmt.Println("[dryrun] Finished dryrunning successfully!")
		return nil
	}

	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!\n", flags.newK8sVersionStr)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets if you haven't already done so.")

	return nil
}

// SetImplicitFlags handles dynamically defaulting flags based on each other's value
func SetImplicitFlags(flags *applyFlags) error {
	// If we are in dry-run or force mode; we should automatically execute this command non-interactively
	if flags.dryRun || flags.force {
		flags.nonInteractiveMode = true
	}

	if len(flags.newK8sVersionStr) == 0 {
		return errors.New("version string can't be empty")
	}

	return nil
}

// EnforceVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func EnforceVersionPolicies(flags *applyFlags, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to change the cluster version to %q\n", flags.newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, flags.newK8sVersionStr, flags.newK8sVersion, flags.allowExperimentalUpgrades, flags.allowRCUpgrades)
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
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.InitConfiguration) error {

	// OK, the cluster is hosted using static pods. Upgrade a static-pod hosted cluster
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q...\n", flags.newK8sVersionStr)

	if flags.dryRun {
		return DryRunStaticPodUpgrade(internalcfg)
	}

	// Don't save etcd backup directory if etcd is HA, as this could cause corruption
	return PerformStaticPodUpgrade(client, waiter, internalcfg, flags.etcdUpgrade)
}

// GetPathManagerForUpgrade returns a path manager properly configured for the given InitConfiguration.
func GetPathManagerForUpgrade(internalcfg *kubeadmapi.InitConfiguration, etcdUpgrade bool) (upgrade.StaticPodPathManager, error) {
	isHAEtcd := etcdutil.CheckConfigurationIsHA(&internalcfg.Etcd)
	return upgrade.NewKubeStaticPodPathManagerUsingTempDirs(constants.GetStaticPodDirectory(), true, etcdUpgrade && !isHAEtcd)
}

// PerformStaticPodUpgrade performs the upgrade of the control plane components for a static pod hosted cluster
func PerformStaticPodUpgrade(client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.InitConfiguration, etcdUpgrade bool) error {
	pathManager, err := GetPathManagerForUpgrade(internalcfg, etcdUpgrade)
	if err != nil {
		return err
	}

	// The arguments oldEtcdClient and newEtdClient, are uninitialized because passing in the clients allow for mocking the client during testing
	return upgrade.StaticPodControlPlane(client, waiter, pathManager, internalcfg, etcdUpgrade, nil, nil)
}

// DryRunStaticPodUpgrade fakes an upgrade of the control plane
func DryRunStaticPodUpgrade(internalcfg *kubeadmapi.InitConfiguration) error {

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

	return dryrunutil.PrintDryRunFiles(files, os.Stdout)
}
