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
	"time"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade/apply"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
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
	renewCerts         bool
	imagePullTimeout   time.Duration
	kustomizeDir       string
}

// sessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run)
func (f *applyFlags) sessionIsInteractive() bool {
	return !(f.nonInteractiveMode || f.dryRun || f.force)
}

// compile-time assert that the local data object satisfies the phases data interface.
var _ phases.ApplyData = &applyData{}

// nodeData defines all the runtime information used when running the kubeadm upgrade node worklow;
// this data is shared across all the phases that are included in the workflow.
type applyData struct {
	renewCerts                bool
	cfg                       *kubeadmapi.InitConfiguration
	dryRun                    bool
	ignorePreflightErrors     sets.String
	userVersion               version.Version
	kubeConfigPath            string
	configPath                string
	outputWriter              io.Writer
	client                    clientset.Interface
	waiter                    apiclient.Waiter
	kustomizeDir              string
	featureGate               string
	etcdUpgrade               bool
	imagePullTimeout          time.Duration
	allowExperimentalUpgrades bool
	allowRCUpgrades           bool
	force                     bool
	versionGetter             upgrade.VersionGetter
}

// NewCmdApply returns the cobra command for `kubeadm upgrade apply`
func NewCmdApply(apf *applyPlanFlags) *cobra.Command {
	flags := &applyFlags{
		applyPlanFlags:   apf,
		imagePullTimeout: defaultImagePullTimeout,
		etcdUpgrade:      true,
		renewCerts:       true,
	}

	cmd := &cobra.Command{
		Use:                   "apply [version]",
		DisableFlagsInUseLine: true,
		Short:                 "Upgrade your Kubernetes cluster to the specified version",
		RunE: func(cmd *cobra.Command, args []string) error {
			userVersion, err := getK8sVersionFromUserInput(flags.applyPlanFlags, args, true)
			if err != nil {
				return err
			}

			return runApply(flags, userVersion)
		},
	}

	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)
	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, options.DryRun, flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, "etcd-upgrade", flags.etcdUpgrade, "Perform the upgrade of etcd.")
	cmd.Flags().BoolVar(&flags.renewCerts, options.CertificateRenewal, flags.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	cmd.Flags().DurationVar(&flags.imagePullTimeout, "image-pull-timeout", flags.imagePullTimeout, "The maximum amount of time to wait for the control plane pods to be downloaded.")
	options.AddKustomizePodsFlag(cmd.Flags(), &flags.kustomizeDir)

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
//   - Creating the RBAC rules for the bootstrap tokens and the cluster-info ConfigMap
//   - Applying new kube-dns and kube-proxy manifests
//   - Uploads the newly used configuration to the cluster ConfigMap
func runApply(flags *applyFlags, userVersion string) error {

	applyData, err := newApplyData(flags, userVersion)
	if err != nil {
		return err
	}

	if err = phases.RunPreflightChecksPhase(applyData); err != nil {
		return err
	}

	// If the user told us to print this information out; do it!
	if flags.printConfig {
		printConfiguration(&applyData.Cfg().ClusterConfiguration, os.Stdout)
	}

	// If the current session is interactive, ask the user whether they really want to upgrade.
	if flags.sessionIsInteractive() {
		if err := InteractivelyConfirmUpgrade("Are you sure you want to proceed with the upgrade?"); err != nil {
			return err
		}
	}

	// Now; perform the upgrade procedure
	klog.V(1).Infoln("[upgrade/apply] performing upgrade")
	if err := PerformControlPlaneUpgrade(flags, applyData.Client(), applyData.Waiter(), applyData.Cfg()); err != nil {
		return errors.Wrap(err, "[upgrade/apply] FATAL")
	}

	// Perform postupgrade tasks.
	klog.V(1).Infoln("[upgrade/postupgrade] running post-upgrade tasks")
	if err := upgrade.PerformPostUpgradeTasks(applyData.Client(), applyData.Cfg(), applyData.UserVersion(), flags.dryRun); err != nil {
		return errors.Wrap(err, "[upgrade/postupgrade] FATAL post-upgrade error")
	}

	// perform addon tasks
	klog.V(1).Infoln("[upgrade/addons] upgrading addons")
	if err := addons.UpgradeAddons(applyData.Client(), applyData.Cfg(), flags.dryRun); err != nil {
		return errors.Wrap(err, "[upgrade/addons] FATAL addons error")
	}

	if flags.dryRun {
		fmt.Println("[dryrun] Finished dryrunning successfully!")
		return nil
	}

	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!\n", applyData.Cfg().KubernetesVersion)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets if you haven't already done so.")

	return nil
}

func newApplyData(flags *applyFlags, userVersion string) (*applyData, error) {

	klog.V(1).Infoln("[upgrade/apply] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/apply] retrieving configuration from cluster")
	client, cfg, versionGetter, err := upgrade.GetUpgradeVariables(flags.applyPlanFlags.kubeConfigPath, flags.applyPlanFlags.cfgPath, flags.dryRun, userVersion)
	if err != nil {
		return nil, err
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(flags.applyPlanFlags.ignorePreflightErrors, cfg.NodeRegistration.IgnorePreflightErrors)
	if err != nil {
		return nil, err
	}

	// Use normalized version string in all following code.
	newK8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return nil, errors.Errorf("unable to parse normalized version %q as a semantic version", cfg.KubernetesVersion)
	}

	// Also set the union of pre-flight errors to InitConfiguration, to provide a consistent view of the runtime configuration:
	cfg.NodeRegistration.IgnorePreflightErrors = ignorePreflightErrorsSet.List()

	waiter := getWaiter(flags.dryRun, client)

	return &applyData{
		renewCerts:                flags.renewCerts,
		kubeConfigPath:            flags.kubeConfigPath,
		configPath:                flags.cfgPath,
		kustomizeDir:              flags.kustomizeDir,
		featureGate:               flags.featureGatesString,
		etcdUpgrade:               flags.etcdUpgrade,
		imagePullTimeout:          flags.imagePullTimeout,
		allowExperimentalUpgrades: flags.allowExperimentalUpgrades,
		allowRCUpgrades:           flags.allowRCUpgrades,
		force:                     flags.force,
		waiter:                    waiter,
		dryRun:                    flags.dryRun,
		userVersion:               *newK8sVersion,
		ignorePreflightErrors:     ignorePreflightErrorsSet,
		versionGetter:             versionGetter,
		client:                    client,
		cfg:                       cfg,
	}, nil
}

// RenewCerts returns the renewCerts flag.
func (d *applyData) RenewCerts() bool {
	return d.renewCerts
}

// Cfg returns the kubadmapi.InitConfiguration
func (d *applyData) Cfg() *kubeadmapi.InitConfiguration {
	return d.cfg
}

func (d *applyData) DryRun() bool {
	return d.dryRun
}

func (d *applyData) IgnorePreflightErrors() sets.String {
	return d.ignorePreflightErrors
}

func (d *applyData) UserVersion() *version.Version {
	return &d.userVersion
}

func (d *applyData) KubeConfigPath() string {
	return d.kubeConfigPath
}

func (d *applyData) ConfigPath() string {
	return d.configPath
}

func (d *applyData) Client() clientset.Interface {
	return d.client
}

func (d *applyData) Waiter() apiclient.Waiter {
	return d.waiter
}

func (d *applyData) KustomizeDir() string {
	return d.kustomizeDir
}

func (d *applyData) FeatureGates() string {
	return d.featureGate
}

func (d *applyData) UpgradeETCD() bool {
	return d.etcdUpgrade
}

func (d *applyData) ImagePullTimeout() time.Duration {
	return d.imagePullTimeout
}

func (d *applyData) AllowExperimentalUpgrades() bool {
	return d.allowExperimentalUpgrades
}

func (d *applyData) AllowRCUpgrades() bool {
	return d.allowRCUpgrades
}

func (d *applyData) Force() bool {
	return d.force
}

func (d *applyData) VersionGetter() upgrade.VersionGetter {
	return d.versionGetter
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
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.InitConfiguration) error {

	// OK, the cluster is hosted using static pods. Upgrade a static-pod hosted cluster
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q...\n", internalcfg.KubernetesVersion)

	if flags.dryRun {
		return upgrade.DryRunStaticPodUpgrade(flags.kustomizeDir, internalcfg)
	}

	return upgrade.PerformStaticPodUpgrade(client, waiter, internalcfg, flags.etcdUpgrade, flags.renewCerts, flags.kustomizeDir)
}
