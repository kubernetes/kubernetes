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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade/apply"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// applyFlags holds the information about the flags that can be passed to apply
type applyFlags struct {
	*applyPlanFlags

	nonInteractiveMode    bool
	force                 bool
	dryRun                bool
	etcdUpgrade           bool
	renewCerts            bool
	patchesDir            string
	ignorePreflightErrors []string
}

type applyData struct {
	etcdUpgrade           bool
	renewCerts            bool
	dryRun                bool
	force                 bool
	nonInteractiveMode    bool
	versionGetter         upgrade.VersionGetter
	cfg                   *kubeadmapi.InitConfiguration
	client                clientset.Interface
	patchesDir            string
	ignorePreflightErrors sets.String
	kubeConfigPath        string
}

// newCmdApply returns the cobra command for `kubeadm upgrade apply`
func newCmdApply(apf *applyPlanFlags) *cobra.Command {
	flags := newApplyFlags(apf)
	applyRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:                   "apply [version]",
		DisableFlagsInUseLine: true,
		Short:                 "Upgrade your Kubernetes cluster to the specified version",
		RunE: func(cmd *cobra.Command, args []string) error {
			// c, err := applyRunner.InitData(args)
			// if err != nil {
			// 	return err
			// }
			// data := c.(*applyData)

			if err := applyRunner.Run(args); err != nil {
				return err
			}
			return nil
		},
	}
	// Register the common flags for apply and plan
	addApplyPlanFlags(cmd.Flags(), flags.applyPlanFlags)

	// initialize the workflow runner with the list of phases
	applyRunner.AppendPhase(apply.NewPreflightPhase())
	applyRunner.AppendPhase(apply.NewControlPlane())
	applyRunner.AppendPhase(apply.NewKubeletConfigPhase())

	// Specify the valid flags specific for apply
	cmd.Flags().BoolVarP(&flags.nonInteractiveMode, "yes", "y", flags.nonInteractiveMode, "Perform the upgrade and do not prompt for confirmation (non-interactive mode).")
	cmd.Flags().BoolVarP(&flags.force, "force", "f", flags.force, "Force upgrading although some requirements might not be met. This also implies non-interactive mode.")
	cmd.Flags().BoolVar(&flags.dryRun, options.DryRun, flags.dryRun, "Do not change any state, just output what actions would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, "etcd-upgrade", flags.etcdUpgrade, "Perform the upgrade of etcd.")
	cmd.Flags().BoolVar(&flags.renewCerts, options.CertificateRenewal, flags.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	cmd.Flags().StringSliceVar(&flags.ignorePreflightErrors, options.IgnorePreflightErrors, flags.ignorePreflightErrors, "A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.")
	options.AddPatchesFlag(cmd.Flags(), &flags.patchesDir)

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	applyRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newApplyData(cmd, args, flags)
	})

	// binds the Runner to kubeadm upgrade apply command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	applyRunner.BindToCommand(cmd)

	return cmd
}

// newNodeOptions returns a struct ready for being used for creating cmd kubeadm upgrade node flags.
func newApplyFlags(apf *applyPlanFlags) *applyFlags {
	return &applyFlags{
		applyPlanFlags: apf,
		etcdUpgrade:    true,
		renewCerts:     true,
	}
}

// runApply takes care of the actual upgrade functionality
// It does the following things:
// 0. pre init data
// - Checks if the cluster is healthy
// - Gets the configuration from the kubeadm-config ConfigMap in the cluster
// 1. preflight
// - Enforces all version skew policies
//
// 2. upgrade cluster
// - Asks the user if they really want to upgrade
//
// 3. upgrade control plane node
// - Makes sure the control plane images are available locally on the control-plane(s)
// - Upgrades the control plane components
//
// 4. upgrade other parts
// - Applies the other resources that'd be created with kubeadm init as well, like
//   - Creating the RBAC rules for the bootstrap tokens and the cluster-info ConfigMap
//   - Applying new CorDNS and kube-proxy manifests
//   - Uploads the newly used configuration to the cluster ConfigMap
func runApply(flags *applyFlags, args []string) error {

	waiter := getWaiter(flags.dryRun, client, upgrade.UpgradeManifestTimeout)

	// Now; perform the upgrade procedure
	if err := PerformControlPlaneUpgrade(flags, client, waiter, cfg); err != nil {
		return errors.Wrap(err, "[upgrade/apply] FATAL")
	}

	// TODO: https://github.com/kubernetes/kubeadm/issues/2200
	fmt.Printf("[upgrade/postupgrade] Removing the deprecated label %s='' from all control plane Nodes. "+
		"After this step only the label %s='' will be present on control plane Nodes.\n",
		kubeadmconstants.LabelNodeRoleOldControlPlane, kubeadmconstants.LabelNodeRoleControlPlane)
	if err := upgrade.RemoveOldControlPlaneLabel(client); err != nil {
		return err
	}

	// TODO: https://github.com/kubernetes/kubeadm/issues/2200
	fmt.Printf("[upgrade/postupgrade] Adding the new taint %s to all control plane Nodes. "+
		"After this step both taints %s and %s should be present on control plane Nodes.\n",
		kubeadmconstants.ControlPlaneTaint.String(), kubeadmconstants.ControlPlaneTaint.String(),
		kubeadmconstants.OldControlPlaneTaint.String())
	if err := upgrade.AddNewControlPlaneTaint(client); err != nil {
		return err
	}

	// Upgrade RBAC rules and addons.
	klog.V(1).Infoln("[upgrade/postupgrade] upgrading RBAC rules and addons")
	if err := upgrade.PerformPostUpgradeTasks(client, cfg, flags.dryRun); err != nil {
		return errors.Wrap(err, "[upgrade/postupgrade] FATAL post-upgrade error")
	}

	if flags.dryRun {
		fmt.Println("[upgrade/successful] Finished dryrunning successfully!")
		return nil
	}

	fmt.Println("")
	fmt.Printf("[upgrade/successful] SUCCESS! Your cluster was upgraded to %q. Enjoy!\n", cfg.KubernetesVersion)
	fmt.Println("")
	fmt.Println("[upgrade/kubelet] Now that your control plane is upgraded, please proceed with upgrading your kubelets if you haven't already done so.")

	return nil
}

// PerformControlPlaneUpgrade actually performs the upgrade procedure for the cluster of your type (self-hosted or static-pod-hosted)
func PerformControlPlaneUpgrade(flags *applyFlags, client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.InitConfiguration) error {

	// OK, the cluster is hosted using static pods. Upgrade a static-pod hosted cluster
	fmt.Printf("[upgrade/apply] Upgrading your Static Pod-hosted control plane to version %q (timeout: %v)...\n",
		internalcfg.KubernetesVersion, upgrade.UpgradeManifestTimeout)

	if flags.dryRun {
		return upgrade.DryRunStaticPodUpgrade(flags.patchesDir, internalcfg)
	}

	return upgrade.PerformStaticPodUpgrade(client, waiter, internalcfg, flags.etcdUpgrade, flags.renewCerts, flags.patchesDir)
}

// newApplyData returns a new applyData struct to be used for execution of the kubeadm upgrade apply workflows
func newApplyData(cmd *cobra.Command, args []string, flags *applyFlags) (*applyData, error) {
	// Start with the basics, verify that the cluster is healthy and get the configuration from the cluster (using the ConfigMap)
	klog.V(1).Infoln("[upgrade/apply] verifying health of cluster")
	klog.V(1).Infoln("[upgrade/apply] retrieving configuration from cluster")

	client, err := getClient(flags.kubeConfigPath, flags.dryRun)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	// Fetch the configuration from a file or ConfigMap and validate it
	fmt.Println("[upgrade/config] Making sure the configuration is correct:")

	var newK8sVersion string
	cfg, legacyReconfigure, err := loadConfig(flags.cfgPath, client, false)
	if err != nil {
		if apierrors.IsNotFound(err) {
			fmt.Printf("[upgrade/config] In order to upgrade, a ConfigMap called %q in the %s namespace must exist.\n", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			fmt.Println("[upgrade/config] Without this information, 'kubeadm upgrade' won't know how to configure your upgraded cluster.")
			fmt.Println("")
			fmt.Println("[upgrade/config] Next steps:")
			fmt.Printf("\t- OPTION 1: Run 'kubeadm config upload from-flags' and specify the same CLI arguments you passed to 'kubeadm init' when you created your control-plane.\n")
			fmt.Printf("\t- OPTION 2: Run 'kubeadm config upload from-file' and specify the same config file you passed to 'kubeadm init' when you created your control-plane.\n")
			fmt.Printf("\t- OPTION 3: Pass a config file to 'kubeadm upgrade' using the --config flag.\n")
			fmt.Println("")
			err = errors.Errorf("the ConfigMap %q in the %s namespace used for getting configuration information was not found", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
		}
		return nil, errors.Wrap(err, "[upgrade/config] FATAL")
	} else if legacyReconfigure {
		// Set the newK8sVersion to the value in the ClusterConfiguration. This is done, so that users who use the --config option
		// to supply a new ClusterConfiguration don't have to specify the Kubernetes version twice,
		// if they don't want to upgrade but just change a setting.
		newK8sVersion = cfg.KubernetesVersion
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors, cfg.NodeRegistration.IgnorePreflightErrors)
	if err != nil {
		return nil, err
	}

	// Also set the union of pre-flight errors to JoinConfiguration, to provide a consistent view of the runtime configuration:
	cfg.NodeRegistration.IgnorePreflightErrors = ignorePreflightErrorsSet.List()

	// The version arg is mandatory, during upgrade apply, unless it's specified in the config file
	if newK8sVersion == "" {
		if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
			return nil, err
		}
	}

	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		newK8sVersion = args[0]
		// The `upgrade apply` version always overwrites the KubernetesVersion in the returned cfg with the target
		// version. While this is not the same for `upgrade plan` where the KubernetesVersion should be the old
		// one (because the call to getComponentConfigVersionStates requires the currently installed version).
		// This also makes the KubernetesVersion value returned for `upgrade plan` consistent as that command
		// allows to not specify a target version in which case KubernetesVersion will always hold the currently
		// installed one.
		cfg.KubernetesVersion = newK8sVersion
	}

	// If features gates are passed to the command line, use it (otherwise use featureGates from configuration)
	if flags.featureGatesString != "" {
		cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, flags.featureGatesString)
		if err != nil {
			return nil, errors.Wrap(err, "[upgrade/config] FATAL")
		}
	}

	// If the user told us to print this information out; do it!
	if flags.printConfig {
		printConfiguration(&cfg.ClusterConfiguration, os.Stdout)
	}

	return &applyData{
		etcdUpgrade:           flags.etcdUpgrade,
		renewCerts:            flags.renewCerts,
		dryRun:                flags.dryRun,
		nonInteractiveMode:    flags.nonInteractiveMode,
		cfg:                   cfg,
		client:                client,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		patchesDir:            flags.patchesDir,
		kubeConfigPath:        flags.kubeConfigPath,
		versionGetter:         upgrade.NewOfflineVersionGetter(upgrade.NewKubeVersionGetter(client), newK8sVersion),
	}, nil
}

// Force returns the force flag.
func (d *applyData) Force() bool {
	return d.force
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

// Cfg returns initConfiguration.
func (d *applyData) Cfg() *kubeadmapi.InitConfiguration {
	return d.cfg
}

// // IsControlPlaneNode returns the isControlPlaneNode flag.
// func (d *applyData) IsControlPlaneNode() bool {
// 	return d.isControlPlaneNode
// }

// Client returns a Kubernetes client to be used by kubeadm.
func (d *applyData) Client() clientset.Interface {
	return d.client
}

// PatchesDir returns the folder where patches for components are stored
func (d *applyData) PatchesDir() string {
	return d.patchesDir
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (d *applyData) IgnorePreflightErrors() sets.String {
	return d.ignorePreflightErrors
}

// KubeconfigPath returns the path to the user kubeconfig file.
func (d *applyData) KubeConfigPath() string {
	return d.kubeConfigPath
}

// KubeconfigPath returns the path to the user kubeconfig file.
func (d *applyData) VersionGetter() upgrade.VersionGetter {
	return d.versionGetter
}

// sessionIsInteractive returns true if the session is of an interactive type (the default, can be opted out of with -y, -f or --dry-run)
func (d *applyData) sessionIsInteractive() bool {
	return !(d.nonInteractiveMode || d.dryRun || d.force)
}
