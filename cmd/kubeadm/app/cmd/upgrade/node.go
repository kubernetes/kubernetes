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

package upgrade

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	upgradeNodeConfigLongDesc = normalizer.LongDesc(`
		Downloads the kubelet configuration from a ConfigMap of the form "kubelet-config-1.X" in the cluster,
		where X is the minor version of the kubelet. kubeadm uses the --kubelet-version parameter to determine
		what the _desired_ kubelet version is. Give
		`)

	upgradeNodeConfigExample = normalizer.Examples(fmt.Sprintf(`
		# Downloads the kubelet configuration from the ConfigMap in the cluster. Uses a specific desired kubelet version.
		kubeadm upgrade node config --kubelet-version %s

		# Simulates the downloading of the kubelet configuration from the ConfigMap in the cluster with a specific desired
		# version. Does not change any state locally on the node.
		kubeadm upgrade node config --kubelet-version %[1]s --dry-run
		`, constants.CurrentKubernetesVersion))
)

type nodeUpgradeFlags struct {
	kubeConfigPath    string
	kubeletVersionStr string
	dryRun            bool
}

type controlplaneUpgradeFlags struct {
	kubeConfigPath   string
	advertiseAddress string
	nodeName         string
	etcdUpgrade      bool
	dryRun           bool
}

// NewCmdNode returns the cobra command for `kubeadm upgrade node`
func NewCmdNode() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "node",
		Short: "Upgrade commands for a node in the cluster. Currently only supports upgrading the configuration, not the kubelet itself.",
		RunE:  cmdutil.SubCmdRunE("node"),
	}
	cmd.AddCommand(NewCmdUpgradeNodeConfig())
	cmd.AddCommand(NewCmdUpgradeControlPlane())
	return cmd
}

// NewCmdUpgradeNodeConfig returns the cobra.Command for downloading the new/upgrading the kubelet configuration from the kubelet-config-1.X
// ConfigMap in the cluster
func NewCmdUpgradeNodeConfig() *cobra.Command {
	flags := &nodeUpgradeFlags{
		kubeConfigPath:    constants.GetKubeletKubeConfigPath(),
		kubeletVersionStr: "",
		dryRun:            false,
	}

	cmd := &cobra.Command{
		Use:     "config",
		Short:   "Downloads the kubelet configuration from the cluster ConfigMap kubelet-config-1.X, where X is the minor version of the kubelet.",
		Long:    upgradeNodeConfigLongDesc,
		Example: upgradeNodeConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunUpgradeNodeConfig(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeConfigPath)
	cmd.Flags().BoolVar(&flags.dryRun, "dry-run", flags.dryRun, "Do not change any state, just output the actions that would be performed.")
	cmd.Flags().StringVar(&flags.kubeletVersionStr, "kubelet-version", flags.kubeletVersionStr, "The *desired* version for the kubelet after the upgrade.")
	return cmd
}

// NewCmdUpgradeControlPlane returns the cobra.Command for upgrading the controlplane instance on this node
func NewCmdUpgradeControlPlane() *cobra.Command {

	flags := &controlplaneUpgradeFlags{
		kubeConfigPath:   constants.GetKubeletKubeConfigPath(),
		advertiseAddress: "",
		etcdUpgrade:      true,
		dryRun:           false,
	}

	cmd := &cobra.Command{
		Use:     "experimental-control-plane",
		Short:   "Upgrades the control plane instance deployed on this node. IMPORTANT. This command should be executed after executing `kubeadm upgrade apply` on another control plane instance",
		Long:    upgradeNodeConfigLongDesc,
		Example: upgradeNodeConfigExample,
		Run: func(cmd *cobra.Command, args []string) {

			if flags.nodeName == "" {
				klog.V(1).Infoln("[upgrade] found NodeName empty; considered OS hostname as NodeName")
			}
			nodeName, err := node.GetHostname(flags.nodeName)
			if err != nil {
				kubeadmutil.CheckErr(err)
			}
			flags.nodeName = nodeName

			if flags.advertiseAddress == "" {
				ip, err := configutil.ChooseAPIServerBindAddress(nil)
				if err != nil {
					kubeadmutil.CheckErr(err)
					return
				}

				flags.advertiseAddress = ip.String()
			}

			err = RunUpgradeControlPlane(flags)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeConfigPath)
	cmd.Flags().BoolVar(&flags.dryRun, "dry-run", flags.dryRun, "Do not change any state, just output the actions that would be performed.")
	cmd.Flags().BoolVar(&flags.etcdUpgrade, "etcd-upgrade", flags.etcdUpgrade, "Perform the upgrade of etcd.")
	return cmd
}

// RunUpgradeNodeConfig is executed when `kubeadm upgrade node config` runs.
func RunUpgradeNodeConfig(flags *nodeUpgradeFlags) error {
	if len(flags.kubeletVersionStr) == 0 {
		return errors.New("the --kubelet-version argument is required")
	}

	// Set up the kubelet directory to use. If dry-running, use a fake directory
	kubeletDir, err := getKubeletDir(flags.dryRun)
	if err != nil {
		return err
	}

	client, err := getClient(flags.kubeConfigPath, flags.dryRun)
	if err != nil {
		return errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	// Parse the desired kubelet version
	kubeletVersion, err := version.ParseSemantic(flags.kubeletVersionStr)
	if err != nil {
		return err
	}
	// TODO: Checkpoint the current configuration first so that if something goes wrong it can be recovered
	if err := kubeletphase.DownloadConfig(client, kubeletVersion, kubeletDir); err != nil {
		return err
	}

	// If we're dry-running, print the generated manifests, otherwise do nothing
	if err := printFilesIfDryRunning(flags.dryRun, kubeletDir); err != nil {
		return errors.Wrap(err, "error printing files on dryrun")
	}

	fmt.Println("[upgrade] The configuration for this node was successfully updated!")
	fmt.Println("[upgrade] Now you should go ahead and upgrade the kubelet package using your package manager.")
	return nil
}

// getKubeletDir gets the kubelet directory based on whether the user is dry-running this command or not.
func getKubeletDir(dryRun bool) (string, error) {
	if dryRun {
		dryRunDir, err := ioutil.TempDir("", "kubeadm-init-dryrun")
		if err != nil {
			return "", errors.Wrap(err, "couldn't create a temporary directory")
		}
		return dryRunDir, nil
	}
	return constants.KubeletRunDirectory, nil
}

// printFilesIfDryRunning prints the Static Pod manifests to stdout and informs about the temporary directory to go and lookup
func printFilesIfDryRunning(dryRun bool, kubeletDir string) error {
	if !dryRun {
		return nil
	}

	// Print the contents of the upgraded file and pretend like they were in kubeadmconstants.KubeletRunDirectory
	fileToPrint := dryrunutil.FileToPrint{
		RealPath:  filepath.Join(kubeletDir, constants.KubeletConfigurationFileName),
		PrintPath: filepath.Join(constants.KubeletRunDirectory, constants.KubeletConfigurationFileName),
	}
	return dryrunutil.PrintDryRunFiles([]dryrunutil.FileToPrint{fileToPrint}, os.Stdout)
}

// RunUpgradeControlPlane is executed when `kubeadm upgrade node controlplane` runs.
func RunUpgradeControlPlane(flags *controlplaneUpgradeFlags) error {

	client, err := getClient(flags.kubeConfigPath, flags.dryRun)
	if err != nil {
		return errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	waiter := apiclient.NewKubeWaiter(client, upgrade.UpgradeManifestTimeout, os.Stdout)

	// Fetches the cluster configuration
	cfg, err := configutil.FetchInitConfigurationFromCluster(client, os.Stdout, "upgrade", false)
	if err != nil {
		return errors.Wrap(err, "unable to fetch the kubeadm-config ConfigMap")
	}

	// Rotate API server certificate if needed
	if err := upgrade.BackupAPIServerCertIfNeeded(cfg, flags.dryRun); err != nil {
		return errors.Wrap(err, "unable to rotate API server certificate")
	}

	// Upgrade the control plane and etcd if installed on this node
	fmt.Printf("[upgrade] Upgrading your Static Pod-hosted control plane instance to version %q...\n", cfg.KubernetesVersion)
	if flags.dryRun {
		return DryRunStaticPodUpgrade(cfg)
	}

	if err := PerformStaticPodUpgrade(client, waiter, cfg, flags.etcdUpgrade); err != nil {
		return errors.Wrap(err, "couldn't complete the static pod upgrade")
	}

	fmt.Println("[upgrade] The control plane instance for this node was successfully updated!")
	return nil
}
