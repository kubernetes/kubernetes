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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/pkg/util/normalizer"
	"k8s.io/kubernetes/pkg/util/version"
)

var (
	upgradeNodeConfigLongDesc = normalizer.LongDesc(`
		Downloads the kubelet configuration from a ConfigMap of the form "kubelet-config-1.X" in the cluster,
		where X is the minor version of the kubelet. kubeadm uses the --kubelet-version parameter to determine
		what the _desired_ kubelet version is. Give 
		`)

	upgradeNodeConfigExample = normalizer.Examples(`
		# Downloads the kubelet configuration from the ConfigMap in the cluster. Uses a specific desired kubelet version.
		kubeadm upgrade node config --kubelet-version v1.11.0

		# Simulates the downloading of the kubelet configuration from the ConfigMap in the cluster with a specific desired
		# version. Does not change any state locally on the node.
		kubeadm upgrade node config --kubelet-version v1.11.0 --dry-run
		`)
)

type nodeUpgradeFlags struct {
	kubeConfigPath    string
	kubeletVersionStr string
	dryRun            bool
}

// NewCmdNode returns the cobra command for `kubeadm upgrade node`
func NewCmdNode() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "node",
		Short: "Upgrade commands for a node in the cluster. Currently only supports upgrading the configuration, not the kubelet itself.",
		RunE:  cmdutil.SubCmdRunE("node"),
	}
	cmd.AddCommand(NewCmdUpgradeNodeConfig())
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

// RunUpgradeNodeConfig is executed when `kubeadm upgrade node config` runs.
func RunUpgradeNodeConfig(flags *nodeUpgradeFlags) error {
	if len(flags.kubeletVersionStr) == 0 {
		return fmt.Errorf("The --kubelet-version argument is required")
	}

	// Set up the kubelet directory to use. If dry-running, use a fake directory
	kubeletDir, err := getKubeletDir(flags.dryRun)
	if err != nil {
		return err
	}

	client, err := getClient(flags.kubeConfigPath, flags.dryRun)
	if err != nil {
		return fmt.Errorf("couldn't create a Kubernetes client from file %q: %v", flags.kubeConfigPath, err)
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
		return fmt.Errorf("error printing files on dryrun: %v", err)
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
			return "", fmt.Errorf("couldn't create a temporary directory: %v", err)
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
