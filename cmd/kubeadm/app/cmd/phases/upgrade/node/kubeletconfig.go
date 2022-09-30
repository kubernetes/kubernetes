/*
Copyright 2019 The Kubernetes Authors.

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

package node

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

var (
	kubeletConfigLongDesc = cmdutil.LongDesc(`
		Download the kubelet configuration from the kubelet-config ConfigMap stored in the cluster
		`)
)

// NewKubeletConfigPhase creates a kubeadm workflow phase that implements handling of kubelet-config upgrade.
func NewKubeletConfigPhase() workflow.Phase {
	phase := workflow.Phase{
		Name:  "kubelet-config",
		Short: "Upgrade the kubelet configuration for this node",
		Long:  kubeletConfigLongDesc,
		Run:   runKubeletConfigPhase(),
		InheritFlags: []string{
			options.DryRun,
			options.KubeconfigPath,
			options.Patches,
		},
	}
	return phase
}

func runKubeletConfigPhase() func(c workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(Data)
		if !ok {
			return errors.New("kubelet-config phase invoked with an invalid data struct")
		}

		// otherwise, retrieve all the info required for kubelet config upgrade
		cfg := data.Cfg()
		dryRun := data.DryRun()

		// Set up the kubelet directory to use. If dry-running, this will return a fake directory
		kubeletDir, err := upgrade.GetKubeletDir(dryRun)
		if err != nil {
			return err
		}

		// TODO: Checkpoint the current configuration first so that if something goes wrong it can be recovered

		// Store the kubelet component configuration.
		if err = kubeletphase.WriteConfigToDisk(&cfg.ClusterConfiguration, kubeletDir, data.PatchesDir(), data.OutputWriter()); err != nil {
			return err
		}

		// If we're dry-running, print the generated manifests
		if dryRun {
			if err := printFilesIfDryRunning(dryRun, kubeletDir); err != nil {
				return errors.Wrap(err, "error printing files on dryrun")
			}
			return nil
		}

		// TODO: Temporary workaround. Remove in 1.27:
		// https://github.com/kubernetes/kubeadm/issues/2626
		if err := upgrade.CleanupKubeletDynamicEnvFileContainerRuntime(dryRun); err != nil {
			return err
		}

		fmt.Println("[upgrade] The configuration for this node was successfully updated!")
		fmt.Println("[upgrade] Now you should go ahead and upgrade the kubelet package using your package manager.")
		return nil
	}
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
