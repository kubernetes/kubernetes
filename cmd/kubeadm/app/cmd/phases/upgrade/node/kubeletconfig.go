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
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
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

		// Handle a dupliate prefix URL scheme in the Node CRI socket.
		// Older versions of kubeadm(v1.24.0~2) upgrade may add one or two extra prefix `unix://`
		// TODO: this fix can be removed in 1.26 once all user node sockets have a correct URL scheme:
		// https://github.com/kubernetes/kubeadm/issues/2426
		var dupURLScheme bool
		nro := &kubeadmapi.NodeRegistrationOptions{}
		if !dryRun {
			if err := configutil.GetNodeRegistration(data.KubeConfigPath(), data.Client(), nro); err != nil {
				return errors.Wrap(err, "could not retrieve the node registration options for this node")
			}
			dupURLScheme = strings.HasPrefix(nro.CRISocket, kubeadmapiv1.DefaultContainerRuntimeURLScheme+"://"+kubeadmapiv1.DefaultContainerRuntimeURLScheme+"://")
		}
		if dupURLScheme {
			if !dryRun {
				newSocket := strings.ReplaceAll(nro.CRISocket, kubeadmapiv1.DefaultContainerRuntimeURLScheme+"://", "")
				newSocket = kubeadmapiv1.DefaultContainerRuntimeURLScheme + "://" + newSocket
				klog.V(2).Infof("ensuring that Node %q has a CRI socket annotation with correct URL scheme %q", nro.Name, newSocket)
				if err := patchnodephase.AnnotateCRISocket(data.Client(), nro.Name, newSocket); err != nil {
					return errors.Wrapf(err, "error updating the CRI socket for Node %q", nro.Name)
				}
			} else {
				fmt.Println("[upgrade] Would update the node CRI socket path to remove dup URL scheme prefix")
			}
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
