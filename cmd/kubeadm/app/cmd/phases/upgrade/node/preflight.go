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

package node

import (
	"fmt"

	utilsexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// NewPreflightPhase returns a new preflight phase.
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "preflight",
		Short: "Run upgrade node pre-flight checks",
		Long:  "Run pre-flight checks for kubeadm upgrade node.",
		Run:   runPreflight,
		InheritFlags: []string{
			options.CfgPath,
			options.IgnorePreflightErrors,
		},
	}
}

// runPreflight executes preflight checks logic.
func runPreflight(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}
	fmt.Println("[upgrade/preflight] Running pre-flight checks")

	// First, check if we're root separately from the other preflight checks and fail fast.
	if err := preflight.RunRootCheckOnly(data.IgnorePreflightErrors()); err != nil {
		return err
	}
	if err := preflight.RunUpgradeChecks(data.IgnorePreflightErrors()); err != nil {
		return err
	}

	// If this is a control-plane node, pull the basic images.
	if data.IsControlPlaneNode() {
		// Update the InitConfiguration used for RunPullImagesCheck with ImagePullPolicy and ImagePullSerial
		// that come from UpgradeNodeConfiguration.
		initConfig := data.InitCfg()
		initConfig.NodeRegistration.ImagePullPolicy = data.Cfg().Node.ImagePullPolicy
		initConfig.NodeRegistration.ImagePullSerial = data.Cfg().Node.ImagePullSerial

		if !data.DryRun() {
			fmt.Println("[upgrade/preflight] Pulling images required for setting up a Kubernetes cluster")
			fmt.Println("[upgrade/preflight] This might take a minute or two, depending on the speed of your internet connection")
			fmt.Println("[upgrade/preflight] You can also perform this action beforehand using 'kubeadm config images pull'")
			if err := preflight.RunPullImagesCheck(utilsexec.New(), initConfig, data.IgnorePreflightErrors()); err != nil {
				return err
			}
		} else {
			fmt.Println("[upgrade/preflight] Would pull the required images (like 'kubeadm config images pull')")
		}
	} else {
		fmt.Println("[upgrade/preflight] Skipping prepull. Not a control plane node.")
		return nil
	}

	return nil
}
