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

	"github.com/pkg/errors"

	utilsexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

// NewPreflightPhase creates a kubeadm workflow phase that implements preflight checks for a new node join
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "preflight",
		Short: "Run upgrade node pre-flight checks",
		Long:  "Run pre-flight checks for kubeadm upgrade node.",
		Run:   runPreflight,
		InheritFlags: []string{
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
	fmt.Println("[preflight] Running pre-flight checks")

	// First, check if we're root separately from the other preflight checks and fail fast
	if err := preflight.RunRootCheckOnly(data.IgnorePreflightErrors()); err != nil {
		return err
	}

	// if this is a control-plane node, pull the basic images
	if data.IsControlPlaneNode() {
		if !data.DryRun() {
			fmt.Println("[preflight] Pulling images required for setting up a Kubernetes cluster")
			fmt.Println("[preflight] This might take a minute or two, depending on the speed of your internet connection")
			fmt.Println("[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'")
			if err := preflight.RunPullImagesCheck(utilsexec.New(), data.InitCfg(), data.IgnorePreflightErrors()); err != nil {
				return err
			}
		} else {
			fmt.Println("[preflight] Would pull the required images (like 'kubeadm config images pull')")
		}
	} else {
		fmt.Println("[preflight] Skipping prepull. Not a control plane node.")
		return nil
	}

	return nil
}
