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

package phases

import (
	"fmt"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

// NewPreflightPhase creates a kubeadm workflow phase implements preflight checks for reset
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "preflight",
		Aliases: []string{"pre-flight"},
		Short:   "Run reset pre-flight checks",
		Long:    "Run pre-flight checks for kubeadm reset.",
		Run:     runPreflight,
		InheritFlags: []string{
			options.IgnorePreflightErrors,
			options.ForceReset,
		},
	}
}

// runPreflight executes preflight checks logic.
func runPreflight(c workflow.RunData) error {
	r, ok := c.(resetData)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}

	if !r.ForceReset() && !r.DryRun() {
		klog.Warning("[reset] WARNING: Changes made to this host by 'kubeadm init' or 'kubeadm join' will be reverted.")
		if err := util.InteractivelyConfirmAction("reset", "Are you sure you want to proceed?", r.InputReader()); err != nil {
			return err
		}
	}

	fmt.Println("[preflight] Running pre-flight checks")
	return preflight.RunRootCheckOnly(r.IgnorePreflightErrors())
}
