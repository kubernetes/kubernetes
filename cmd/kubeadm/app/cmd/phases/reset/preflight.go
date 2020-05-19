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
	"bufio"
	"errors"
	"fmt"
	"strings"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
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

	if !r.ForceReset() {
		fmt.Println("[reset] WARNING: Changes made to this host by 'kubeadm init' or 'kubeadm join' will be reverted.")
		fmt.Print("[reset] Are you sure you want to proceed? [y/N]: ")
		s := bufio.NewScanner(r.InputReader())
		s.Scan()
		if err := s.Err(); err != nil {
			return err
		}
		if strings.ToLower(s.Text()) != "y" {
			return errors.New("aborted reset operation")
		}
	}

	fmt.Println("[preflight] Running pre-flight checks")
	return preflight.RunRootCheckOnly(r.IgnorePreflightErrors())
}
