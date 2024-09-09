/*
Copyright 2024 The Kubernetes Authors.

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

// Package node implements phases of 'kubeadm upgrade node'.
package node

import (
	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
)

// NewPostUpgradePhase returns a new post-upgrade phase.
func NewPostUpgradePhase() workflow.Phase {
	return workflow.Phase{
		Name:  "post-upgrade",
		Short: "Run post upgrade tasks",
		Run:   runPostUpgrade,
		InheritFlags: []string{
			options.CfgPath,
			options.KubeconfigPath,
			options.DryRun,
		},
	}
}

func runPostUpgrade(c workflow.RunData) error {
	_, ok := c.(Data)
	if !ok {
		return errors.New("post-upgrade phase invoked with an invalid data struct")
	}
	// PLACEHOLDER: this phase should contain any release specific post-upgrade tasks.

	return nil
}
