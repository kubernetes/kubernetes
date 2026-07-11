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

// Package upgrade holds the common phases for 'kubeadm upgrade'.
package upgrade

import (
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
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
	data, ok := c.(Data)
	if !ok {
		return errors.New("post-upgrade phase invoked with an invalid data struct")
	}
	// PLACEHOLDER: this phase should contain any release specific post-upgrade tasks.

	// Rewrite the kubelet env file without unwanted flags to disk and print the remaining flags instead of dry-running.
	// If not dry-running, the kubelet env file will be backed up to the /etc/kubernetes/tmp/ dir, so that it could be
	// recovered if anything goes wrong.
	unwantedFlags := []string{}
	err := upgrade.RemoveKubeletArgsFromFile(data.KubeletDir(), data.KubeConfigDir(), unwantedFlags, data.DryRun(), data.OutputWriter())
	if err != nil {
		return err
	}

	return nil
}
