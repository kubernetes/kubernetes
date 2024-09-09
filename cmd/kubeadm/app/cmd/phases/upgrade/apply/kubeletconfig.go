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

// Package apply implements phases of 'kubeadm upgrade apply'.
package apply

import (
	"fmt"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
)

var (
	kubeletConfigLongDesc = cmdutil.LongDesc(`
		Upgrade the kubelet configuration for this node by downloading it from the kubelet-config ConfigMap stored in the cluster
		`)
)

// NewKubeletConfigPhase returns a new kubelet-config phase.
func NewKubeletConfigPhase() workflow.Phase {
	phase := workflow.Phase{
		Name:  "kubelet-config",
		Short: "Upgrade the kubelet configuration for this node",
		Long:  kubeletConfigLongDesc,
		Run:   runKubeletConfigPhase,
		InheritFlags: []string{
			options.CfgPath,
			options.DryRun,
			options.KubeconfigPath,
			options.Patches,
		},
	}
	return phase
}

func runKubeletConfigPhase(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("kubelet-config phase invoked with an invalid data struct")
	}

	// Write the configuration for the kubelet down to disk and print the generated manifests instead of dry-running.
	// If not dry-running, the kubelet config file will be backed up to the /etc/kubernetes/tmp/ dir, so that it could be
	// recovered if anything goes wrong.
	err := upgrade.WriteKubeletConfigFiles(data.InitCfg(), data.PatchesDir(), data.DryRun(), data.OutputWriter())
	if err != nil {
		return err
	}

	fmt.Println("[upgrade/kubelet-config] The kubelet configuration for this node was successfully upgraded!")
	return nil
}
