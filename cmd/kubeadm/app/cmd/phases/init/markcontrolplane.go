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

package phases

import (
	"github.com/pkg/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	markcontrolplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markcontrolplane"
)

var (
	markControlPlaneExample = cmdutil.Examples(`
		# Applies control-plane label and taint to the current node, functionally equivalent to what executed by kubeadm init.
		kubeadm init phase mark-control-plane --config config.yml

		# Applies control-plane label and taint to a specific node
		kubeadm init phase mark-control-plane --node-name myNode
		`)
)

// NewMarkControlPlanePhase creates a kubeadm workflow phase that implements mark-controlplane checks.
func NewMarkControlPlanePhase() workflow.Phase {
	return workflow.Phase{
		Name:    "mark-control-plane",
		Short:   "Mark a node as a control-plane",
		Example: markControlPlaneExample,
		InheritFlags: []string{
			options.NodeName,
			options.CfgPath,
		},
		Run: runMarkControlPlane,
	}
}

// runMarkControlPlane executes mark-control-plane checks logic.
func runMarkControlPlane(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("mark-control-plane phase invoked with an invalid data struct")
	}

	client, err := data.Client()
	if err != nil {
		return err
	}

	nodeRegistration := data.Cfg().NodeRegistration
	return markcontrolplanephase.MarkControlPlane(client, nodeRegistration.Name, nodeRegistration.Taints)
}
