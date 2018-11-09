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
	"path/filepath"

	"github.com/pkg/errors"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	markcontrolplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markcontrolplane"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	markControlPlaneExample = normalizer.Examples(`
		# Applies control-plane label and taint to the current node, functionally equivalent to what executed by kubeadm init.
		kubeadm init phase mark-control-plane --config config.yml

		# Applies control-plane label and taint to a specific node
		kubeadm init phase mark-control-plane --node-name myNode
		`)
)

type markControlPlaneData interface {
	Cfg() *kubeadmapi.InitConfiguration
	KubeConfigPath() string
	KubeConfigDir() string
	DryRun() bool
}

// NewMarkControlPlanePhase creates a kubeadm workflow phase that implements mark-controlplane checks.
func NewMarkControlPlanePhase() workflow.Phase {
	return workflow.Phase{
		Name:    "mark-control-plane",
		Short:   "Mark a node as a control-plane",
		Example: markControlPlaneExample,
		Run:     runMarkControlPlane,
	}
}

// runMarkControlPlane executes markcontrolplane checks logic.
func runMarkControlPlane(c workflow.RunData) error {
	data, ok := c.(markControlPlaneData)
	if !ok {
		return errors.New("mark-control-plane phase invoked with an invalid data struct")
	}

	kubeConfigFile := filepath.Join(data.KubeConfigDir(), kubeadmconstants.AdminKubeConfigFileName)
	kubeConfigFile = data.KubeConfigPath()
	client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return err
	}

	if err := markcontrolplanephase.MarkControlPlane(client, data.Cfg().NodeRegistration.Name, data.Cfg().NodeRegistration.Taints); err != nil {
		return err
	}

	return nil
}
