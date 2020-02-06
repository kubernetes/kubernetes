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
	"errors"
	"os"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
)

// NewUpdateClusterStatus creates a kubeadm workflow phase for update-cluster-status
func NewUpdateClusterStatus() workflow.Phase {
	return workflow.Phase{
		Name:  "update-cluster-status",
		Short: "Remove this node from the ClusterStatus object.",
		Long:  "Remove this node from the ClusterStatus object if the node is a control plane node.",
		Run:   runUpdateClusterStatus,
	}
}

func runUpdateClusterStatus(c workflow.RunData) error {
	r, ok := c.(resetData)
	if !ok {
		return errors.New("update-cluster-status phase invoked with an invalid data struct")
	}

	// Reset the ClusterStatus for a given control-plane node.
	cfg := r.Cfg()
	if isControlPlane() && cfg != nil {
		if err := uploadconfig.ResetClusterStatusForNode(cfg.NodeRegistration.Name, r.Client()); err != nil {
			return err
		}
	}

	return nil
}

// isControlPlane checks if a node is a control-plane node by looking up
// the kube-apiserver manifest file
func isControlPlane() bool {
	filepath := kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.KubeAPIServer, kubeadmconstants.GetStaticPodDirectory())
	_, err := os.Stat(filepath)
	return !os.IsNotExist(err)
}
