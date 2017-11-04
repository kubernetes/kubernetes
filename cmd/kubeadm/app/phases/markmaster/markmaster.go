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

package markmaster

import (
	"encoding/json"
	"fmt"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

// MarkMaster taints the master and sets the master label
func MarkMaster(client clientset.Interface, masterName string) error {

	fmt.Printf("[markmaster] Will mark node %s as master by adding a label and a taint\n", masterName)

	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	return wait.Poll(kubeadmconstants.APICallRetryInterval, kubeadmconstants.MarkMasterTimeout, func() (bool, error) {
		// First get the node object
		n, err := client.CoreV1().Nodes().Get(masterName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		// The node may appear to have no labels at first,
		// so we wait for it to get hostname label.
		if _, found := n.ObjectMeta.Labels[kubeletapis.LabelHostname]; !found {
			return false, nil
		}

		oldData, err := json.Marshal(n)
		if err != nil {
			return false, err
		}

		// The master node should be tainted and labelled accordingly
		markMasterNode(n)

		newData, err := json.Marshal(n)
		if err != nil {
			return false, err
		}

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		if err != nil {
			return false, err
		}

		if _, err := client.CoreV1().Nodes().Patch(n.Name, types.StrategicMergePatchType, patchBytes); err != nil {
			if apierrs.IsConflict(err) {
				fmt.Println("[markmaster] Temporarily unable to update master node metadata due to conflict (will retry)")
				return false, nil
			}
			return false, err
		}

		fmt.Printf("[markmaster] Master %s tainted and labelled with key/value: %s=%q\n", masterName, kubeadmconstants.LabelNodeRoleMaster, "")

		return true, nil
	})
}

func markMasterNode(n *v1.Node) {
	n.ObjectMeta.Labels[kubeadmconstants.LabelNodeRoleMaster] = ""
	addTaintIfNotExists(n, kubeadmconstants.MasterTaint)
}

func addTaintIfNotExists(n *v1.Node, t v1.Taint) {
	for _, taint := range n.Spec.Taints {
		if taint == t {
			return
		}
	}

	n.Spec.Taints = append(n.Spec.Taints, t)
}
