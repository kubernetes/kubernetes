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
	"fmt"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// MarkMaster taints the master and sets the master label
func MarkMaster(client clientset.Interface, masterName string, taints []v1.Taint) error {

	fmt.Printf("[markmaster] Marking the node %s as master by adding the label \"%s=''\"\n", masterName, constants.LabelNodeRoleMaster)

	if taints != nil && len(taints) > 0 {
		taintStrs := []string{}
		for _, taint := range taints {
			taintStrs = append(taintStrs, taint.ToString())
		}
		fmt.Printf("[markmaster] Marking the node %s as master by adding the taints %v\n", masterName, taintStrs)
	}

	return apiclient.PatchNode(client, masterName, func(n *v1.Node) {
		markMasterNode(n, taints)
	})
}

func taintExists(taint v1.Taint, taints []v1.Taint) bool {
	for _, t := range taints {
		if t == taint {
			return true
		}
	}

	return false
}

func markMasterNode(n *v1.Node, taints []v1.Taint) {
	n.ObjectMeta.Labels[constants.LabelNodeRoleMaster] = ""

	for _, nt := range n.Spec.Taints {
		if !taintExists(nt, taints) {
			taints = append(taints, nt)
		}
	}

	n.Spec.Taints = taints
}
