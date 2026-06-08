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

package markcontrolplane

import (
	"fmt"
	"slices"

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// labelsToAdd holds a list of labels that are applied on kubeadm managed control plane nodes
var labelsToAdd = []string{
	constants.LabelNodeRoleControlPlane,
	constants.LabelExcludeFromExternalLB,
}

// MarkControlPlane taints the control-plane and sets the control-plane label
func MarkControlPlane(client clientset.Interface, controlPlaneName string, taints []v1.Taint) error {
	fmt.Printf("[mark-control-plane] Marking the node %s as control-plane by adding the labels: %v\n",
		controlPlaneName, labelsToAdd)

	if len(taints) > 0 {
		taintStrs := []string{}
		for _, taint := range taints {
			taintStrs = append(taintStrs, taint.ToString())
		}
		fmt.Printf("[mark-control-plane] Marking the node %s as control-plane by adding the taints %v\n", controlPlaneName, taintStrs)
	}

	return apiclient.PatchNode(client, controlPlaneName, func(n *v1.Node) {
		markControlPlaneNode(n, taints)
	})
}

func markControlPlaneNode(n *v1.Node, taints []v1.Taint) {
	for _, label := range labelsToAdd {
		n.ObjectMeta.Labels[label] = ""
	}

	for _, nt := range n.Spec.Taints {
		if !slices.Contains(taints, nt) {
			taints = append(taints, nt)
		}
	}

	n.Spec.Taints = taints
}
