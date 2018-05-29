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
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// MarkMaster taints the master and sets the master label
func MarkMaster(client clientset.Interface, masterName string, taints []v1.Taint) error {

	glog.Infof("[markmaster] Marking the node %s as master by adding the label \"%s=''\"\n", masterName, constants.LabelNodeRoleMaster)

	if taints != nil && len(taints) > 0 {
		glog.Infof("[markmaster] Marking the node %s as master by adding the taints %v\n", masterName, taints)
	}

	return apiclient.PatchNode(client, masterName, func(n *v1.Node) {
		markMasterNode(n, taints)
	})
}

func markMasterNode(n *v1.Node, taints []v1.Taint) {
	n.ObjectMeta.Labels[constants.LabelNodeRoleMaster] = ""
	// TODO: Append taints, don't override?
	n.Spec.Taints = taints
}
