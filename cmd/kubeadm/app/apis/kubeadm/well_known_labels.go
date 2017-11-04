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

package kubeadm

// Role labels are applied to Nodes to mark their purpose.  In particular, we
// usually want to distinguish the master, so that we can isolate privileged
// pods and operations.
//
// Originally we relied on not registering the master, on the fact that the
// master was Unschedulable, and on static manifests for master components.
// But we now do register masters in many environments, are generally moving
// away from static manifests (for better manageability), and working towards
// deprecating the unschedulable field (replacing it with taints & tolerations
// instead).
//
// Even with tainting, a label remains the easiest way of making a positive
// selection, so that pods can schedule only to master nodes for example, and
// thus installations will likely define a label for their master nodes.
//
// So that we can recognize master nodes in consequent places though (such as
// kubectl get nodes), we encourage installations to use the well-known labels.
// We define NodeLabelRole, which is the preferred form, but we will also recognize
// other forms that are known to be in widespread use (NodeLabelKubeadmAlphaRole).

const (
	// NodeLabelKubeadmAlphaRole is a label that kubeadm applies to a Node as a hint that it has a particular purpose.
	// Use of NodeLabelRole is preferred.
	NodeLabelKubeadmAlphaRole = "kubeadm.alpha.kubernetes.io/role"
)
