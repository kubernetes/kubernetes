/*
Copyright 2015 The Kubernetes Authors.

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

package v1

const (
	LabelHostname          = "kubernetes.io/hostname"
	LabelZoneFailureDomain = "failure-domain.beta.kubernetes.io/zone"
	LabelZoneRegion        = "failure-domain.beta.kubernetes.io/region"

	LabelInstanceType = "beta.kubernetes.io/instance-type"

	LabelOS   = "beta.kubernetes.io/os"
	LabelArch = "beta.kubernetes.io/arch"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeNotReady would be automatically added by node controller
	// when node is not ready, and removed when node becomes ready.
	TaintNodeNotReady = "node.alpha.kubernetes.io/notReady"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeUnreachable would be automatically added by node controller
	// when node becomes unreachable (corresponding to NodeReady status ConditionUnknown)
	// and removed when node becomes reachable (NodeReady status ConditionTrue).
	TaintNodeUnreachable = "node.alpha.kubernetes.io/unreachable"

	// When kubelet is started with the "external" cloud provider, then
	// it sets this taint on a node to mark it as unusable, until a controller
	// from the cloud-controller-manager intitializes this node, and then removes
	// the taint
	TaintExternalCloudProvider = "node.cloudprovider.kubernetes.io/uninitialized"
)

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
	// NodeLabelRole is the preferred label applied to a Node as a hint that it has a particular purpose (defined by the value).
	NodeLabelRole = "kubernetes.io/role"

	// NodeLabelKubeadmAlphaRole is a label that kubeadm applies to a Node as a hint that it has a particular purpose.
	// Use of NodeLabelRole is preferred.
	NodeLabelKubeadmAlphaRole = "kubeadm.alpha.kubernetes.io/role"

	// NodeLabelRoleMaster is the value of a NodeLabelRole or NodeLabelKubeadmAlphaRole label, indicating a master node.
	// A master node typically runs kubernetes system components and will not typically run user workloads.
	NodeLabelRoleMaster = "master"

	// NodeLabelRoleNode is the value of a NodeLabelRole or NodeLabelKubeadmAlphaRole label, indicating a "normal" node,
	// as opposed to a RoleMaster node.
	NodeLabelRoleNode = "node"
)
