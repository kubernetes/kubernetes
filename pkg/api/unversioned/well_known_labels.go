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

package unversioned

const (
	// If you add a new topology domain here, also consider adding it to the set of default values
	// for the scheduler's --failure-domain command-line argument.
	LabelHostname          = "kubernetes.io/hostname"
	LabelZoneFailureDomain = "failure-domain.beta.kubernetes.io/zone"
	LabelZoneRegion        = "failure-domain.beta.kubernetes.io/region"

	LabelInstanceType = "beta.kubernetes.io/instance-type"

	LabelOS   = "beta.kubernetes.io/os"
	LabelArch = "beta.kubernetes.io/arch"
)

// Role labels are applied to Nodes to mark their purpose.
const (
	// LabelRole is a label applied to a Node as a hint that it has a particular purpose (defined by the value).
	LabelRole = "kubernetes.io/role"

	// LabelKubeadmAlphaRole is a label that kubeadm applies to a Node as a hint that it has a particular purpose.
	LabelKubeadmAlphaRole = "kubeadm.alpha.kubernetes.io/role"

	// RoleMaster is the value of a LabelRole or LabelKubeadmAlphaRole label, indicating a master node.
	// A master node typically runs kubernetes system components and will not typically run user workloads.
	RoleMaster = "master"

	// RoleNode is the value of a LabelRole or LabelKubeadmAlphaRole label, indicating a "normal" node,
	// as opposed to a RoleMaster node.
	RoleNode = "node"
)
