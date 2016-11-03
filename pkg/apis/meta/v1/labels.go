/*
Copyright 2016 The Kubernetes Authors.

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

// TODO: these should really be part of the Kubernetes core API.
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

// TODO: remove me when watch is refactored
func LabelSelectorQueryParam(version string) string {
	return "labelSelector"
}

// TODO: remove me when watch is refactored
func FieldSelectorQueryParam(version string) string {
	return "fieldSelector"
}

// Note:
// There are two different styles of label selectors used in versioned types:
// an older style which is represented as just a string in versioned types, and a
// newer style that is structured.  LabelSelector is an internal representation for the
// latter style.

// A label selector is a label query over a set of resources. The result of matchLabels and
// matchExpressions are ANDed. An empty label selector matches all objects. A null
// label selector matches no objects.
type LabelSelector struct {
	// matchLabels is a map of {key,value} pairs. A single {key,value} in the matchLabels
	// map is equivalent to an element of matchExpressions, whose key field is "key", the
	// operator is "In", and the values array contains only "value". The requirements are ANDed.
	// +optional
	MatchLabels map[string]string `json:"matchLabels,omitempty" protobuf:"bytes,1,rep,name=matchLabels"`
	// matchExpressions is a list of label selector requirements. The requirements are ANDed.
	// +optional
	MatchExpressions []LabelSelectorRequirement `json:"matchExpressions,omitempty" protobuf:"bytes,2,rep,name=matchExpressions"`
}

// A label selector requirement is a selector that contains values, a key, and an operator that
// relates the key and values.
type LabelSelectorRequirement struct {
	// key is the label key that the selector applies to.
	Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key" protobuf:"bytes,1,opt,name=key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators ard In, NotIn, Exists and DoesNotExist.
	Operator LabelSelectorOperator `json:"operator" protobuf:"bytes,2,opt,name=operator,casttype=LabelSelectorOperator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. This array is replaced during a strategic
	// merge patch.
	// +optional
	Values []string `json:"values,omitempty" protobuf:"bytes,3,rep,name=values"`
}

// A label selector operator is the set of operators that can be used in a selector requirement.
type LabelSelectorOperator string

const (
	LabelSelectorOpIn           LabelSelectorOperator = "In"
	LabelSelectorOpNotIn        LabelSelectorOperator = "NotIn"
	LabelSelectorOpExists       LabelSelectorOperator = "Exists"
	LabelSelectorOpDoesNotExist LabelSelectorOperator = "DoesNotExist"
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
