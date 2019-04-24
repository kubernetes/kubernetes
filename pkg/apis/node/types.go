/*
Copyright 2018 The Kubernetes Authors.

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

package node

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClass defines a class of container runtime supported in the cluster.
// The RuntimeClass is used to determine which container runtime is used to run
// all containers in a pod. RuntimeClasses are (currently) manually defined by a
// user or cluster provisioner, and referenced in the PodSpec. The Kubelet is
// responsible for resolving the RuntimeClassName reference before running the
// pod.  For more details, see
// https://git.k8s.io/enhancements/keps/sig-node/runtime-class.md
type RuntimeClass struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Handler specifies the underlying runtime and configuration that the CRI
	// implementation will use to handle pods of this class. The possible values
	// are specific to the node & CRI configuration.  It is assumed that all
	// handlers are available on every node, and handlers of the same name are
	// equivalent on every node.
	// For example, a handler called "runc" might specify that the runc OCI
	// runtime (using native Linux containers) will be used to run the containers
	// in a pod.
	// The Handler must conform to the DNS Label (RFC 1123) requirements, and is
	// immutable.
	Handler string

	// Topology describes the set of nodes in the cluster that support this
	// RuntimeClass. The rules are applied applied to pods running with this
	// RuntimeClass and semantically merged with other scheduling constraints on
	// the pod.
	// If topology is nil, this RuntimeClass is assumed to be supported by all
	// nodes.
	// +optional
	Topology *Topology
}

// Topology specifies the scheduling constraints for nodes supporting a
// RuntimeClass.
type Topology struct {
	// NodeSelector selects the set of nodes that support this RuntimeClass.
	// Pods using this RuntimeClass can only be scheduled to a node matched by
	// this selector. The NodeSelector is intersected (AND) with a pod's other
	// NodeAffinity or NodeSelector requirements.
	// A nil NodeSelector selects all nodes.
	// +optional
	NodeSelector *core.NodeSelector

	// Tolerations are appended (excluding duplicates) to pods running with this
	// RuntimeClass during admission, effectively unioning the set of nodes
	// tolerated by the pod and the RuntimeClass.
	// +optional
	Tolerations []core.Toleration
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClassList is a list of RuntimeClass objects.
type RuntimeClassList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of schema objects.
	Items []RuntimeClass
}
