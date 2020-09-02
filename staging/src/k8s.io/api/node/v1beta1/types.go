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

package v1beta1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.13
// +k8s:prerelease-lifecycle-gen:deprecated=1.22

// RuntimeClass defines a class of container runtime supported in the cluster.
// The RuntimeClass is used to determine which container runtime is used to run
// all containers in a pod. RuntimeClasses are (currently) manually defined by a
// user or cluster provisioner, and referenced in the PodSpec. The Kubelet is
// responsible for resolving the RuntimeClassName reference before running the
// pod.  For more details, see
// https://git.k8s.io/enhancements/keps/sig-node/runtime-class.md
type RuntimeClass struct {
	metav1.TypeMeta `json:",inline"`
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Handler specifies the underlying runtime and configuration that the CRI
	// implementation will use to handle pods of this class. The possible values
	// are specific to the node & CRI configuration.  It is assumed that all
	// handlers are available on every node, and handlers of the same name are
	// equivalent on every node.
	// For example, a handler called "runc" might specify that the runc OCI
	// runtime (using native Linux containers) will be used to run the containers
	// in a pod.
	// The Handler must be lowercase, conform to the DNS Label (RFC 1123) requirements,
	// and is immutable.
	Handler string `json:"handler" protobuf:"bytes,2,opt,name=handler"`

	// Overhead represents the resource overhead associated with running a pod for a
	// given RuntimeClass. For more details, see
	// https://git.k8s.io/enhancements/keps/sig-node/20190226-pod-overhead.md
	// This field is alpha-level as of Kubernetes v1.15, and is only honored by servers that enable the PodOverhead feature.
	// +optional
	Overhead *Overhead `json:"overhead,omitempty" protobuf:"bytes,3,opt,name=overhead"`

	// Scheduling holds the scheduling constraints to ensure that pods running
	// with this RuntimeClass are scheduled to nodes that support it.
	// If scheduling is nil, this RuntimeClass is assumed to be supported by all
	// nodes.
	// +optional
	Scheduling *Scheduling `json:"scheduling,omitempty" protobuf:"bytes,4,opt,name=scheduling"`
}

// Overhead structure represents the resource overhead associated with running a pod.
type Overhead struct {
	// PodFixed represents the fixed resource overhead associated with running a pod.
	// +optional
	PodFixed corev1.ResourceList `json:"podFixed,omitempty" protobuf:"bytes,1,opt,name=podFixed,casttype=k8s.io/api/core/v1.ResourceList,castkey=k8s.io/api/core/v1.ResourceName,castvalue=k8s.io/apimachinery/pkg/api/resource.Quantity"`
}

// Scheduling specifies the scheduling constraints for nodes supporting a
// RuntimeClass.
type Scheduling struct {
	// nodeSelector lists labels that must be present on nodes that support this
	// RuntimeClass. Pods using this RuntimeClass can only be scheduled to a
	// node matched by this selector. The RuntimeClass nodeSelector is merged
	// with a pod's existing nodeSelector. Any conflicts will cause the pod to
	// be rejected in admission.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty" protobuf:"bytes,1,opt,name=nodeSelector"`

	// tolerations are appended (excluding duplicates) to pods running with this
	// RuntimeClass during admission, effectively unioning the set of nodes
	// tolerated by the pod and the RuntimeClass.
	// +optional
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations,omitempty" protobuf:"bytes,2,rep,name=tolerations"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.13
// +k8s:prerelease-lifecycle-gen:deprecated=1.22

// RuntimeClassList is a list of RuntimeClass objects.
type RuntimeClassList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of schema objects.
	Items []RuntimeClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}
