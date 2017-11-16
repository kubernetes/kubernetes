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

// This file was collated from types used in:
// https://github.com/coreos/etcd-operator/tree/e7f18696bbdc127fa028a99ca8166a8519749328/pkg/apis/etcd/v1beta2.
// When kubeadm moves to its own repo and controls its own dependencies,
// this file will be no longer be needed.

package spec

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	// CRDResourceKind is the CRD resource kind
	CRDResourceKind = "EtcdCluster"
	// CRDResourcePlural is the CRD resource plural
	CRDResourcePlural = "etcdclusters"
	groupName         = "etcd.database.coreos.com"
)

var (
	// SchemeBuilder is a scheme builder
	SchemeBuilder = runtime.NewSchemeBuilder(AddKnownTypes)
	// AddToScheme adds to the scheme
	AddToScheme = SchemeBuilder.AddToScheme
	// SchemeGroupVersion is the scheme version
	SchemeGroupVersion = schema.GroupVersion{Group: groupName, Version: "v1beta2"}
	// CRDName is the name of the CRD
	CRDName = CRDResourcePlural + "." + groupName
)

// Resource gets an EtcdCluster GroupResource for a specified resource
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

// AddKnownTypes adds the set of types defined in this package to the supplied scheme.
func AddKnownTypes(s *runtime.Scheme) error {
	s.AddKnownTypes(SchemeGroupVersion,
		&EtcdCluster{},
		&EtcdClusterList{},
	)
	metav1.AddToGroupVersion(s, SchemeGroupVersion)
	return nil
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EtcdClusterList is a list of etcd clusters.
type EtcdClusterList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EtcdCluster `json:"items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EtcdCluster represents an etcd cluster
type EtcdCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              ClusterSpec `json:"spec"`
}

// ClusterSpec represents a cluster spec
type ClusterSpec struct {
	// Size is the expected size of the etcd cluster.
	// The etcd-operator will eventually make the size of the running
	// cluster equal to the expected size.
	// The vaild range of the size is from 1 to 7.
	Size int `json:"size"`

	// BaseImage is the base etcd image name that will be used to launch
	// etcd clusters. This is useful for private registries, etc.
	//
	// If image is not set, default is quay.io/coreos/etcd
	BaseImage string `json:"baseImage"`

	// Version is the expected version of the etcd cluster.
	// The etcd-operator will eventually make the etcd cluster version
	// equal to the expected version.
	//
	// The version must follow the [semver]( http://semver.org) format, for example "3.1.8".
	// Only etcd released versions are supported: https://github.com/coreos/etcd/releases
	//
	// If version is not set, default is "3.1.8".
	Version string `json:"version,omitempty"`

	// Paused is to pause the control of the operator for the etcd cluster.
	Paused bool `json:"paused,omitempty"`

	// Pod defines the policy to create pod for the etcd pod.
	//
	// Updating Pod does not take effect on any existing etcd pods.
	Pod *PodPolicy `json:"pod,omitempty"`

	// SelfHosted determines if the etcd cluster is used for a self-hosted
	// Kubernetes cluster.
	//
	// SelfHosted is a cluster initialization configuration. It cannot be updated.
	SelfHosted *SelfHostedPolicy `json:"selfHosted,omitempty"`

	// etcd cluster TLS configuration
	TLS *TLSPolicy `json:"TLS,omitempty"`
}

// PodPolicy defines the policy to create pod for the etcd container.
type PodPolicy struct {
	// Labels specifies the labels to attach to pods the operator creates for the
	// etcd cluster.
	// "app" and "etcd_*" labels are reserved for the internal use of the etcd operator.
	// Do not overwrite them.
	Labels map[string]string `json:"labels,omitempty"`

	// NodeSelector specifies a map of key-value pairs. For the pod to be eligible
	// to run on a node, the node must have each of the indicated key-value pairs as
	// labels.
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// AntiAffinity determines if the etcd-operator tries to avoid putting
	// the etcd members in the same cluster onto the same node.
	AntiAffinity bool `json:"antiAffinity,omitempty"`

	// Resources is the resource requirements for the etcd container.
	// This field cannot be updated once the cluster is created.
	Resources v1.ResourceRequirements `json:"resources,omitempty"`

	// Tolerations specifies the pod's tolerations.
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`

	// List of environment variables to set in the etcd container.
	// This is used to configure etcd process. etcd cluster cannot be created, when
	// bad environement variables are provided. Do not overwrite any flags used to
	// bootstrap the cluster (for example `--initial-cluster` flag).
	// This field cannot be updated.
	EtcdEnv []v1.EnvVar `json:"etcdEnv,omitempty"`

	// By default, kubernetes will mount a service account token into the etcd pods.
	// AutomountServiceAccountToken indicates whether pods running with the service account should have an API token automatically mounted.
	AutomountServiceAccountToken *bool `json:"automountServiceAccountToken,omitempty"`
}

// TLSPolicy defines the TLS policy of an etcd cluster
type TLSPolicy struct {
	// StaticTLS enables user to generate static x509 certificates and keys,
	// put them into Kubernetes secrets, and specify them into here.
	Static *StaticTLS `json:"static,omitempty"`
}

// StaticTLS represents static TLS
type StaticTLS struct {
	// Member contains secrets containing TLS certs used by each etcd member pod.
	Member *MemberSecret `json:"member,omitempty"`
	// OperatorSecret is the secret containing TLS certs used by operator to
	// talk securely to this cluster.
	OperatorSecret string `json:"operatorSecret,omitempty"`
}

// MemberSecret represents a member secret
type MemberSecret struct {
	// PeerSecret is the secret containing TLS certs used by each etcd member pod
	// for the communication between etcd peers.
	PeerSecret string `json:"peerSecret,omitempty"`
	// ServerSecret is the secret containing TLS certs used by each etcd member pod
	// for the communication between etcd server and its clients.
	ServerSecret string `json:"serverSecret,omitempty"`
}

// SelfHostedPolicy represents a self-hosted policy
type SelfHostedPolicy struct {
	// BootMemberClientEndpoint specifies a bootstrap member for the cluster.
	// If there is no bootstrap member, a completely new cluster will be created.
	// The boot member will be removed from the cluster once the self-hosted cluster
	// setup successfully.
	BootMemberClientEndpoint string `json:"bootMemberClientEndpoint,omitempty"`

	// SkipBootMemberRemoval specifies whether the removal of the bootstrap member
	// should be skipped. By default the operator will automatically remove the
	// bootstrap member from the new cluster - this happens during the pivot
	// procedure and is the first step of decommissioning the bootstrap member.
	// If unspecified, the default is `false`. If set to `true`, you are
	// expected to remove the boot member yourself from the etcd cluster.
	SkipBootMemberRemoval bool `json:"skipBootMemberRemoval,omitempty"`
}
