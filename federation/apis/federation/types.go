/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package federation

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// ServerAddressByClientCIDR helps the client to determine the server address that they should use, depending on the clientCIDR that they match.
type ServerAddressByClientCIDR struct {
	// The CIDR with which clients can match their IP to figure out the server address that they should use.
	ClientCIDR string `json:"clientCIDR" protobuf:"bytes,1,opt,name=clientCIDR"`
	// Address of this server, suitable for a client that matches the above CIDR.
	// This can be a hostname, hostname:port, IP or IP:port.
	ServerAddress string `json:"serverAddress" protobuf:"bytes,2,opt,name=serverAddress"`
}

// ClusterSpec describes the attributes of a kubernetes cluster.
type ClusterSpec struct {
	// A map of client CIDR to server address.
	// This is to help clients reach servers in the most network-efficient way possible.
	// Clients can use the appropriate server address as per the CIDR that they match.
	// In case of multiple matches, clients should use the longest matching CIDR.
	ServerAddressByClientCIDRs []ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" patchStrategy:"merge" patchMergeKey:"clientCIDR"`
	// Name of the secret containing kubeconfig to access this cluster.
	// The secret is read from the kubernetes cluster that is hosting federation control plane.
	// Admin needs to ensure that the required secret exists. Secret should be in the same namespace where federation control plane is hosted and it should have kubeconfig in its data with key "kubeconfig".
	// This will later be changed to a reference to secret in federation control plane when the federation control plane supports secrets.
	SecretRef *api.LocalObjectReference `json:"secretRef"`
}

type ClusterConditionType string

// These are valid conditions of a cluster.
const (
	// ClusterReady means the cluster is ready to accept workloads.
	ClusterReady ClusterConditionType = "Ready"
	// ClusterOffline means the cluster is temporarily down or not reachable
	ClusterOffline ClusterConditionType = "Offline"
)

// ClusterCondition describes current state of a cluster.
type ClusterCondition struct {
	// Type of cluster condition, Complete or Failed.
	Type ClusterConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status api.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transit from one status to another.
	LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}

// Cluster metadata
type ClusterMeta struct {
	// Release version of the cluster.
	Version string `json:"version,omitempty"`
}

// ClusterStatus is information about the current status of a cluster updated by cluster controller peridocally.
type ClusterStatus struct {
	// Conditions is an array of current cluster conditions.
	Conditions []ClusterCondition `json:"conditions,omitempty"`
	// Capacity represents the total resources of the cluster
	Capacity api.ResourceList `json:"capacity,omitempty"`
	// Allocatable represents the total resources of a cluster that are available for scheduling.
	Allocatable api.ResourceList `json:"allocatable,omitempty"`
	ClusterMeta `json:",inline"`
	// Zones is the list of avaliability zones in which the nodes of the cluster exist, e.g. 'us-east1-a'.
	// These will always be in the same region.
	Zones []string `json:"zones,omitempty"`
	// Region is the name of the region in which all of the nodes in the cluster exist.  e.g. 'us-east1'.
	Region string `json:"region,omitempty"`
}

// +genclient=true,nonNamespaced=true

// Information about a registered cluster in a federated kubernetes setup. Clusters are not namespaced and have unique names in the federation.
type Cluster struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of the Cluster.
	Spec ClusterSpec `json:"spec,omitempty"`
	// Status describes the current status of a Cluster
	Status ClusterStatus `json:"status,omitempty"`
}

// A list of all the kubernetes clusters registered to the federation
type ClusterList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`

	// List of Cluster objects.
	Items []Cluster `json:"items"`
}
