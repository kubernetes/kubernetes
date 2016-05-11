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

package v1alpha1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
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
	ServerAddressByClientCIDRs []ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" patchStrategy:"merge" patchMergeKey:"clientCIDR" protobuf:"bytes,1,rep,name=serverAddressByClientCIDRs"`
	// the type (e.g. bearer token, client certificate etc) and data of the credential used to access cluster.
	// It’s used for system routines (not behalf of users)
	// TODO: string may not enough, https://github.com/kubernetes/kubernetes/pull/23847#discussion_r59301275
	Credential string `json:"credential,omitempty" protobuf:"bytes,2,opt,name=credential"`
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
	Type ClusterConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=ClusterConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/kubernetes/pkg/api/v1.ConditionStatus"`
	// Last time the condition was checked.
	LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty" protobuf:"bytes,3,opt,name=lastProbeTime"`
	// Last time the condition transit from one status to another.
	LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,5,opt,name=reason"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// Cluster metadata
type ClusterMeta struct {
	// Release version of the cluster.
	Version string `json:"version,omitempty" protobuf:"bytes,1,opt,name=version"`
}

// ClusterStatus is information about the current status of a cluster updated by cluster controller peridocally.
type ClusterStatus struct {
	// Conditions is an array of current cluster conditions.
	Conditions []ClusterCondition `json:"conditions,omitempty" protobuf:"bytes,1,rep,name=conditions"`
	// Capacity represents the total resources of the cluster
	Capacity v1.ResourceList `json:"capacity,omitempty" protobuf:"bytes,2,rep,name=capacity,casttype=k8s.io/kubernetes/pkg/api/v1.ResourceList,castkey=k8s.io/kubernetes/pkg/api/v1.ResourceName"`
	// Allocatable represents the total resources of a cluster that are available for scheduling.
	Allocatable v1.ResourceList `json:"allocatable,omitempty" protobuf:"bytes,3,rep,name=allocatable,casttype=k8s.io/kubernetes/pkg/api/v1.ResourceList,castkey=k8s.io/kubernetes/pkg/api/v1.ResourceName"`
	ClusterMeta `json:",inline" protobuf:"bytes,4,opt,name=clusterMeta"`
}

// +genclient=true,nonNamespaced=true

// Information about a registered cluster in a federated kubernetes setup. Clusters are not namespaced and have unique names in the federation.
type Cluster struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the behavior of the Cluster.
	Spec ClusterSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Status describes the current status of a Cluster
	Status ClusterStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// A list of all the kubernetes clusters registered to the federation
type ClusterList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of Cluster objects.
	Items []Cluster `json:"items" protobuf:"bytes,2,rep,name=items"`
}
