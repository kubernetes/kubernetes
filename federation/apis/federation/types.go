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

// ClusterSpec describes the attributes of a kubernetes cluster.
type ClusterSpec struct {
	// A map of client CIDR to server address.
	// This is to help clients reach servers in the most network-efficient way possible.
	// Clients can use the appropriate server address as per the CIDR that they match.
	// In case of multiple matches, clients should use the longest matching CIDR.
	ServerAddressByClientCIDRs []unversioned.ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" patchStrategy:"merge" patchMergeKey:"clientCIDR"`
	// the type (e.g. bearer token, client certificate etc) and data of the credential used to access cluster.
	// It’s used for system routines (not behalf of users)
	// TODO: string may not enough, https://github.com/kubernetes/kubernetes/pull/23847#discussion_r59301275
	Credential string `json:"credential,omitempty"`
}

type ClusterPhase string

// These are the valid phases of a cluster.
const (
	// Newly registered clusters or clusters suspended by admin for various reasons. They are not eligible for accepting workloads
	ClusterPending ClusterPhase = "pending"
	// Clusters in normal status that can accept workloads
	ClusterRunning ClusterPhase = "running"
	// Clusters temporarily down or not reachable
	ClusterOffline ClusterPhase = "offline"
	// Clusters removed from federation
	ClusterTerminated ClusterPhase = "terminated"
)

// Cluster metadata
type ClusterMeta struct {
	// Release version of the cluster.
	Version string `json:"version,omitempty"`
}

// ClusterStatus is information about the current status of a cluster updated by cluster controller peridocally.
type ClusterStatus struct {
	// Phase is the recently observed lifecycle phase of the cluster.
	Phase ClusterPhase `json:"phase,omitempty"`
	// Capacity represents the total resources of the cluster
	Capacity api.ResourceList `json:"capacity,omitempty"`
	// Allocatable represents the total resources of a cluster that are available for scheduling.
	Allocatable api.ResourceList `json:"allocatable,omitempty"`
	ClusterMeta `json:",inline"`
}

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
