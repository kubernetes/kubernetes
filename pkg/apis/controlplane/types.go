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

package controlplane

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Address of a cluster
type ClusterAddress struct {
	// URL to access the cluster
	Url string `json:"url"`
}

// ClusterSpec describes the attributes on a Cluster.
type ClusterSpec struct {
	// Address of the cluster
	Address ClusterAddress `json:"address"`
	// The credential used to access cluster. It’s used for system routines (not behalf of users)
	Credential string `json:"credential",omitempty`
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
	// Version of the cluster
	Version string `json:"version,omitempty"`
}

// ClusterStatus is information about the current status of a cluster.
type ClusterStatus struct {
	// Phase is the recently observed lifecycle phase of the cluster.
	Phase ClusterPhase `json:"phase,omitempty"`
	// Capacity represents the total resources of the cluster
	Capacity    api.ResourceList `json:"capacity,omitempty"`
	ClusterMeta string           `json:",inline"`
}

// +genclient=true,nonNamespaced=true

// Cluster information in Ubernetes
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

// A list of Clusters
type ClusterList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`

	// List of Cluster objects.
	Items []Cluster `json:"items"`
}

type ClusterSelector struct {
       //Required. A list of node selector terms. The terms are ORed.
       ClusterSelectorTerms []ClusterSelectorTerm `json:"clusterSelectorTerms"`
}

type ClusterSelectorTerm struct {
       //Required. A list of node selector requirements. The requirements are ANDed.
       MatchExpressions []ClusterSelectorRequirement `json:"matchExpressions"`
}

type ClusterSelectorRequirement struct {
       // The label key that the selector applies to.
       Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key"`
       // Represents a key's relationship to a set of values.
       // Valid operators are In, NotIn, Exists, DoesNotExist. Gt, and Lt.
       Operator ClusterSelectorOperator `json:"operator"`
       // An array of string values. If the operator is In or NotIn,
       // the values array must be non-empty. If the operator is Exists or DoesNotExist,
       // the values array must be empty. If the operator is Gt or Lt, the values
       // array must have a single element, which will be interpreted as an integer.
       // This array is replaced during a strategic merge patch.
       Values []string `json:"values,omitempty"`
}

type ClusterSelectorOperator string

const (
       ClusterSelectorOpIn        ClusterSelectorOperator = "In"
       ClusterSelectorOpNotIn     ClusterSelectorOperator = "NotIn"
)
