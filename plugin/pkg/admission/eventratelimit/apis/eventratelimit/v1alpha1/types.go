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

package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// LimitType is the type of the limit (e.g., per-namespace)
type LimitType string

const (
	// ServerLimitType is a type of limit where there is one bucket shared by
	// all of the event queries received by the API Server.
	ServerLimitType LimitType = "Server"
	// NamespaceLimitType is a type of limit where there is one bucket used by
	// each namespace
	NamespaceLimitType LimitType = "Namespace"
	// UserLimitType is a type of limit where there is one bucket used by each
	// user
	UserLimitType LimitType = "User"
	// SourceAndObjectLimitType is a type of limit where there is one bucket used
	// by each combination of source and involved object of the event.
	SourceAndObjectLimitType LimitType = "SourceAndObject"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Configuration provides configuration for the EventRateLimit admission
// controller.
type Configuration struct {
	metav1.TypeMeta `json:",inline"`

	// limits are the limits to place on event queries received.
	// Limits can be placed on events received server-wide, per namespace,
	// per user, and per source+object.
	// At least one limit is required.
	Limits []Limit `json:"limits"`
}

// Limit is the configuration for a particular limit type
type Limit struct {
	// type is the type of limit to which this configuration applies
	Type LimitType `json:"type"`

	// qps is the number of event queries per second that are allowed for this
	// type of limit. The qps and burst fields are used together to determine if
	// a particular event query is accepted. The qps determines how many queries
	// are accepted once the burst amount of queries has been exhausted.
	QPS int32 `json:"qps"`

	// burst is the burst number of event queries that are allowed for this type
	// of limit. The qps and burst fields are used together to determine if a
	// particular event query is accepted. The burst determines the maximum size
	// of the allowance granted for a particular bucket. For example, if the burst
	// is 10 and the qps is 3, then the admission control will accept 10 queries
	// before blocking any queries. Every second, 3 more queries will be allowed.
	// If some of that allowance is not used, then it will roll over to the next
	// second, until the maximum allowance of 10 is reached.
	Burst int32 `json:"burst"`

	// cacheSize is the size of the LRU cache for this type of limit. If a bucket
	// is evicted from the cache, then the allowance for that bucket is reset. If
	// more queries are later received for an evicted bucket, then that bucket
	// will re-enter the cache with a clean slate, giving that bucket a full
	// allowance of burst queries.
	//
	// The default cache size is 4096.
	//
	// If limitType is 'server', then cacheSize is ignored.
	// +optional
	CacheSize int32 `json:"cacheSize,omitempty"`
}
