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

package apiregistration

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// APIServiceList is a list of APIService objects.
type APIServiceList struct {
	metav1.TypeMeta
	metav1.ListMeta

	Items []APIService
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// Namespace is the namespace of the service
	Namespace string
	// Name is the name of the service
	Name string
	// If specified, the port on the service that hosting the service.
	// Default to 443 for backward compatibility.
	// `port` should be a valid port number (1-65535, inclusive).
	// +optional
	Port int32
}

// APIServiceSpec contains information for locating and communicating with a server.
// Only https is supported, though you are able to disable certificate verification.
type APIServiceSpec struct {
	// Service is a reference to the service for this API server.  It must communicate
	// on port 443.
	// If the Service is nil, that means the handling for the API groupversion is handled locally on this server.
	// The call will simply delegate to the normal handler chain to be fulfilled.
	// +optional
	Service *ServiceReference
	// Group is the API group name this server hosts
	Group string
	// Version is the API version this server hosts.  For example, "v1"
	Version string

	// InsecureSkipTLSVerify disables TLS certificate verification when communicating with this server.
	// This is strongly discouraged.  You should use the CABundle instead.
	InsecureSkipTLSVerify bool
	// CABundle is a PEM encoded CA bundle which will be used to validate an API server's serving certificate.
	// If unspecified, system trust roots on the apiserver are used.
	// +listType=atomic
	// +optional
	CABundle []byte

	// GroupPriorityMinimum is the priority this group should have at least. Higher priority means that the group is preferred by clients over lower priority ones.
	// Note that other versions of this group might specify even higher GroupPriorityMinimum values such that the whole group gets a higher priority.
	// The primary sort is based on GroupPriorityMinimum, ordered highest number to lowest (20 before 10).
	// The secondary sort is based on the alphabetical comparison of the name of the object.  (v1.bar before v1.foo)
	// We'd recommend something like: *.k8s.io (except extensions) at 18000 and
	// PaaSes (OpenShift, Deis) are recommended to be in the 2000s
	GroupPriorityMinimum int32

	// VersionPriority controls the ordering of this API version inside of its group.  Must be greater than zero.
	// The primary sort is based on VersionPriority, ordered highest to lowest (20 before 10).
	// Since it's inside of a group, the number can be small, probably in the 10s.
	// In case of equal version priorities, the version string will be used to compute the order inside a group.
	// If the version string is "kube-like", it will sort above non "kube-like" version strings, which are ordered
	// lexicographically. "Kube-like" versions start with a "v", then are followed by a number (the major version),
	// then optionally the string "alpha" or "beta" and another number (the minor version). These are sorted first
	// by GA > beta > alpha (where GA is a version with no suffix such as beta or alpha), and then by comparing major
	// version, then minor version. An example sorted list of versions:
	// v10, v2, v1, v11beta2, v10beta3, v3beta1, v12alpha1, v11alpha2, foo1, foo10.
	VersionPriority int32
}

// ConditionStatus indicates the status of a condition (true, false, or unknown).
type ConditionStatus string

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition;
// "ConditionFalse" means a resource is not in the condition; "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

// APIServiceConditionType is a valid value for APIServiceCondition.Type
type APIServiceConditionType string

const (
	// Available indicates that the service exists and is reachable
	Available APIServiceConditionType = "Available"
)

// APIServiceCondition describes conditions for an APIService
type APIServiceCondition struct {
	// Type is the type of the condition.
	Type APIServiceConditionType
	// Status is the status of the condition.
	// Can be True, False, Unknown.
	Status ConditionStatus
	// Last time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time
	// Unique, one-word, CamelCase reason for the condition's last transition.
	Reason string
	// Human-readable message indicating details about last transition.
	Message string
}

// APIServiceStatus contains derived information about an API server
type APIServiceStatus struct {
	// Current service state of apiService.
	// +listType=map
	// +listMapKey=type
	Conditions []APIServiceCondition
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// APIService represents a server for a particular GroupVersion.
// Name must be "version.group".
type APIService struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec contains information for locating and communicating with a server
	Spec APIServiceSpec
	// Status contains derived information about an API server
	Status APIServiceStatus
}
